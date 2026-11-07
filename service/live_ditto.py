"""Live DITTO-style update of one session's PPO policy from the overrides logged in that session.

Used by POST /learn_from_overrides. One round of agents/ditto.py against the frozen base model:
demonstrations = this session's clinician overrides; anchors = this session's accepted choices (the
agent acted and the clinician did not override) plus the base policy's own decisions on a few
validation patients (collected once and cached), excluding states of the types the clinician
overrode. Sized to run in a few seconds on CPU.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from agents.ditto import Demo, action_probs, ditto
from agents.selection import clinical_gate
from core.config import ClinicalConfig, LiveDittoConfig
from core.masks import ACTION_CUES, action_index, action_mask
from core.observation import ObservationBuilder
from core.schema import Action, State, StateTypeChange

CUE_OF = np.array(ACTION_CUES)


def state_type(last_error: str | None) -> str:
    if last_error is None:
        return "at the start of a session"
    if last_error == "correct":
        return "after a correct answer"
    return "after a no-response" if last_error == "no_response" else f"after a {last_error} error"


class LiveDitto:
    def __init__(self, base: MaskablePPO, clinical: ClinicalConfig, env_cfg, cfg: LiveDittoConfig) -> None:
        self.base, self.clinical, self.env_cfg, self.cfg = base, clinical, env_cfg, cfg
        self.obs = ObservationBuilder(clinical, env_cfg.history, env_cfg.nominal_trials, env_cfg.rt_cap_s)
        self._pool: list[dict] | None = None

    def pool(self) -> list[dict]:
        """The base policy's decision states on validation patients (cached after the first call)."""
        if self._pool is None:
            from envs.aphasia_env import AphasiaEnv
            from train.ditto_ppo import collect_states
            env = AphasiaEnv(cohort="population", split="val")
            self._pool = collect_states(self.base, env, range(self.cfg.anchor_pool_patients))
        return self._pool

    def encode(self, state: State) -> tuple[np.ndarray, np.ndarray]:
        mask = action_mask(state.current_tier, self.clinical.approved_cues, self.clinical.max_tier_step)
        return self.obs(state), clinical_gate(state, mask, self.env_cfg.cue_only_after_error)

    def learn(self, overrides: list[tuple[State, Action]], accepted: list[State]) -> tuple[MaskablePPO, dict]:
        t0 = time.time()
        demos, types, skipped = [], defaultdict(list), 0
        for state, a in overrides:
            obs, mask = self.encode(state)
            i = action_index(a.tier, a.cue)
            if not mask[i]:
                skipped += 1
                continue
            demos.append(Demo(obs, mask, i))
            last = state.last_trials[-1].error_type if state.last_trials else None
            types[(state_type(last), a.cue)].append(i)
        if not demos:
            raise ValueError("no usable overrides in this session")
        pool = self.pool()
        # A state type the clinician has overridden (e.g. "after a sound error") no longer counts as
        # accepted: anchors keep only states of other types, so the preference can generalise within
        # its type while everything else stays pinned to the base policy.
        overridden_types = {t for t, _ in types}
        last_type = lambda s: state_type(s.last_trials[-1].error_type if s.last_trials else None)
        anchors = ([self.encode(s) for s in accepted if last_type(s) not in overridden_types]
                   + [(p["obs"], p["mask"]) for p in pool if state_type(p["last_error"]) not in overridden_types])
        c = self.cfg
        tuned, _ = ditto(self.base, demos, c.rounds, c.samples_per_state, c.beta, c.learning_rate,
                         c.gradient_steps, c.seed, anchors=anchors, anchor_weight=c.anchor_weight)

        by_type = []
        for (stype, cue), idx in types.items():
            sel = [p for p in pool if state_type(p["last_error"]) == stype
                   and p["mask"][[action_index(t, cue) for t in (1, 2, 3)]].any()]
            if sel:
                o, m = np.stack([p["obs"] for p in sel]), np.stack([p["mask"] for p in sel])
                is_cue = CUE_OF == cue
                before = float(action_probs(self.base, o, m)[:, is_cue].sum(1).mean())
                after = float(action_probs(tuned, o, m)[:, is_cue].sum(1).mean())
            else:
                before = after = float("nan")
            by_type.append(StateTypeChange(state_type=stype, clinician_cue=cue, n_overrides=len(idx),
                                           n_eval_states=len(sel), p_before=before, p_after=after))
        others = [p for p in pool if state_type(p["last_error"]) not in overridden_types]
        if others:
            o, m = np.stack([p["obs"] for p in others]), np.stack([p["mask"] for p in others])
            unchanged = float(np.mean(action_probs(self.base, o, m).argmax(1) == action_probs(tuned, o, m).argmax(1)))
        else:
            unchanged = float("nan")
        return tuned, {"n_overrides": len(demos), "n_skipped": skipped, "n_anchor_states": len(anchors),
                       "seconds": round(time.time() - t0, 2), "by_state_type": by_type,
                       "other_states_unchanged": unchanged}
