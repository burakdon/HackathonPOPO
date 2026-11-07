"""Gymnasium environment: one episode is one practice session with one simulated patient.

POMDP: the patient's parameters, item learning and fatigue are hidden. The observation is built
from the State JSON only (core/observation.py), with a history stack for memory. Clinical
constraints are action masks (`action_masks()`, used by sb3-contrib MaskablePPO): the tier moves at
most one step per trial, cues outside the approved set are never offered, and (if configured) a
cue is only offered right after an error. The reward is the observable proxy (core/reward.py);
the true objective (learning gain) is reported in `info` at the end of the episode.

Each reset draws a fresh patient. Training uses the broad `population` cohort on the `train` split;
evaluation and learning curves pass a cohort, a split and an explicit patient index.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field

from agents.selection import clinical_gate
from core.config import ClinicalConfig, load_clinical, load_yaml
from core.masks import ACTION_CUES, ACTION_TIERS, N_ACTIONS, action_mask
from core.observation import ObservationBuilder
from core.reward import TERMS, ProxyReward, RewardConfig, load_reward
from core.schema import MAX_LAST_TRIALS, State
from sim.config import SimConfig, load_sim
from sim.items import load_items
from sim.population import make_patients
from sim.session import SessionResult, play_probe


class EnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    history: int = Field(ge=1)
    nominal_trials: int = Field(ge=1)
    rt_cap_s: float = Field(gt=0.0)
    cue_only_after_error: bool


def load_env_config(path: str | Path = "configs/env.yaml") -> EnvConfig:
    return EnvConfig(**load_yaml(path))


class AphasiaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cohort: str = "population", split: str = "train", root_seed: int = 0,
                 sim_cfg: SimConfig | None = None, clinical: ClinicalConfig | None = None,
                 reward_cfg: RewardConfig | None = None, env_cfg: EnvConfig | None = None,
                 strict_masks: bool = True) -> None:
        """strict_masks=True raises on a masked action (catches agent bugs). False maps it to the
        nearest allowed action and flags info["invalid_action"]; only for generic API checkers that
        sample actions without masks (gymnasium's check_env)."""
        super().__init__()
        self.strict_masks = strict_masks
        self.sim_cfg = sim_cfg or load_sim()
        self.clinical = clinical or load_clinical()
        self.env_cfg = env_cfg or load_env_config()
        self.reward = ProxyReward(reward_cfg or load_reward(), self.clinical)
        self.cohort, self.split, self.root_seed = cohort, split, root_seed
        self.bank = load_items(self.sim_cfg.items)
        self.obs_builder = ObservationBuilder(self.clinical, self.env_cfg.history,
                                              self.env_cfg.nominal_trials, self.env_cfg.rt_cap_s)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_builder.dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

    # ---- helpers ----
    def _state(self) -> State:
        return State(session_id="env", last_trials=self.history[-MAX_LAST_TRIALS:], current_tier=self.tier,
                     trial_count=len(self.result.trace), session_minutes=round(self.minutes, 2))

    def action_masks(self) -> np.ndarray:
        mask = action_mask(self.tier, self.clinical.approved_cues, self.clinical.max_tier_step)
        return clinical_gate(self._state(), mask, self.env_cfg.cue_only_after_error)

    # ---- gym API ----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        k = int(options["patient_index"]) if "patient_index" in options else int(self.np_random.integers(2**31 - 1))
        spec = make_patients(self.sim_cfg, self.cohort, self.split, 1, self.root_seed, start=k)[0]
        self.patient, self.selector = spec.build(self.bank, self.sim_cfg, self.clinical)
        self.patient_index = k
        self.tier = self.sim_cfg.session.start_tier
        self.history: list[dict] = []
        self.minutes = 0.0
        self.result = SessionResult(start_retrieval=self.patient.mean_retrieval(), start_tier=self.tier)
        return self.obs_builder(self._state()), {"patient_index": k}

    def step(self, action: int):
        mask = self.action_masks()
        invalid = not mask[int(action)]
        if invalid:
            if self.strict_masks:
                raise ValueError(f"masked action {action} at tier {self.tier}")
            # Nearest allowed action: closest tier, then the requested cue if allowed, else none.
            want_t, want_c = int(ACTION_TIERS[action]), ACTION_CUES[action]
            allowed = np.flatnonzero(mask)
            action = int(min(allowed, key=lambda i: (abs(int(ACTION_TIERS[i]) - want_t), ACTION_CUES[i] != want_c,
                                                     ACTION_CUES[i] != "none")))
        tier, cue = int(ACTION_TIERS[action]), ACTION_CUES[action]
        retrieval_before = self.patient.mean_retrieval()
        item = self.selector.next(tier)
        p_tier = self.patient.p_correct_tier(tier, cue)
        r = self.patient.respond(item, cue)
        observable = {"word": self.bank.words[item], "tier": tier, "cue": cue, "outcome": r.correct,
                      "error_type": r.error_type, "response_time": r.response_time, "probe": False}
        self.history.append(observable)
        self.result.trace.append({**observable, "trial": len(self.result.trace), "p_true": r.p_correct,
                                  "p_tier": p_tier, "fatigue": self.patient.fatigue})
        self.tier = tier
        self.minutes += (r.response_time + self.sim_cfg.response_time.overhead_s) / 60.0
        components = self.reward.step(self.result.trace)
        terms, w = self.reward.cfg.terms, self.reward.cfg.privileged
        if terms.privileged:
            components["control"] = -w.control_weight * abs(r.p_correct - self.clinical.target)
        quit_ = r.quit
        n_max = self.sim_cfg.session.n_trials

        # Review probes scheduled right after this trial are played before the agent's next decision;
        # their (observable) reward terms are credited to this step.
        while not quit_ and len(self.result.trace) < n_max and self.selector.probe_due():
            observable, record, quit_ = play_probe(self.patient, self.selector, len(self.result.trace))
            self.history.append(observable)
            self.result.trace.append(record)
            self.minutes += (observable["response_time"] + self.sim_cfg.response_time.overhead_s) / 60.0
            for key, v in self.reward.step(self.result.trace).items():
                components[key] += v
            if terms.privileged:
                components["probe_p"] += w.probe_weight * 100.0 * record["probe_lift"]
        if terms.privileged:
            components["learning"] = w.learning_weight * 100.0 * (self.patient.mean_retrieval() - retrieval_before)

        terminated = quit_ or len(self.result.trace) >= n_max
        info: dict = {"reward_terms": components, "invalid_action": invalid}
        if terminated:
            self.result.quit = quit_
            self.result.end_retrieval = self.patient.mean_retrieval()
            final = self.reward.terminal(self.result.trace, self.result.start_tier)
            components = {k: components[k] + final[k] for k in TERMS}
            info.update(reward_terms=components, learning_gain=self.result.learning_gain,
                        time_in_band=self.reward.time_in_band(self.result.trace), result=self.result)
        return self.obs_builder(self._state()), float(sum(components.values())), terminated, False, info
