"""Live DITTO loop for the agent service: clinician overrides in, tuned policy swapped in live.

    dashboard override  --POST /override-->  runs/overrides.jsonl   (demonstrations)
    agent choice shown, not overridden --POST /update-->  runs/accepted.jsonl   (anchors)
    POST /ditto/train   --> DITTO on the BASE policy with all demonstrations so far
                        --> the tuned policy replaces the live one (agent == "ppo")
    POST /ditto/reset   --> back to the base policy

Training always starts from the base policy (never from the previously tuned one), so calling
/ditto/train twice with the same overrides gives the same policy instead of compounding updates.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from sb3_contrib import MaskablePPO

from agents.ditto import action_probs, ditto
from agents.ppo import PPOPolicy
from agents.selection import clinical_gate
from core.config import ClinicalConfig, load_yaml, resolve
from core.masks import ACTION_CUES, ACTION_TIERS, action_mask
from core.observation import ObservationBuilder
from core.schema import Action, State
from envs.aphasia_env import EnvConfig, load_env_config


def _describe(i: int) -> dict:
    return {"tier": int(ACTION_TIERS[i]), "cue": ACTION_CUES[i]}


class LiveDitto:
    """Holds the base and the currently served PPO policy; thread-safe swap."""

    def __init__(self, base_model_path: str, ditto_config: str, clinical: ClinicalConfig,
                 env_cfg: EnvConfig | None = None) -> None:
        self.clinical, self.env_cfg = clinical, env_cfg or load_env_config()
        self.cfg = load_yaml(ditto_config)
        self.base_path = base_model_path
        self.policy = PPOPolicy(base_model_path, clinical, self.env_cfg)    # the object every session shares
        self.base_model: MaskablePPO = self.policy.model
        self.obs = ObservationBuilder(clinical, self.env_cfg.history, self.env_cfg.nominal_trials, self.env_cfg.rt_cap_s)
        self.tuned = False
        self.last_report: dict | None = None
        self._sim_states: list[dict] | None = None
        self._lock = threading.Lock()

    # ---- anchors ----
    def _mask(self, state: State) -> np.ndarray:
        m = action_mask(state.current_tier, self.clinical.approved_cues, self.clinical.max_tier_step)
        return clinical_gate(state, m, self.env_cfg.cue_only_after_error)

    def _simulated_states(self) -> list[dict]:
        if self._sim_states is None:
            rng = self.cfg.get("live", {}).get("sim_anchor_patients") or [0, 0]
            if rng[1] > rng[0]:
                from envs.aphasia_env import AphasiaEnv
                from train.ditto_ppo import collect_states
                self._sim_states = collect_states(self.base_model, AphasiaEnv(cohort="population", split="val"),
                                                  range(*rng))
            else:
                self._sim_states = []
        return self._sim_states

    def anchors(self, accepted: list[dict], override_records: list[dict]) -> tuple[list, dict]:
        real = []
        for rec in accepted:
            st = State(**rec["state"])
            real.append((self.obs(st), self._mask(st)))
        # Leave out simulated states in the same context as an override (same last error type, or
        # "no error" if the override followed a correct answer): the preference may apply there.
        contexts = set()
        for rec in override_records:
            trials = rec["state"].get("last_trials") or []
            contexts.add(trials[-1]["error_type"] if trials else None)
        sim = [(s["obs"], s["mask"]) for s in self._simulated_states() if s["last_error"] not in contexts]
        return real + sim, {"accepted": len(real), "simulated": len(sim)}

    # ---- train / reset ----
    def train(self, override_records: list[dict], accepted: list[dict], variant: str | None = None) -> dict:
        from train.ditto_ppo import demos_from_records
        demos, skipped = demos_from_records(override_records, self.clinical, self.env_cfg)
        if not demos:
            raise ValueError(f"no usable overrides ({len(skipped)} skipped: the policy cannot produce that "
                             f"action under the clinical mask)" if skipped else "no overrides logged yet")
        anchors, n_anchor = self.anchors(accepted, override_records)
        c = self.cfg
        variant = variant or c.get("live", {}).get("variant", "paper")
        kw = dict(anchors=anchors, anchor_weight=c["anchor_weight"])
        if variant == "paper":
            p = c["paper"]
            kw.update(sft_steps=p["sft_steps"], mixture=p["mixture"], batch_size=p["batch_size"])
        elif variant != "legacy":
            raise ValueError(f"unknown variant {variant!r} (paper | legacy)")
        t0 = time.time()
        tuned, log = ditto(self.base_model, demos, c["rounds"], c["samples_per_state"], c["beta"],
                           c["learning_rate"], c["gradient_steps_per_round"], c["seed"], **kw)
        obs, masks = np.stack([d.obs for d in demos]), np.stack([d.mask for d in demos])
        before = action_probs(self.base_model, obs, masks)
        after = action_probs(tuned, obs, masks)
        per_override = [{"clinician": _describe(d.action), "base_choice": _describe(int(before[k].argmax())),
                         "tuned_choice": _describe(int(after[k].argmax())),
                         "p_clinician_before": round(float(before[k, d.action]), 3),
                         "p_clinician_after": round(float(after[k, d.action]), 3)}
                        for k, d in enumerate(demos)]
        report = {"variant": variant, "n_overrides_used": len(demos), "n_overrides_skipped": len(skipped),
                  "n_anchors": n_anchor, "seconds": round(time.time() - t0, 2),
                  "agreement_before": float(np.mean([before[k].argmax() == d.action for k, d in enumerate(demos)])),
                  "agreement_after": float(np.mean([after[k].argmax() == d.action for k, d in enumerate(demos)])),
                  "rounds": log, "per_override": per_override}
        out = resolve("runs/ppo_ditto_live.zip")
        out.parent.mkdir(parents=True, exist_ok=True)
        tuned.save(str(out), exclude=["tensorboard_log"])
        report["saved_to"] = str(out.relative_to(resolve(".")))
        with self._lock:
            self.policy.model = tuned
            self.tuned = True
            self.last_report = report
        return report

    def reset(self) -> None:
        with self._lock:
            self.policy.model = self.base_model
            self.tuned = False

    def label(self) -> str:
        return "ppo+ditto" if self.tuned else "ppo"


def same_action(a: Action, b: Action) -> bool:
    return a.tier == b.tier and a.cue == b.cue
