"""Trained MaskablePPO policy behind the common Policy interface (simulation and live service).

The observation is built from the State JSON by core/observation.py, the same builder used during
training, and the action mask is the clinical mask plus the cue-only-after-error rule, exactly as
in the training environment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from agents.selection import clinical_gate
from core.config import ClinicalConfig, resolve
from core.masks import ACTION_CUES, ACTION_TIERS
from core.observation import ObservationBuilder
from core.schema import Action, Outcome, State
from envs.aphasia_env import EnvConfig


class PPOPolicy:
    name = "ppo"

    def __init__(self, model: str | Path | MaskablePPO, clinical: ClinicalConfig, env_cfg: EnvConfig,
                 deterministic: bool = True) -> None:
        # Accepts a path or an already-loaded model, so the service can share one model across
        # sessions (the policy keeps no per-session state) and hot-swap a tuned copy for one session.
        self.model = model if isinstance(model, MaskablePPO) else MaskablePPO.load(str(resolve(model)), device="cpu")
        self.clinical, self.env_cfg, self.deterministic = clinical, env_cfg, deterministic
        self.obs = ObservationBuilder(clinical, env_cfg.history, env_cfg.nominal_trials, env_cfg.rt_cap_s)

    def act(self, state: State, mask: np.ndarray) -> Action:
        allowed = clinical_gate(state, mask, self.env_cfg.cue_only_after_error)
        a, _ = self.model.predict(self.obs(state), action_masks=allowed, deterministic=self.deterministic)
        i = int(a)
        return Action(tier=int(ACTION_TIERS[i]), cue=ACTION_CUES[i],
                      rationale=f"Trained policy (MaskablePPO): tier {int(ACTION_TIERS[i])}, cue '{ACTION_CUES[i]}'.")

    def update(self, state: State, action: Action, outcome: Outcome) -> None:
        pass   # the trained policy adapts through its observation history, not by online updates
