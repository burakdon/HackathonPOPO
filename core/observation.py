"""Fixed-size observation vector for neural policies, computed from the State JSON only.

Layout (all values in [0, 1]):
  history block: the last `history` trials, most recent last, zero-padded at the start; per trial
                 [present, tier one-hot (3), cue one-hot (5), error type one-hot (4), response time,
                  review-probe flag]
  current block: [current tier one-hot (3), rolling accuracy, trials in window / window,
                  fail streak / 3 (capped), session progress]

Because it reads only the State JSON, a policy trained in simulation sees exactly the same inputs
when it runs live behind the service.
"""
from __future__ import annotations

import numpy as np

from core.config import ClinicalConfig
from core.features import context
from core.schema import CUES, ERROR_TYPES, TIERS, State

PER_TRIAL = 1 + len(TIERS) + len(CUES) + len(ERROR_TYPES) + 1 + 1
CURRENT = len(TIERS) + 4


class ObservationBuilder:
    def __init__(self, clinical: ClinicalConfig, history: int, nominal_trials: int, rt_cap_s: float) -> None:
        self.clinical, self.history, self.nominal_trials, self.rt_cap_s = clinical, history, nominal_trials, rt_cap_s
        self.dim = history * PER_TRIAL + CURRENT

    def __call__(self, state: State) -> np.ndarray:
        obs = np.zeros(self.dim, dtype=np.float32)
        trials = state.last_trials[-self.history:]
        offset = (self.history - len(trials)) * PER_TRIAL
        for j, t in enumerate(trials):
            base = offset + j * PER_TRIAL
            obs[base] = 1.0
            obs[base + 1 + TIERS.index(t.tier)] = 1.0
            obs[base + 1 + len(TIERS) + CUES.index(t.cue)] = 1.0
            obs[base + 1 + len(TIERS) + len(CUES) + ERROR_TYPES.index(t.error_type)] = 1.0
            obs[base + PER_TRIAL - 2] = min(t.response_time, self.rt_cap_s) / self.rt_cap_s
            obs[base + PER_TRIAL - 1] = float(t.probe)
        ctx = context(state, self.clinical.window, self.nominal_trials)
        c = self.history * PER_TRIAL
        obs[c + TIERS.index(state.current_tier)] = 1.0
        obs[c + 3] = 0.0 if ctx.rolling_acc is None else ctx.rolling_acc
        obs[c + 4] = ctx.n_recent / self.clinical.window
        obs[c + 5] = min(ctx.fail_streak, 3) / 3.0
        obs[c + 6] = ctx.progress
        return obs
