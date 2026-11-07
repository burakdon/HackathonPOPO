"""Contextual Thompson sampling with a Bayesian logistic model of p(correct).

Model:   p(correct | state, action) = sigmoid(w . phi(state, action)),   w ~ N(mu, Sigma)

phi has three blocks (17 features):
  action       bias, tier 2, tier 3, one-hot of the four non-'none' cues
  context      rolling accuracy minus target, fail streak, session progress, recent response time
  interaction  last error type (sound / meaning / no response) x cue family (phonemic / semantic):
               lets the bandit learn, for example, that this patient's sound errors respond to
               phonemic cues.
Sharing weights across actions is the key difference from the 15 independent Beta arms of the
scaffold: one success on tier 3 also informs every other tier-3 action.

Posterior: online Laplace / extended-Kalman update for logistic regression (rank-one, closed form):
  Sigma' = Sigma - w (Sigma x)(Sigma x)^T / (1 + w x^T Sigma x),  w = p(1-p)
  mu'    = mu + Sigma' x (y - p)
Non-stationarity (fatigue, within-session learning) is handled by adding process noise q*I to
Sigma before every update (random-walk weights), so old evidence slowly loses precision.

Action choice: sample w once, predict p for all 15 actions, pick with the shared progression rule
(agents/selection.py). Standard normal draws have a fixed size, so the random stream consumed per
call does not depend on the state; same seed + same inputs gives the same actions.
"""
from __future__ import annotations

import numpy as np

from agents.selection import clinical_gate, describe, select_action
from core.config import ClinicalConfig, LogisticBanditConfig, load_yaml
from core.features import context
from core.masks import ACTION_CUES, ACTION_TIERS, N_ACTIONS, action_index
from core.schema import Action, Outcome, State

ACTION_FEATURES = ["bias", "tier2", "tier3", "cue_first_sound", "cue_first_letter",
                   "cue_sentence_frame", "cue_semantic"]
CONTEXT_FEATURES = ["acc_minus_target", "fail_streak", "progress", "rt"]
INTERACTION_FEATURES = [f"last_{e}_x_{fam}" for e in ("sound", "meaning", "no_response")
                        for fam in ("phonemic", "semantic")]
FEATURES = ACTION_FEATURES + CONTEXT_FEATURES + INTERACTION_FEATURES
N_FEATURES = len(FEATURES)

_PHONEMIC = {"first_sound", "first_letter"}
_SEMANTIC = {"semantic", "sentence_frame"}


def _action_block() -> np.ndarray:
    """(15, 7) action features, constant across states."""
    rows = []
    for t, c in zip(ACTION_TIERS, ACTION_CUES):
        rows.append([1.0, float(t == 2), float(t == 3)]
                    + [float(c == k) for k in ("first_sound", "first_letter", "sentence_frame", "semantic")])
    return np.array(rows)


_ACTIONS_X = _action_block()


class LogisticThompsonBandit:
    name = "logistic_bandit"

    def __init__(self, cfg: LogisticBanditConfig, clinical: ClinicalConfig, seed: int) -> None:
        self.cfg, self.clinical = cfg, clinical
        self.rng = np.random.default_rng(seed)
        self.mu, self.Sigma = self.prior(cfg)
        self._strength = np.array([clinical.cue_strength[c] for c in ACTION_CUES], dtype=float)

    @classmethod
    def prior(cls, cfg: LogisticBanditConfig) -> tuple[np.ndarray, np.ndarray]:
        """Fitted empirical-Bayes prior if `prior_file` is set, else the hand-set weak prior."""
        if cfg.prior_file:
            data = load_yaml(cfg.prior_file)
            if data["features"] != FEATURES:
                raise ValueError(f"{cfg.prior_file} was fitted for different features; refit it")
            return np.array(data["mean"], dtype=float), np.array(data["cov"], dtype=float)
        mu = np.array([cfg.prior_mean.get(f, 0.0) for f in FEATURES], dtype=float)
        return mu, np.diag([cfg.prior_var[cls._block(f)] for f in FEATURES]).astype(float)

    @staticmethod
    def _block(f: str) -> str:
        return "action" if f in ACTION_FEATURES else "context" if f in CONTEXT_FEATURES else "interaction"

    def features(self, state: State) -> np.ndarray:
        """(15, N_FEATURES) design matrix: one row per action for the given state."""
        ctx = context(state, self.clinical.window, self.cfg.nominal_trials)
        c = np.array([
            0.0 if ctx.rolling_acc is None else ctx.rolling_acc - self.clinical.target,
            min(ctx.fail_streak, 3) / 3.0,
            ctx.progress,
            0.0 if ctx.mean_rt is None else (ctx.mean_rt - self.cfg.rt_center_s) / self.cfg.rt_scale_s,
        ])
        inter = np.zeros((N_ACTIONS, len(INTERACTION_FEATURES)))
        if ctx.last_error in ("sound", "meaning", "no_response"):
            e = ("sound", "meaning", "no_response").index(ctx.last_error)
            for i, cue in enumerate(ACTION_CUES):
                if cue in _PHONEMIC:
                    inter[i, 2 * e] = 1.0
                elif cue in _SEMANTIC:
                    inter[i, 2 * e + 1] = 1.0
        return np.hstack([_ACTIONS_X, np.tile(c, (N_ACTIONS, 1)), inter])

    def predict(self, state: State) -> np.ndarray:
        """Posterior-mean p(correct) per action (for the dashboard and tests; act() samples instead)."""
        return 1.0 / (1.0 + np.exp(-(self.features(state) @ self.mu)))

    def act(self, state: State, mask: np.ndarray) -> Action:
        allowed = clinical_gate(state, mask, self.cfg.cue_only_after_error)
        z = self.rng.standard_normal(N_FEATURES)
        L = np.linalg.cholesky(self.Sigma + 1e-9 * np.eye(N_FEATURES))
        w = self.mu + L @ z
        p = 1.0 / (1.0 + np.exp(-(self.features(state) @ w)))
        i, progressive = select_action(p, allowed, self._strength, self.clinical.target,
                                       self.cfg.tolerance, self.cfg.cue_cost, self.cfg.tier_bonus)
        return Action(tier=int(ACTION_TIERS[i]), cue=ACTION_CUES[i],
                      rationale=describe(i, p[i], self.clinical.target, progressive, "sampled"))

    def update(self, state: State, action: Action, outcome: Outcome) -> None:
        x = self.features(state)[action_index(action.tier, action.cue)]
        self.Sigma = self.Sigma + self.cfg.process_noise * np.eye(N_FEATURES)
        p = 1.0 / (1.0 + np.exp(-(x @ self.mu)))
        Sx = self.Sigma @ x
        w = p * (1.0 - p)
        self.Sigma = self.Sigma - (w / (1.0 + w * (x @ Sx))) * np.outer(Sx, Sx)
        self.mu = self.mu + self.Sigma @ x * (float(outcome.correct) - p)
