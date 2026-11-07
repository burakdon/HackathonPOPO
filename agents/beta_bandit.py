"""Non-contextual Beta-Bernoulli Thompson sampling over the 15 (tier, cue) arms.

Ported from docs/scaffold/rl_starter.py as the first live agent so the frontend could integrate early. The
contextual Bayesian logistic bandit replaces it as the main bandit; this one stays as a
simple, dependency-free baseline.

Each arm keeps a Beta posterior over p(correct). Every trial we draw one sample per arm and pick
the allowed arm whose sample is closest to the target accuracy, with small tie-breakers for
harder tiers and weaker cues. This is "target-tracking" Thompson sampling: the bandit's reward is
closeness to target, not correctness, which is why we model p(correct) and score afterwards.

Changes from the scaffold, and why:
1. The current tier is read from `state.current_tier`. The scaffold remembered its own last
   choice, so after a clinician override it planned from the wrong tier.
2. Constraints come in as an explicit action mask shared with every other agent.
3. Discounting decays *all* arms toward the prior on each update (standard discounted Thompson
   sampling) instead of only the arm just pulled, so stale estimates fade uniformly.
4. Samples are drawn for all 15 arms every call, so the random stream consumed per call does not
   depend on the mask; same seed + same inputs gives the same actions.
5. Selection uses the shared progression rule (agents/selection.py): among arms within tolerance of
   target, the highest tier and weakest cue win, so the bandit no longer coasts on an easy tier.
"""
from __future__ import annotations

import numpy as np

from agents.selection import clinical_gate, describe, select_action
from core.config import BetaBanditConfig, ClinicalConfig
from core.masks import ACTION_CUES, ACTION_TIERS, action_index
from core.schema import Action, Outcome, State


class BetaThompsonBandit:
    name = "beta_bandit"

    def __init__(self, cfg: BetaBanditConfig, clinical: ClinicalConfig, seed: int) -> None:
        self.cfg, self.clinical = cfg, clinical
        self.rng = np.random.default_rng(seed)
        self.prior_alpha = np.array([cfg.prior[t][0] for t in ACTION_TIERS], dtype=float)
        self.prior_beta = np.array([cfg.prior[t][1] for t in ACTION_TIERS], dtype=float)
        self.alpha = self.prior_alpha.copy()
        self.beta = self.prior_beta.copy()
        self._strength = np.array([clinical.cue_strength[c] for c in ACTION_CUES], dtype=float)

    def posterior_mean(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def act(self, state: State, mask: np.ndarray) -> Action:
        allowed = clinical_gate(state, mask, self.cfg.cue_only_after_error)
        samples = self.rng.beta(self.alpha, self.beta)
        i, progressive = select_action(samples, allowed, self._strength, self.clinical.target,
                                       self.cfg.tolerance, self.cfg.cue_cost, self.cfg.tier_bonus)
        return Action(tier=int(ACTION_TIERS[i]), cue=ACTION_CUES[i],
                      rationale=describe(i, samples[i], self.clinical.target, progressive, "sampled"))

    def update(self, state: State, action: Action, outcome: Outcome) -> None:
        d = self.cfg.discount
        self.alpha = self.prior_alpha + d * (self.alpha - self.prior_alpha)
        self.beta = self.prior_beta + d * (self.beta - self.prior_beta)
        i = action_index(action.tier, action.cue)
        if outcome.correct:
            self.alpha[i] += 1.0
        else:
            self.beta[i] += 1.0
