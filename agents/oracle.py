"""Privileged oracle: knows the patient's hidden parameters (simulation only).

At every trial it computes the TRUE mean p(correct) of each allowed (tier, cue) for this patient at
this moment (current ability, item learning and fatigue) and applies the same selection rule as the
bandits (agents/selection.py): hardest tier, then weakest cue, among actions within tolerance of
the target; otherwise the closest to target. It obeys the same clinical masks, including
"cue only after an error".

It is a strong privileged *heuristic*, not a proven optimum: it is myopic (one trial ahead) and
targets the band rather than learning directly, so a learned policy can beat it on some metrics.
"""
from __future__ import annotations

import numpy as np

from agents.selection import clinical_gate, describe, select_action
from core.config import ClinicalConfig
from core.masks import ACTION_CUES, ACTION_TIERS, ACTIONS
from core.schema import Action, Outcome, State
from sim.patient import SimPatient


class OraclePolicy:
    name = "oracle"

    def __init__(self, patient: SimPatient, clinical: ClinicalConfig, tolerance: float, cue_cost: float,
                 tier_bonus: float, cue_only_after_error: bool) -> None:
        self.patient, self.clinical = patient, clinical
        self.tolerance, self.cue_cost, self.tier_bonus = tolerance, cue_cost, tier_bonus
        self.cue_only_after_error = cue_only_after_error
        self._strength = np.array([clinical.cue_strength[c] for c in ACTION_CUES], dtype=float)

    def act(self, state: State, mask: np.ndarray) -> Action:
        allowed = clinical_gate(state, mask, self.cue_only_after_error)
        p = np.array([self.patient.p_correct_tier(t, c) for t, c in ACTIONS])
        i, progressive = select_action(p, allowed, self._strength, self.clinical.target, self.tolerance,
                                       self.cue_cost, self.tier_bonus)
        return Action(tier=int(ACTION_TIERS[i]), cue=ACTION_CUES[i],
                      rationale=describe(i, p[i], self.clinical.target, progressive, "true"))

    def update(self, state: State, action: Action, outcome: Outcome) -> None:
        pass
