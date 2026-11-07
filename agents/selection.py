"""Shared action-selection rule for target-tracking agents (both bandits, and the oracle).

Given a predicted (or sampled) p(correct) for each of the 15 actions:

1. Among allowed actions whose p is within `tolerance` of the target, prefer progression: the
   highest tier, then the weakest cue, then the p closest to target.
2. If none is within tolerance, take the allowed action closest to target, with small tie-breaks
   (a cost for stronger cues, a bonus for harder tiers).

Why rule 1: "closest to 85%" alone lets an agent coast on an easy tier whose p happens to sit near
target (the 1a smoke run spent 37/40 trials on tier 1). Within the band, a harder item with a
weaker cue is the more useful practice, and it is what an SLP would choose.
"""
from __future__ import annotations

import numpy as np

from core.masks import ACTION_CUES, ACTION_TIERS, no_cue_mask
from core.schema import State

_NO_CUE = no_cue_mask()


def clinical_gate(state: State, mask: np.ndarray, cue_only_after_error: bool) -> np.ndarray:
    """Optionally forbid cues unless the previous trial was an error (as an SLP would)."""
    allowed = mask.copy()
    last_correct = not state.last_trials or state.last_trials[-1].outcome
    if cue_only_after_error and last_correct:
        allowed &= _NO_CUE
    if not allowed.any():
        raise ValueError("action mask leaves no allowed action")
    return allowed


def select_action(p: np.ndarray, allowed: np.ndarray, strengths: np.ndarray, target: float,
                  tolerance: float, cue_cost: float, tier_bonus: float) -> tuple[int, bool]:
    """Return (action index, whether it came from the within-tolerance progression rule)."""
    dist = np.abs(p - target)
    near = allowed & (dist <= tolerance)
    if near.any():
        # np.lexsort sorts by the LAST key first: tier descending, then strength, then distance.
        order = np.lexsort((dist, strengths, -ACTION_TIERS))
        return int(next(i for i in order if near[i])), True
    score = dist + cue_cost * strengths - tier_bonus * ACTION_TIERS
    score = np.where(allowed, score, np.inf)
    return int(np.argmin(score)), False


def describe(i: int, p: float, target: float, progressive: bool, what: str) -> str:
    tier, cue = int(ACTION_TIERS[i]), ACTION_CUES[i]
    why = (f"within tolerance of the {target:.0%} target, so the hardest tier and weakest cue was chosen"
           if progressive else f"no option within tolerance; closest to the {target:.0%} target")
    return f"Tier {tier}, cue '{cue}': {what} p(correct)={p:.2f}, {why}."
