"""Discrete action space and action masks.

Clinical constraints are hard masks, not reward penalties: a penalty only discourages a
forbidden action on average, while a mask makes it impossible, which is what a clinician
needs to trust the agent. The same mask feeds MaskablePPO, the bandits and the LLM validators.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from core.schema import CUES, TIERS, Action

# Tier-major ordering: index = (tier - 1) * len(CUES) + cue_index. 3 x 5 = 15 actions.
ACTIONS: tuple[tuple[int, str], ...] = tuple((t, c) for t in TIERS for c in CUES)
N_ACTIONS = len(ACTIONS)
ACTION_TIERS = np.array([t for t, _ in ACTIONS])
ACTION_CUES: tuple[str, ...] = tuple(c for _, c in ACTIONS)


def action_index(tier: int, cue: str) -> int:
    return (tier - 1) * len(CUES) + CUES.index(cue)


def action_mask(current_tier: int, approved_cues: Iterable[str], max_tier_step: int) -> np.ndarray:
    """Boolean mask of shape (15,): True where the action is allowed."""
    approved = set(approved_cues)
    return np.array(
        [abs(t - current_tier) <= max_tier_step and c in approved for t, c in ACTIONS],
        dtype=bool,
    )


def no_cue_mask() -> np.ndarray:
    """True only for actions with cue == 'none'."""
    return np.array([c == "none" for c in ACTION_CUES], dtype=bool)


def is_allowed(action: Action, mask: np.ndarray) -> bool:
    return bool(mask[action_index(action.tier, action.cue)])
