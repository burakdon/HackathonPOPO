"""Common interface every therapist policy implements.

Fixed, staircase, bandits, PPO, LLM agents and the oracle all expose the same two calls, so the
session runner, the evaluation harness and the live service can treat them interchangeably.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.schema import Action, Outcome, State


@runtime_checkable
class Policy(Protocol):
    name: str

    def act(self, state: State, mask: np.ndarray) -> Action:
        """Choose the next (tier, cue). Must return an action with mask[index] == True."""
        ...

    def update(self, state: State, action: Action, outcome: Outcome) -> None:
        """Observe the outcome of `action` taken in `state`. No-op for non-learning policies."""
        ...
