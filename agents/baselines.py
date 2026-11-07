"""Non-learning baselines: fixed difficulty (what worksheets do) and a 2-up/1-down staircase.

Both decide from the State JSON alone, so they behave identically in simulation and live, and a
clinician override (which changes state.current_tier) is respected automatically.
"""
from __future__ import annotations

import numpy as np

from core.masks import action_index
from core.schema import Action, Outcome, State


class FixedPolicy:
    """Same tier and cue every trial. If the tier is not reachable yet (tier-step mask), step toward it."""

    name = "fixed"

    def __init__(self, tier: int = 2, cue: str = "none") -> None:
        self.tier, self.cue = tier, cue

    def act(self, state: State, mask: np.ndarray) -> Action:
        tier = state.current_tier + int(np.sign(self.tier - state.current_tier))
        return Action(tier=tier, cue=self.cue if mask[action_index(tier, self.cue)] else "none",
                      rationale=f"Fixed difficulty: tier {self.tier}, cue '{self.cue}'.")

    def update(self, state: State, action: Action, outcome: Outcome) -> None:
        pass


class StaircasePolicy:
    """Up one tier after 2 consecutive correct answers at the current tier, down one after an error.
    Review probes (not chosen by the policy) are ignored. Never cues. Converges to ~70.7% correct for a fixed difficulty-accuracy curve (Levitt 1971)."""

    name = "staircase"

    def act(self, state: State, mask: np.ndarray) -> Action:
        last = [t for t in state.last_trials if not t.probe]   # react only to its own trials
        tier = state.current_tier
        if last and not last[-1].outcome:
            tier, why = max(1, tier - 1), "error on the last trial: step down"
        elif len(last) >= 2 and all(t.outcome and t.tier == tier for t in last[-2:]):
            tier, why = min(3, tier + 1), "two correct in a row at this tier: step up"
        else:
            why = "hold"
        return Action(tier=tier, cue="none", rationale=f"Staircase 2-up/1-down: {why}.")

    def update(self, state: State, action: Action, outcome: Outcome) -> None:
        pass
