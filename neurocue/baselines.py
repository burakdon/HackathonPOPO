"""Rule-based therapy policies on the engine environment: what clinics do, as a bar for a learned policy.

Each decides from `env.observable()` only. All return an action index into neurocue.env.ACTIONS.
  spaced_plain        round-robin through the words, never cues (Middleton's spaced retrieval practice)
  massed_plain        each word three times in a row, then the next (Middleton's massed condition)
  spaced_hierarchy    round-robin; after an error, repeat the word with a first sound, then the word
                      (the standard cueing hierarchy)
  errorless           round-robin, the whole word every time (errorless learning)
  weakest_hierarchy   practise the word with the lowest observed accuracy, with the hierarchy
  random              uniform over allowed actions
"""
from __future__ import annotations

import numpy as np

from neurocue.env import ACTIONS, action_index


class RulePolicy:
    name = "rule"

    def act(self, obs: dict, mask: np.ndarray, rng: np.random.Generator) -> int:
        raise NotImplementedError

    @staticmethod
    def _hierarchy_step(obs: dict) -> str | None:
        """Cue for the next trial if the last one was an error on the same word; else None."""
        last = obs["last_trials"][-1] if obs["last_trials"] else None
        if last is None or last["correct"] or obs["trial_in_session"] == 0:
            return None
        return {None: "phonemic_initial", "phonemic_initial": "whole_word"}.get(last["cue"], None)


class SpacedPlain(RulePolicy):
    name = "spaced_plain"

    def act(self, obs, mask, rng):
        return action_index("due", None)


class MassedPlain(RulePolicy):
    name = "massed_plain"

    def __init__(self, presentations: int = 3) -> None:
        self.presentations = presentations

    def act(self, obs, mask, rng):
        li = obs["last_item"]
        if li is not None and obs["trial_in_session"] > 0 and obs["items"][li].session_presentations < self.presentations:
            return action_index("repeat", None)
        return action_index("due", None)


class SpacedHierarchy(RulePolicy):
    name = "spaced_hierarchy"

    def act(self, obs, mask, rng):
        cue = self._hierarchy_step(obs)
        if cue is not None and mask[action_index("repeat", cue)]:
            return action_index("repeat", cue)
        return action_index("due", None)


class Errorless(RulePolicy):
    name = "errorless"

    def act(self, obs, mask, rng):
        a = action_index("due", "whole_word")
        return a if mask[a] else action_index("due", None)


class WeakestHierarchy(RulePolicy):
    name = "weakest_hierarchy"

    def act(self, obs, mask, rng):
        cue = self._hierarchy_step(obs)
        if cue is not None and mask[action_index("repeat", cue)]:
            return action_index("repeat", cue)
        return action_index("weakest", None)


class RandomPolicy(RulePolicy):
    name = "random"

    def act(self, obs, mask, rng):
        return int(rng.choice(np.flatnonzero(mask)))


BASELINES: dict[str, type[RulePolicy]] = {c.name: c for c in (SpacedPlain, MassedPlain, SpacedHierarchy, Errorless,
                                                             WeakestHierarchy, RandomPolicy)}
