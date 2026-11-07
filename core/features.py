"""Features computed from the State JSON only.

Every learned agent (bandit, PPO, LLM prompt builder) gets its inputs from here, in simulation and
live alike. Nothing reads simulator internals, so whatever works in simulation can be deployed
unchanged on the real app's State JSON.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.schema import State


@dataclass(frozen=True)
class Context:
    rolling_acc: float | None   # accuracy over the last `window` trials, None before any trial
    n_recent: int               # number of trials the rolling accuracy is based on
    fail_streak: int            # consecutive failures at the end of the history
    last_correct: bool | None   # None before any trial
    last_error: str | None      # error_type of the last trial
    mean_rt: float | None       # mean response time over the last `window` trials (seconds)
    progress: float             # trial_count / nominal session length, capped at 1


def context(state: State, window: int, nominal_trials: int) -> Context:
    recent = state.last_trials[-window:]
    streak = 0
    for t in reversed(state.last_trials):
        if t.outcome:
            break
        streak += 1
    last = state.last_trials[-1] if state.last_trials else None
    return Context(
        rolling_acc=(sum(t.outcome for t in recent) / len(recent)) if recent else None,
        n_recent=len(recent),
        fail_streak=streak,
        last_correct=None if last is None else last.outcome,
        last_error=None if last is None else last.error_type,
        mean_rt=(sum(t.response_time for t in recent) / len(recent)) if recent else None,
        progress=min(1.0, state.trial_count / nominal_trials),
    )
