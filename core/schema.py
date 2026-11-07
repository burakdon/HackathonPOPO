"""The State / Action / Outcome contract agreed with the team.

This module is the single source of truth for the JSON interfaces. The simulator, the service,
and the LLM agents all import these models, so the contract cannot drift between components.
Unknown extra fields are ignored (not rejected) so the frontend can add fields like timestamps
without breaking the agent.
"""
from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Cue = Literal["none", "first_sound", "first_letter", "sentence_frame", "semantic"]
ErrorType = Literal["correct", "sound", "meaning", "no_response"]

CUES: tuple[str, ...] = get_args(Cue)
ERROR_TYPES: tuple[str, ...] = get_args(ErrorType)
TIERS: tuple[int, ...] = (1, 2, 3)

# Interface constant (part of the agreed contract, not a tunable): State carries at most the
# last 10 trials.
MAX_LAST_TRIALS = 10

_LENIENT = ConfigDict(extra="ignore")


def _check_correct_matches_error(correct: bool, error_type: str) -> None:
    if correct != (error_type == "correct"):
        raise ValueError(
            f"inconsistent outcome: correct={correct} but error_type={error_type!r} "
            "(error_type must be 'correct' iff the trial was correct)"
        )


class TrialRecord(BaseModel):
    """One past trial as reported by the app."""

    model_config = _LENIENT
    word: str
    tier: int = Field(ge=1, le=3)
    cue: Cue
    outcome: bool
    error_type: ErrorType
    response_time: float = Field(ge=0.0, description="seconds")
    probe: bool = Field(default=False, description="True for a review probe (chosen by the app, not the agent)")

    @model_validator(mode="after")
    def _consistent(self) -> "TrialRecord":
        _check_correct_matches_error(self.outcome, self.error_type)
        return self


class State(BaseModel):
    """What the agent sees before choosing the next trial."""

    model_config = _LENIENT
    session_id: str | None = Field(
        default=None, min_length=1, max_length=128,
        description="Optional (added after the original contract). Identifies the patient "
                    "session so the service keeps one online-learning agent per session.",
    )
    last_trials: list[TrialRecord] = Field(default_factory=list)
    current_tier: int = Field(ge=1, le=3, description="tier of the last trial the AGENT chose (review "
                                                       "probes do not change it)")
    trial_count: int = Field(ge=0)
    session_minutes: float = Field(ge=0.0)

    @field_validator("last_trials")
    @classmethod
    def _keep_last_ten(cls, v: list[TrialRecord]) -> list[TrialRecord]:
        # Truncate rather than reject: a frontend sending its whole history should still work.
        return v[-MAX_LAST_TRIALS:]


class Action(BaseModel):
    """The agent's choice for the next trial."""

    model_config = _LENIENT
    tier: int = Field(ge=1, le=3)
    cue: Cue
    rationale: str | None = Field(default=None, max_length=500)


class Outcome(BaseModel):
    """Result of the trial the action was applied to (body of POST /update)."""

    model_config = _LENIENT
    correct: bool
    error_type: ErrorType
    response_time: float = Field(ge=0.0, description="seconds")

    @model_validator(mode="after")
    def _consistent(self) -> "Outcome":
        _check_correct_matches_error(self.correct, self.error_type)
        return self


class UpdateRequest(BaseModel):
    """POST /update: the state the action was chosen in, the action actually shown, its outcome.

    If the clinician overrode the agent, `action` must be the clinician's action (what the
    patient actually saw), so the bandit learns from what really happened.
    """

    model_config = _LENIENT
    state: State
    action: Action
    outcome: Outcome
    probe: bool = Field(default=False, description="True if this trial was a review probe; the agent does "
                                                   "not learn from trials it did not choose")


class NextTrial(BaseModel):
    """POST /next_trial response: the complete next trial, with the item chosen by the shared rule."""

    model_config = _LENIENT
    word: str
    image_file: str
    tier: int = Field(ge=1, le=3)
    cue: Cue
    probe: bool
    rationale: str | None = None


class LearnRequest(BaseModel):
    """POST /learn_from_overrides body."""

    model_config = _LENIENT
    session_id: str = Field(min_length=1, max_length=128)


class StateTypeChange(BaseModel):
    state_type: str                 # e.g. "after a sound error"
    clinician_cue: str              # the cue the clinician chose in overrides of this state type
    n_overrides: int
    n_eval_states: int              # reference states of this type where that cue is allowed
    p_before: float                 # mean probability of choosing that cue, before
    p_after: float                  # ... and after the update


class LearnResult(BaseModel):
    session_id: str
    n_overrides: int
    n_skipped: int                  # overrides whose clinician action is not allowed in its state
    n_anchor_states: int
    seconds: float
    by_state_type: list[StateTypeChange]
    other_states_unchanged: float   # share of other reference states whose top action is unchanged


class OverrideRequest(BaseModel):
    """POST /override: one clinician demonstration, stored for DITTO."""

    model_config = _LENIENT
    state: State
    agent_action: Action
    clinician_action: Action
    note: str | None = Field(default=None, max_length=1000)
