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
# "unclassified" (added for the practice flow): a helper-confirmed error whose kind (sound / meaning)
# could not be identified. It is a real, confirmed error; it is never a stand-in for an uncertain or
# failed recording, which never reaches the agent at all.
ErrorType = Literal["correct", "sound", "meaning", "no_response", "unclassified"]

CUES: tuple[str, ...] = get_args(Cue)
# Error types with their own one-hot slot in trained models' observations (PPO was trained on these
# four). "unclassified" deliberately has no slot: it is encoded as "an error of unknown kind".
ERROR_TYPES: tuple[str, ...] = ("correct", "sound", "meaning", "no_response")
ALL_ERROR_TYPES: tuple[str, ...] = get_args(ErrorType)
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
    # Optional (practice flow): stable identifiers of the exercise and of this attempt.
    exercise_id: str | None = Field(default=None, max_length=128)
    attempt_id: str | None = Field(default=None, max_length=128)

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
    # Optional (practice flow): the picture currently on screen and what has happened on it so far.
    # Language-model policies need it to choose between retrying this picture and moving on.
    current_exercise: "CurrentExercise | None" = None

    @field_validator("last_trials")
    @classmethod
    def _keep_last_ten(cls, v: list[TrialRecord]) -> list[TrialRecord]:
        # Truncate rather than reject: a frontend sending its whole history should still work.
        return v[-MAX_LAST_TRIALS:]


class CurrentExercise(BaseModel):
    """The exercise on screen: identity, tier, hints already shown on it, attempts so far."""

    model_config = _LENIENT
    exercise_id: str = Field(min_length=1, max_length=128)
    word: str
    tier: int = Field(ge=1, le=3)
    hints_shown: list[Cue] = Field(default_factory=list)
    attempts: int = Field(default=0, ge=0)


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
    # Optional (practice flow): makes /update idempotent. A repeated request with the same attempt_id
    # and the same content is acknowledged without learning twice; different content is a 409.
    attempt_id: str | None = Field(default=None, min_length=1, max_length=128)


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


State.model_rebuild()
