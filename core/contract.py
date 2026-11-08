"""Response and request models of the CueAgain API contract (docs/API_CONTRACT.md).

These are the shapes the frontend (product/), the backend and the decision model agree on. The sample
responses in docs/contract_examples/ are generated from them (scripts/build_contract_examples.py),
and the real endpoints will return these same models, so samples and backend can't drift apart.

Shared logic that both the sample generator and the real backend use lives here too:
`summarize` (the three per-picture summaries) and `rule_matches` / `rule_action` (clinician rules).
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.practice import DecisionContext

_STRICT = ConfigDict(extra="forbid")

Outcome = Literal["correct", "incorrect", "unclear"]
ErrorType = Literal["sound", "meaning", "no_response", "unclassified"]
Hint = Literal["first_sound", "first_letter", "sentence_frame", "semantic"]
HINTS: tuple[str, ...] = ("first_sound", "first_letter", "sentence_frame", "semantic")
Level = Annotated[int, Field(ge=1, le=3)]
PictureResult = Literal["independent", "assisted", "unsuccessful", "not_assessed"]
HintOrder = Literal["match_error", "first_sound_first", "meaning_first", "least_help"]


class _M(BaseModel):
    model_config = _STRICT


# ---------------------------------------------------------------- actions (§5)

class RetryAction(_M):
    op: Literal["retry"]
    hint: Hint


class NextAction(_M):
    op: Literal["next"]
    tier: Level


Action = Annotated[RetryAction | NextAction, Field(discriminator="op")]


# ---------------------------------------------------------------- pictures (§2)

class HintClip(_M):
    text: str
    audio_url: str | None          # None while the clip hasn't been generated


class Picture(_M):
    exercise_id: str
    word: str
    accepted: list[str]
    level: Level
    image_url: str
    hints: dict[Hint, HintClip]
    whole_word_audio_url: str | None
    attribution: str

    @model_validator(mode="after")
    def _all_hints(self) -> Picture:
        if set(self.hints) != set(HINTS):
            raise ValueError("every picture has all four hints")
        return self


# ---------------------------------------------------------------- patients and plans (§3)

class Plan(_M):
    level: Level
    pictures_per_session: int = Field(ge=1, le=50)
    max_attempts: int = Field(ge=1, le=6)
    break_after_misses: int = Field(ge=1, le=10)
    hint_order: HintOrder                     # fallback only (D1)
    session_minutes: int = Field(ge=1, le=120)
    days_per_week: int = Field(ge=1, le=7)
    personal_words: list[str] = Field(default_factory=list)   # stored, not used yet


class ActiveModel(_M):
    version: str
    patient_adapter: str | None


class Patient(_M):
    id: str
    name: str
    first: str
    simulated: bool
    about: str
    plan: Plan
    active_model: ActiveModel
    pending_examples: int = Field(ge=0)


class NewPatient(_M):
    name: str = Field(min_length=1, max_length=80)
    first: str = Field(min_length=1, max_length=40)
    about: str = Field(default="", max_length=500)


class LogEntry(_M):
    when: str
    who: str
    what: str


# ---------------------------------------------------------------- summaries (§6)

class Counts(_M):
    assessed: int = Field(ge=0)
    independent: int = Field(ge=0)
    assisted: int = Field(ge=0)
    unsuccessful: int = Field(ge=0)
    not_assessed: int = Field(ge=0)

    @model_validator(mode="after")
    def _adds_up(self) -> Counts:
        if self.independent + self.assisted + self.unsuccessful != self.assessed:
            raise ValueError("assessed must equal independent + assisted + unsuccessful")
        return self


class ErrorCounts(_M):
    sound: int = Field(ge=0)
    wrong_word: int = Field(ge=0)
    no_answer: int = Field(ge=0)
    unclassified: int = Field(ge=0)


class SessionRow(_M):
    session_id: str
    date: str
    simulated: bool
    model_version: str
    pictures: int = Field(ge=0)
    named_without_hint: float | None = Field(ge=0, le=1)
    named_with_hint: float | None = Field(ge=0, le=1)
    unsuccessful: float | None = Field(ge=0, le=1)
    counts: Counts
    errors: ErrorCounts
    mean_level: float | None
    minutes: float = Field(ge=0)
    ended_early: bool

    @model_validator(mode="after")
    def _fractions(self) -> SessionRow:
        n = self.counts.assessed
        want = (None, None, None) if n == 0 else (
            round(self.counts.independent / n, 4), round(self.counts.assisted / n, 4),
            round(self.counts.unsuccessful / n, 4))
        if (self.named_without_hint, self.named_with_hint, self.unsuccessful) != want:
            raise ValueError("fractions must equal counts / assessed (rounded to 4 places)")
        if self.pictures != n + self.counts.not_assessed:
            raise ValueError("pictures must equal assessed + not_assessed")
        return self


class PatientRow(_M):
    id: str
    name: str
    first: str
    simulated: bool
    last_session: SessionRow | None


# ---------------------------------------------------------------- practice (§4)

class ShownHint(_M):
    type: Hint
    text: str
    audio_url: str | None


class Presentation(_M):
    presentation_id: str
    number: int = Field(ge=1)
    of: int = Field(ge=1)
    exercise_id: str
    word: str
    level: Level
    image_url: str
    attempt_number: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    hint: ShownHint | None


class SessionStarted(_M):
    session_id: str
    patient_id: str
    model_version: str
    plan: Plan
    picture: Presentation


class Signals(_M):
    audio_seconds: float | None = Field(default=None, ge=0)
    no_speech_prob: float | None = Field(default=None, ge=0, le=1)
    word_confidence: float | None = Field(default=None, ge=0, le=1)


class Score(_M):
    outcome: Outcome
    error_type: ErrorType | None
    heard_word: str | None
    matched_name: str | None
    reason: str
    signals: Signals

    @model_validator(mode="after")
    def _consistent(self) -> Score:
        if (self.outcome == "incorrect") != (self.error_type is not None):
            raise ValueError("error_type is set exactly when the outcome is incorrect")
        if (self.outcome == "correct") != (self.matched_name is not None):
            raise ValueError("matched_name is set exactly when the outcome is correct")
        return self


NextKind = Literal["correct", "hint", "show_word", "try_again", "not_assessed", "move_on"]
LEAVES_PICTURE = {"correct", "show_word", "not_assessed", "move_on"}


class NextStep(_M):
    kind: NextKind
    message: str
    hint: ShownHint | None = None
    attempt_number: int | None = None
    max_attempts: int | None = None
    decision_id: str | None = None
    word: str | None = None                       # show_word: the word to say
    whole_word_audio_url: str | None = None       # show_word: its clip (None if not generated)
    picture: Presentation | None = None
    session_complete: bool = False
    summary: SessionRow | None = None
    break_suggested: bool = False

    @model_validator(mode="after")
    def _shape(self) -> NextStep:
        k = self.kind
        if k == "hint" and (self.hint is None or self.decision_id is None or self.attempt_number is None):
            raise ValueError("a hint step carries the hint, the attempt number and the decision_id")
        if k != "hint" and self.hint is not None:
            raise ValueError("only a hint step carries a hint")
        if k == "show_word" and self.word is None:
            raise ValueError("show_word carries the word")
        if k in LEAVES_PICTURE:
            if self.session_complete == (self.picture is not None):
                raise ValueError("leaving a picture gives either the next picture or session_complete")
            if self.session_complete and self.summary is None:
                raise ValueError("a completed session carries its summary")
        else:
            if self.picture is not None or self.session_complete or self.break_suggested:
                raise ValueError("hint and try_again stay on the current picture")
        if k in ("try_again", "not_assessed") and self.decision_id is not None:
            raise ValueError("the model isn't asked after an unclear recording")
        return self


class AttemptResult(_M):
    attempt_id: str
    presentation_id: str
    counted: bool
    transcript: str | None
    score: Score
    next: NextStep

    @model_validator(mode="after")
    def _counted(self) -> AttemptResult:
        if self.counted != (self.score.outcome != "unclear"):
            raise ValueError("an attempt counts exactly when it isn't unclear")
        if self.score.outcome == "unclear" and self.next.kind not in ("try_again", "not_assessed"):
            raise ValueError("an unclear attempt leads to try_again or not_assessed")
        return self


class NoAnswerIn(_M):
    presentation_id: str
    client_attempt_id: str = Field(min_length=8, max_length=64)


class SessionState(_M):
    session_id: str
    patient_id: str
    status: Literal["active", "complete", "ended"]
    picture: Presentation | None
    pictures_done: int = Field(ge=0)
    summary: SessionRow | None


# ---------------------------------------------------------------- decisions (§5)

class ContractContext(DecisionContext):
    """The existing DecisionContext plus the patient it belongs to."""
    patient_id: str


class ModelDecision(_M):
    """What decide() returns (Python interface, not HTTP)."""
    action: Action
    source: str
    fallback_reason: str | None
    raw_output: str | None
    latency_ms: int = Field(ge=0)


class Example(_M):
    example_id: str
    patient_id: str
    source: Literal["correction", "note"]
    decision_id: str
    note_id: str | None
    model_chose: Action
    preferred: Action
    note: str | None
    status: str                                    # "pending" or "used in <run_id>"

    @model_validator(mode="after")
    def _status(self) -> Example:
        if not (self.status == "pending" or self.status.startswith("used in ")):
            raise ValueError("status is 'pending' or 'used in <run_id>'")
        if (self.source == "note") != (self.note_id is not None):
            raise ValueError("note_id is set exactly for examples made from a note")
        return self


class StoredDecision(_M):
    decision_id: str
    session_id: str
    presentation_id: str
    after_attempt_id: str
    context: ContractContext
    options: list[Action] = Field(min_length=1)
    action: Action
    source: str
    fallback_reason: str | None
    latency_ms: int = Field(ge=0)
    at: str
    correction: Example | None = None

    @model_validator(mode="after")
    def _in_options(self) -> StoredDecision:
        if self.action not in self.options:
            raise ValueError("the applied action must be one of the options")
        return self


# ---------------------------------------------------------------- review (§6)

class ReviewAttempt(_M):
    attempt_id: str
    number: int | None = Field(ge=1)               # None for unclear (not counted)
    hint_shown: Hint | None
    transcript: str | None
    outcome: Outcome
    error_type: ErrorType | None
    counted: bool
    reason: str
    audio_url: str | None                          # None for "I'm not sure"
    at: str


class ReviewPicture(_M):
    presentation_id: str
    exercise_id: str
    word: str
    level: Level
    image_url: str
    result: PictureResult
    attempts: list[ReviewAttempt]
    decisions: list[StoredDecision]


class SessionReview(_M):
    session_id: str
    summary: SessionRow
    pictures: list[ReviewPicture]


# ---------------------------------------------------------------- steering (§7)

class CorrectionIn(_M):
    preferred: Action
    note: str | None = Field(default=None, max_length=1000)


class Range(_M):
    min: int | None = Field(default=None, ge=0)
    max: int | None = Field(default=None, ge=0)


class RuleWhen(_M):
    last_error: list[Literal["correct", "sound", "meaning", "no_response", "unclassified"]] | None = None
    failed_attempts_on_picture: Range | None = None
    hints_tried_include: list[Hint] | None = None
    hints_tried_exclude: list[Hint] | None = None
    missed_pictures_in_a_row: Range | None = None
    level: list[Level] | None = None


class RetryThen(_M):
    op: Literal["retry"]
    hint: Hint


class NextThen(_M):
    op: Literal["next"]
    level: Literal["easier", "same", "harder"]


RuleThen = Annotated[RetryThen | NextThen, Field(discriminator="op")]


class Rule(_M):
    when: RuleWhen
    then: RuleThen
    summary: str


class NoteIn(_M):
    text: str = Field(min_length=1, max_length=2000)


class Note(_M):
    note_id: str
    patient_id: str
    text: str
    rule: Rule | None
    rule_source: Literal["claude", "clinician", "none"]
    rule_status: Literal["draft", "confirmed", "none"]
    problem: str | None

    @model_validator(mode="after")
    def _shape(self) -> Note:
        if (self.rule is None) != (self.rule_status == "none"):
            raise ValueError("rule_status is 'none' exactly when there is no rule")
        return self


class RuleIn(_M):
    rule: Rule


class RuleMatch(_M):
    decision_id: str
    date: str
    word: str
    situation: str
    model_chose: Action
    rule_says: Action | None                       # None when the rule's action isn't allowed there
    already_agrees: bool
    allowed: bool


class ExamplesIn(_M):
    decision_ids: list[str] = Field(min_length=1)


# ---------------------------------------------------------------- model updates (§8)

class Progress(_M):
    stage: Literal["queued", "sft", "ditto", "preview", "done"]
    round: int | None = None
    of: int | None = None


class Agreement(_M):
    new: int = Field(ge=0)
    current: int = Field(ge=0)
    total: int = Field(ge=0)
    label: str | None = None


class Unchanged(_M):
    same: int = Field(ge=0)
    total: int = Field(ge=0)


class Preview(_M):
    follows_your_examples: Agreement
    follows_your_rules_new_situations: Agreement
    unchanged_elsewhere: Unchanged


class TrainingRun(_M):
    run_id: str
    patient_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: Progress
    started: str | None
    finished: str | None
    base: str
    example_ids: list[str]
    preview: Preview | None
    prepared: bool = False                 # True: result taken from a run trained earlier (§10.5)
    prepared_run_id: str | None = None
    error: str | None

    @model_validator(mode="after")
    def _shape(self) -> TrainingRun:
        if (self.status == "completed") != (self.preview is not None):
            raise ValueError("a preview exists exactly for completed runs")
        if self.prepared != (self.prepared_run_id is not None):
            raise ValueError("prepared runs name the run they were taken from")
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("error is set exactly for failed runs")
        return self


class UsePreparedIn(_M):
    prepared_run_id: str


class ActivateIn(_M):
    version: str


class ModelVersion(_M):
    id: str
    label: str
    kind: Literal["baseline", "shared", "patient"]
    available: bool
    active: bool


class ErrorBody(_M):
    detail: str


# ---------------------------------------------------------------- shared logic

def picture_result(attempts: list[ReviewAttempt]) -> PictureResult:
    counted = [a for a in attempts if a.counted]
    if not counted:
        return "not_assessed"
    hit = next((a for a in counted if a.outcome == "correct"), None)
    if hit is None:
        return "unsuccessful"
    return "independent" if hit.number == 1 and hit.hint_shown is None else "assisted"


_ERR_KEY = {"sound": "sound", "meaning": "wrong_word", "no_response": "no_answer", "unclassified": "unclassified"}


def summarize(session_id: str, date: str, simulated: bool, model_version: str,
              pictures: list[ReviewPicture], minutes: float, ended_early: bool) -> SessionRow:
    """Per-picture summaries (never per recording). Fractions are of assessed pictures."""
    res = [p.result for p in pictures]
    c = Counts(assessed=sum(r != "not_assessed" for r in res), independent=res.count("independent"),
               assisted=res.count("assisted"), unsuccessful=res.count("unsuccessful"),
               not_assessed=res.count("not_assessed"))
    errs = {k: 0 for k in _ERR_KEY.values()}
    for p in pictures:
        for a in p.attempts:
            if a.counted and a.error_type:
                errs[_ERR_KEY[a.error_type]] += 1
    n = c.assessed
    frac = (lambda k: round(k / n, 4)) if n else (lambda k: None)
    assessed_levels = [p.level for p in pictures if p.result != "not_assessed"]
    return SessionRow(
        session_id=session_id, date=date, simulated=simulated, model_version=model_version,
        pictures=len(pictures), named_without_hint=frac(c.independent), named_with_hint=frac(c.assisted),
        unsuccessful=frac(c.unsuccessful), counts=c, errors=ErrorCounts(**errs),
        mean_level=round(sum(assessed_levels) / len(assessed_levels), 2) if assessed_levels else None,
        minutes=minutes, ended_early=ended_early)


def _in(r: Range | None, v: int) -> bool:
    return r is None or ((r.min is None or v >= r.min) and (r.max is None or v <= r.max))


def rule_matches(rule: Rule, ctx: DecisionContext, missed_pictures_in_a_row: int = 0) -> bool:
    """Whether the situation a decision was made in satisfies every given part of `rule.when`."""
    w, latest = rule.when, ctx.latest
    on_picture = [a for a in ctx.history if a.exercise_id == ctx.current.exercise_id]
    failed = sum(not a.correct for a in on_picture)
    last = "correct" if latest.correct else latest.error
    tried = set(ctx.current.hints_shown)
    return ((w.last_error is None or last in w.last_error)
            and _in(w.failed_attempts_on_picture, failed)
            and (w.hints_tried_include is None or set(w.hints_tried_include) <= tried)
            and (w.hints_tried_exclude is None or not (set(w.hints_tried_exclude) & tried))
            and _in(w.missed_pictures_in_a_row, missed_pictures_in_a_row)
            and (w.level is None or ctx.current.tier in w.level))


def rule_action(rule: Rule, ctx: DecisionContext) -> RetryAction | NextAction:
    """The rule's `then` as an absolute action in this situation (relative levels resolved)."""
    t = rule.then
    if isinstance(t, RetryThen):
        return RetryAction(op="retry", hint=t.hint)
    step = {"easier": -1, "same": 0, "harder": 1}[t.level]
    return NextAction(op="next", tier=min(3, max(1, ctx.current.tier + step)))


def missed_before(ctx: DecisionContext) -> int:
    """Pictures missed in a row just before the current one, from the session history: a picture is
    missed when none of its counted attempts was correct. Unclear (not counted) pictures leave no trace in
    the history, so they are skipped. A suggested break doesn't reset the count."""
    groups: list[list[bool]] = []
    last = None
    for a in ctx.history:
        if a.exercise_id != last:
            groups.append([])
            last = a.exercise_id
        groups[-1].append(a.correct)
    groups = groups[:-1]                       # the last group is the current picture
    n = 0
    for g in reversed(groups):
        if any(g):
            break
        n += 1
    return n


def matches_rule(rule: Rule, ctx: DecisionContext) -> bool:
    return rule_matches(rule, ctx, missed_before(ctx))


_HINT_WORD = {"semantic": "meaning", "first_sound": "first sound", "first_letter": "first letter",
              "sentence_frame": "sentence"}
_LAST_WORD = {"meaning": "a wrong word", "sound": "a sound error", "no_response": "no answer",
              "unclassified": "unclassified"}


def situation(ctx: DecisionContext) -> str:
    """The situation a decision was made in, in plain English (for the clinician's list of matches)."""
    c = ctx
    failed = sum(not a.correct for a in c.history if a.exercise_id == c.current.exercise_id)
    hints = ", ".join(_HINT_WORD[h] for h in c.current.hints_shown) or "none"
    last = "correct" if c.latest.correct else _LAST_WORD[c.latest.error]
    return f"{failed} failed tr{'y' if failed == 1 else 'ies'} on “{c.current.word}”, hints so far: {hints}, last answer {last}"
