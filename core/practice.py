"""Picture-naming practice: exercise manifest, decision context, valid options and the canonical
text serialization of contexts and actions.

One module on purpose: the demonstration dataset (training/build_demonstrations.py), DITTO
training (training/ditto.py), evaluation (evaluation/compare.py) and live inference
(agents/llm_policy.py) all render contexts and parse actions through the functions here, so the
model sees exactly the same text in training, evaluation and the app.

A decision is made after every *confirmed* attempt (a helper confirmed the result). It is one of:
  retry_current  show the same picture again, optionally with a predefined hint
  next_exercise  move on to a new picture at a tier within one step of the current one
The language model only ever chooses among `valid_options(...)`; which picture "next" means is
resolved deterministically by `ItemBank.next_item`, never by the model.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.config import resolve
from core.schema import CUES, CurrentExercise, Cue, State, TrialRecord

Operation = Literal["retry_current", "next_exercise"]
ErrorCategory = Literal["sound", "meaning", "no_response", "unclassified"]
ERROR_CATEGORIES: tuple[str, ...] = ("sound", "meaning", "no_response", "unclassified")

_LENIENT = ConfigDict(extra="ignore")
_STRICT = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------------
# Configuration

class LLMConfig(BaseModel):
    model_config = _STRICT
    base_model: str
    revision: str | None = None           # pinned Hugging Face revision of the base model
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    # "generate": free greedy generation, then parse. "score": compute the log-probability of every
    # valid option and take the most likely one (output is always a valid option; also used to
    # sample DITTO alternatives in training).
    decoding: Literal["generate", "score"] = "generate"
    # Context window used to fit the "prompted_all" baseline's demonstrations (Qwen3: 32,768 native).
    context_tokens: int = Field(default=32768, ge=1024)
    max_new_tokens: int = Field(ge=4, le=64)
    timeout_s: float = Field(gt=0)
    runs_dir: str                         # where training runs (adapters + manifests) live


class PracticeConfig(BaseModel):
    model_config = _STRICT
    items_path: str
    start_tier: int = Field(ge=1, le=3)
    max_attempts_per_exercise: int = Field(ge=1, le=10)
    history_in_prompt: int = Field(ge=0, le=10)
    sessions_dir: str
    demonstrations_path: str
    eval_dir: str
    default_model: str
    llm: LLMConfig


def load_practice(path: str | Path = "configs/practice.yaml") -> PracticeConfig:
    import yaml
    with open(resolve(path)) as f:
        return PracticeConfig(**yaml.safe_load(f))


# --------------------------------------------------------------------------------------------
# Exercise manifest

class Item(BaseModel):
    model_config = _LENIENT
    exercise_id: str
    image_path: str
    preferred_name: str
    accepted_alternatives: list[str] = Field(default_factory=list)
    tier: int = Field(ge=1, le=3)
    cues: dict[str, str]
    source: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _all_hints(self) -> "Item":
        missing = [c for c in CUES if c != "none" and c not in self.cues]
        if missing:
            raise ValueError(f"{self.exercise_id}: missing predefined hints {missing}")
        return self


class ItemBank:
    def __init__(self, items: list[Item], version: str) -> None:
        ids = [i.exercise_id for i in items]
        if len(set(ids)) != len(ids):
            raise ValueError("exercise_id values must be unique")
        self.items, self.version = items, version
        self.by_id = {i.exercise_id: i for i in items}

    def next_item(self, tier: int, shown: list[str]) -> Item:
        """Deterministic: in manifest order, the first item of `tier` never shown this session;
        otherwise the least recently shown one (never the picture just shown if another exists)."""
        candidates = [i for i in self.items if i.tier == tier]
        if not candidates:
            raise ValueError(f"no items in tier {tier}")
        unseen = [i for i in candidates if i.exercise_id not in shown]
        if unseen:
            return unseen[0]
        last_seen = {eid: k for k, eid in enumerate(shown)}
        ordered = sorted(candidates, key=lambda i: last_seen[i.exercise_id])
        if len(ordered) > 1 and shown and ordered[0].exercise_id == shown[-1]:
            return ordered[1]
        return ordered[0]

    def hint_text(self, exercise_id: str, cue: str) -> str | None:
        return None if cue == "none" else self.by_id[exercise_id].cues[cue]


@lru_cache(maxsize=4)
def load_items(path: str = "data/items.json") -> ItemBank:
    doc = json.loads(resolve(path).read_text())
    return ItemBank([Item(**i) for i in doc["items"]], doc.get("version", "unversioned"))


# --------------------------------------------------------------------------------------------
# Decision context and actions

class AttemptSummary(BaseModel):
    """One confirmed attempt, as the decision model sees it."""

    model_config = _LENIENT
    exercise_id: str
    word: str
    tier: int = Field(ge=1, le=3)
    hint: Cue                                  # hint visible during this attempt ("none" = independent)
    correct: bool
    error: ErrorCategory | None = None         # None iff correct
    response_time_s: float = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> "AttemptSummary":
        if self.correct != (self.error is None):
            raise ValueError("error must be None iff the attempt was correct")
        return self


class DecisionContext(BaseModel):
    """Everything a decision is based on. `history` ends with the latest attempt, which is on the
    current exercise."""

    model_config = _LENIENT
    current: CurrentExercise
    history: list[AttemptSummary] = Field(min_length=1)
    session_minutes: float = Field(ge=0)

    @model_validator(mode="after")
    def _latest_on_current(self) -> "DecisionContext":
        if self.history[-1].exercise_id != self.current.exercise_id:
            raise ValueError("the latest attempt must be on the current exercise")
        return self

    @property
    def latest(self) -> AttemptSummary:
        return self.history[-1]


class PracticeAction(BaseModel):
    """A decision. retry_current: same picture with `cue` as the hint (tier is the picture's own).
    next_exercise: a new picture at `tier`, shown without a hint."""

    model_config = _STRICT
    operation: Operation
    cue: Cue = "none"
    tier: int | None = Field(default=None, ge=1, le=3)

    @model_validator(mode="after")
    def _shape(self) -> "PracticeAction":
        if self.operation == "retry_current" and self.tier is not None:
            raise ValueError("retry_current keeps the picture's tier; tier must be omitted")
        if self.operation == "next_exercise" and (self.tier is None or self.cue != "none"):
            raise ValueError("next_exercise needs a tier and is shown without a hint")
        return self

    def key(self) -> str:
        return format_action(self)


def retry(cue: str) -> PracticeAction:
    return PracticeAction(operation="retry_current", cue=cue)


def next_ex(tier: int) -> PracticeAction:
    return PracticeAction(operation="next_exercise", tier=tier)


def valid_options(ctx: DecisionContext, approved_cues: list[str], max_tier_step: int,
                  max_attempts: int) -> list[PracticeAction]:
    """Hard constraints (never a preference): retry only after an incorrect attempt and while the
    picture has fewer than `max_attempts` attempts, with a clinician-approved hint; next only to a
    tier within `max_tier_step` of the current picture's tier."""
    opts: list[PracticeAction] = []
    if not ctx.latest.correct and ctx.current.attempts < max_attempts:
        opts += [retry(c) for c in CUES if c in approved_cues]
    t = ctx.current.tier
    opts += [next_ex(k) for k in (1, 2, 3) if abs(k - t) <= max_tier_step]
    return opts


def contract_options(ctx: DecisionContext, approved_cues: list[str], max_tier_step: int,
                     max_attempts: int) -> list[PracticeAction]:
    """Options under the API contract (section 5 and review answer 3): retry only after an incorrect
    attempt while tries are left, with an approved hint that has not already been shown on this
    picture ("none" is not a hint the model can choose); next to a tier within `max_tier_step`."""
    opts: list[PracticeAction] = []
    if not ctx.latest.correct and ctx.current.attempts < max_attempts:
        opts += [retry(c) for c in CUES
                 if c != "none" and c in approved_cues and c not in ctx.current.hints_shown]
    t = ctx.current.tier
    opts += [next_ex(k) for k in (1, 2, 3) if abs(k - t) <= max_tier_step]
    return opts


# --------------------------------------------------------------------------------------------
# Canonical text serialization (shared by data, training, evaluation and inference)

TASK_INSTRUCTION = (
    "You help run picture-naming practice for an adult practising word finding. After each "
    "confirmed attempt you choose what happens next: retry the same picture (optionally with one "
    "of its predefined hints), or move on to a new picture at a difficulty tier (1 easiest, "
    "3 hardest). Choose exactly one of the listed options and reply with that option exactly as "
    "written, with nothing else."
)


def format_action(a: PracticeAction) -> str:
    if a.operation == "retry_current":
        return json.dumps({"op": "retry", "hint": a.cue}, separators=(",", ":"))
    return json.dumps({"op": "next", "tier": a.tier}, separators=(",", ":"))


_JSON_OBJ = re.compile(r"\{[^{}]*\}")


def parse_action(text: str) -> PracticeAction | None:
    """The first well-formed action object in `text`, else None. Validity against the options is a
    separate check (`validate`)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    for m in _JSON_OBJ.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        try:
            if obj.get("op") == "retry" and set(obj) <= {"op", "hint"}:
                return retry(obj.get("hint", "none"))
            if obj.get("op") == "next" and set(obj) == {"op", "tier"} and isinstance(obj["tier"], int):
                return next_ex(obj["tier"])
        except ValueError:
            continue
        return None          # first object found but malformed: do not search further
    return None


def _attempt_line(a: AttemptSummary) -> str:
    hint = "no hint" if a.hint == "none" else f"hint {a.hint}"
    if a.correct:
        res = "correct"
    elif a.error == "unclassified":
        res = "incorrect (type not identified)"
    elif a.error == "no_response":
        res = "incorrect (no response)"
    else:
        res = f"incorrect ({a.error} error)"
    return f"{a.word}, tier {a.tier}, {hint}: {res}, {a.response_time_s:.1f} s"


def render_context(ctx: DecisionContext, options: list[PracticeAction], history_in_prompt: int) -> str:
    cur = ctx.current
    shown = ", ".join(cur.hints_shown) if cur.hints_shown else "none"
    earlier = ctx.history[:-1][-history_in_prompt:] if history_in_prompt else []
    lines = [
        f"Current picture: {cur.word} (tier {cur.tier})",
        f"Attempts on this picture: {cur.attempts}; hints already shown for it: {shown}",
        f"Latest attempt: {_attempt_line(ctx.latest)}",
        "Earlier attempts this session (oldest first):" if earlier else "Earlier attempts this session: none",
        *[f"- {_attempt_line(a)}" for a in earlier],
        f"Session time: {ctx.session_minutes:.0f} min",
        "Options:",
        *[format_action(o) for o in options],
    ]
    return "\n".join(lines)


def messages(ctx: DecisionContext, options: list[PracticeAction], history_in_prompt: int,
             examples: list[tuple[str, str]] | None = None) -> list[dict[str, str]]:
    """Chat messages for one decision. `examples` (rendered context, answer) turns this into the
    few-shot prompted baseline; they are training demonstrations only, never test situations."""
    msgs = [{"role": "system", "content": TASK_INSTRUCTION}]
    for ctx_text, answer in examples or []:
        msgs += [{"role": "user", "content": ctx_text}, {"role": "assistant", "content": answer}]
    msgs.append({"role": "user", "content": render_context(ctx, options, history_in_prompt)})
    return msgs


# --------------------------------------------------------------------------------------------
# Human-readable phrasing for the clinician view

CUE_PHRASES = {
    "none": "Try the same picture again, no hint",
    "first_sound": "Give the first sound",
    "first_letter": "Show the first letter",
    "sentence_frame": "Give a sentence to complete",
    "semantic": "Describe what it is",
}


def describe(a: PracticeAction, current_tier: int) -> str:
    if a.operation == "retry_current":
        return CUE_PHRASES[a.cue]
    if a.tier == current_tier:
        return f"Move on: new picture, same level (tier {a.tier})"
    word = "harder" if a.tier > current_tier else "easier"
    return f"Move on: {word} picture (tier {a.tier})"


# --------------------------------------------------------------------------------------------
# Bridge to the original State/Action contract (bandit, PPO and other tier/cue agents)

def to_state(ctx: DecisionContext, session_id: str | None, trial_count: int) -> State:
    trials = [TrialRecord(word=a.word, tier=a.tier, cue=a.hint, outcome=a.correct,
                          error_type="correct" if a.correct else a.error, response_time=a.response_time_s,
                          exercise_id=a.exercise_id)
              for a in ctx.history[-10:]]
    return State(session_id=session_id, last_trials=trials, current_tier=ctx.current.tier,
                 trial_count=trial_count, session_minutes=ctx.session_minutes, current_exercise=ctx.current)


def from_tier_cue(tier: int, cue: str, options: list[PracticeAction]) -> PracticeAction:
    """Deterministic mapping of a (tier, cue) answer from an original-contract agent onto the
    practice options: a hint means "retry this picture with that hint" when that is allowed;
    otherwise "next picture at that tier" (clamped to the allowed tiers)."""
    keys = {o.key(): o for o in options}
    if cue != "none" and retry(cue).key() in keys:
        return retry(cue)
    tiers = sorted(o.tier for o in options if o.operation == "next_exercise")
    return next_ex(min(tiers, key=lambda k: (abs(k - tier), k)))
