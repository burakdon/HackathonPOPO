"""Build the fictional demonstration dataset (demonstrator A) with fixed splits.

    uv run python -m training.build_demonstrations      # -> data/demonstrations/demonstrator_a_v1.jsonl

Every example is one *underlying situation* (a scenario family of size one): splits are assigned
per family before anything is generated, no variants or paraphrases are made, and the current
picture of every test situation is a word that never appears as the current picture in training
or development. The preferred action comes from the written strategy of demonstrator A
(data/demonstrations/README.md). These are mock preferences drafted by an AI assistant from that
written strategy and are marked as pending human review; they are not clinical knowledge.

Because the preferences follow a written rule, a hand-coded rule would score 100% on them. The
experiment asks whether a language model picks the preference up from 20 demonstrations, not
whether a language model is needed for this rule.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from core.config import load_clinical, resolve
from core.practice import (AttemptSummary, CurrentExercise, DecisionContext, PracticeAction, format_action,
                           load_items, load_practice, next_ex, retry, valid_options)

OUT = "data/demonstrations/demonstrator_a_v1.jsonl"
FAST_S = 4.0          # demonstrator A: "fast" independent answer
LATE_MIN = 20.0       # demonstrator A: "late in the session"
ERROR_HINT = {"sound": "first_sound", "meaning": "semantic", "no_response": "sentence_frame",
              "unclassified": "sentence_frame"}


def demonstrator_a(ctx: DecisionContext, options: list[PracticeAction]) -> tuple[str, list[str], str]:
    """(rule id, accepted action keys with the preferred first, short description)."""
    allowed = {o.key() for o in options}
    L, cur = ctx.latest, ctx.current
    easier, harder = next_ex(max(1, cur.tier - 1)), next_ex(min(3, cur.tier + 1))
    if L.correct:
        if L.hint != "none":
            return "C", [next_ex(cur.tier).key()], "correct with a hint: new picture, same tier"
        if L.response_time_s <= FAST_S:
            return "A", [harder.key()], "fast independent answer: one tier harder"
        return "B", [next_ex(cur.tier).key()], "slow independent answer: new picture, same tier"
    if ctx.session_minutes >= LATE_MIN:
        return "G", [easier.key()], "error late in the session: easier picture instead of a hint"
    n_hints = len(cur.hints_shown)
    if n_hints >= 2 or retry("none").key() not in allowed:
        return "F", [easier.key()], "still incorrect after two hints: easier picture"
    if n_hints == 0:
        return f"D-{L.error}", [retry(ERROR_HINT[L.error]).key()], "first error: hint matched to the error"
    if "first_sound" not in cur.hints_shown:
        return "E", [retry("first_sound").key()], "error after one hint: give the first sound"
    return "E-alt", [retry("sentence_frame").key(), retry("semantic").key()], \
        "error after the first sound: sentence (or description) instead"


@dataclass
class Spec:
    family: str
    split: str
    word: str
    latest: tuple                      # ("ok", rt) or (error_category, rt)
    prior_on_current: list = field(default_factory=list)   # [(hint, error_category, rt), ...] before latest
    minutes: float = 8.0
    n_earlier: int = 3


# 35 hand-specified situations. Test pictures (kangaroo, glasses, apple, cactus, rabbit, key,
# helicopter, pretzel, tree, umbrella) are never the current picture in train or dev.
SPECS = [
    # ---- train (20)
    Spec("F01", "train", "cat", ("ok", 2.1), minutes=3), Spec("F02", "train", "banana", ("ok", 3.2), minutes=9),
    Spec("F03", "train", "octopus", ("ok", 1.9), minutes=12),                       # A at tier 3
    Spec("F04", "train", "chair", ("ok", 6.8), minutes=5), Spec("F05", "train", "dog", ("ok", 8.4), minutes=17),
    Spec("F06", "train", "hammer", ("ok", 5.5), prior_on_current=[("none", "sound", 7.0)]),
    Spec("F07", "train", "anchor", ("ok", 3.0), prior_on_current=[("none", "meaning", 6.1)], minutes=14),
    Spec("F08", "train", "house", ("sound", 5.2), minutes=4), Spec("F09", "train", "microscope", ("sound", 7.9), minutes=11),
    Spec("F10", "train", "carrot", ("meaning", 4.4), minutes=6), Spec("F11", "train", "giraffe", ("meaning", 6.0), minutes=16),
    Spec("F12", "train", "fish", ("no_response", 12.0), minutes=7),
    Spec("F13", "train", "bicycle", ("unclassified", 8.8), minutes=13),
    Spec("F14", "train", "car", ("meaning", 6.5), prior_on_current=[("none", "meaning", 5.0)], minutes=9),
    Spec("F15", "train", "car", ("unclassified", 9.1), prior_on_current=[("none", "no_response", 12.0)], minutes=5,
         n_earlier=1),
    Spec("F16", "train", "banana", ("sound", 6.2), prior_on_current=[("none", "sound", 6.0)], minutes=15),
    Spec("F17", "train", "chair", ("meaning", 7.0),
         prior_on_current=[("none", "meaning", 6.0), ("semantic", "meaning", 6.5)], minutes=10),
    Spec("F18", "train", "dog", ("no_response", 12.0),
         prior_on_current=[("none", "no_response", 12.0), ("sentence_frame", "no_response", 12.0)], minutes=6),
    Spec("F19", "train", "microscope", ("sound", 6.6), minutes=22),
    Spec("F20", "train", "fish", ("meaning", 5.1), prior_on_current=[("none", "meaning", 5.4)], minutes=26),
    # ---- dev (5)
    Spec("F21", "dev", "house", ("ok", 2.6), minutes=10),
    Spec("F22", "dev", "hammer", ("meaning", 5.7), minutes=8),
    Spec("F23", "dev", "octopus", ("unclassified", 7.2), prior_on_current=[("none", "sound", 6.0)], minutes=12),
    Spec("F24", "dev", "giraffe", ("sound", 6.3),
         prior_on_current=[("none", "sound", 6.9), ("first_sound", "sound", 6.1)], minutes=9),
    Spec("F25", "dev", "carrot", ("no_response", 12.0), minutes=24),
    # ---- test (10)
    Spec("F26", "test", "apple", ("ok", 2.4), minutes=6),
    Spec("F27", "test", "helicopter", ("ok", 2.2), minutes=11),                    # A at tier 3
    Spec("F28", "test", "glasses", ("ok", 7.3), minutes=13),
    Spec("F29", "test", "kangaroo", ("ok", 4.9), prior_on_current=[("none", "sound", 8.0)], minutes=9),
    Spec("F30", "test", "key", ("sound", 4.8), minutes=5),
    Spec("F31", "test", "cactus", ("unclassified", 9.5), minutes=14),
    Spec("F32", "test", "umbrella", ("meaning", 6.0), prior_on_current=[("none", "sound", 5.5)], minutes=7),
    Spec("F33", "test", "rabbit", ("sound", 6.4),
         prior_on_current=[("none", "meaning", 5.8), ("first_sound", "meaning", 6.0)], minutes=10),
    Spec("F34", "test", "pretzel", ("no_response", 12.0), prior_on_current=[("none", "no_response", 12.0),
                                                                          ("sentence_frame", "no_response", 12.0)],
         minutes=8),
    Spec("F35", "test", "tree", ("sound", 5.0), minutes=23),
]


def build_context(spec: Spec, bank, rng: np.random.Generator) -> DecisionContext:
    item = bank.by_id[f"ex-{spec.word}"]
    others = [i for i in bank.items if i.preferred_name != spec.word]
    history: list[AttemptSummary] = []
    for i in rng.choice(len(others), size=spec.n_earlier, replace=False):
        it = others[int(i)]
        ok = bool(rng.random() < 0.65)
        history.append(AttemptSummary(exercise_id=it.exercise_id, word=it.preferred_name, tier=it.tier, hint="none",
                                      correct=ok, error=None if ok else str(rng.choice(["sound", "meaning"])),
                                      response_time_s=round(float(rng.uniform(2.0, 7.0)), 1)))
    hints: list[str] = []
    for hint, err, rt in spec.prior_on_current:
        history.append(AttemptSummary(exercise_id=item.exercise_id, word=item.preferred_name, tier=item.tier,
                                      hint=hint, correct=False, error=err, response_time_s=rt))
        if hint != "none":
            hints.append(hint)
    # The latest attempt's hint is the one the demonstrator gave after the previous attempt.
    latest_hint = "none"
    if spec.prior_on_current:
        prev = spec.prior_on_current[-1][1]
        latest_hint = ERROR_HINT[prev] if len(spec.prior_on_current) == 1 else (
            "first_sound" if "first_sound" not in hints else "sentence_frame")
        if latest_hint in hints:      # never repeat a hint already shown on this picture
            latest_hint = "first_sound" if "first_sound" not in hints else "semantic"
        hints.append(latest_hint)
    res, rt = spec.latest
    ok = res == "ok"
    history.append(AttemptSummary(exercise_id=item.exercise_id, word=item.preferred_name, tier=item.tier,
                                  hint=latest_hint, correct=ok, error=None if ok else res, response_time_s=rt))
    cur = CurrentExercise(exercise_id=item.exercise_id, word=item.preferred_name, tier=item.tier,
                          hints_shown=hints, attempts=len(spec.prior_on_current) + 1)
    return DecisionContext(current=cur, history=history, session_minutes=spec.minutes)


def build() -> list[dict]:
    clinical, pcfg = load_clinical(), load_practice()
    bank = load_items(pcfg.items_path)
    rows = []
    for k, spec in enumerate(SPECS):
        rng = np.random.default_rng([2026, k])
        ctx = build_context(spec, bank, rng)
        opts = valid_options(ctx, clinical.approved_cues, clinical.max_tier_step, pcfg.max_attempts_per_exercise)
        rule, accepted, why = demonstrator_a(ctx, opts)
        keys = [format_action(o) for o in opts]
        assert all(a in keys for a in accepted), (spec.family, accepted, keys)
        rows.append({
            "example_id": f"demoA-v1-{spec.family}",
            "split": spec.split,
            "family_id": spec.family,
            "context": ctx.model_dump(),
            "options": keys,
            "preferred": accepted[0],
            "accepted": accepted,
            "rule": rule,
            "note": why,
            "scope": "global",
            "provenance": {"source": "team-authored (fictional scenario)",
                           "drafted_by": "AI assistant, from the written strategy of demonstrator A",
                           "review_status": "pending human review",
                           "demonstrator_id": "demonstrator-A"},
        })
    return rows


def dataset_hash(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()).hexdigest()[:16]


def main() -> None:
    rows = build()
    out = resolve(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    from collections import Counter
    print(f"wrote {len(rows)} examples to {OUT} (hash {dataset_hash(rows)})")
    print("splits:", dict(Counter(r["split"] for r in rows)))
    for split in ("train", "dev", "test"):
        print(split, dict(sorted(Counter(r["rule"] for r in rows if r["split"] == split).items())))


if __name__ == "__main__":
    main()
