"""Per-patient adapters (API contract section 8): rule matching and anchors.

A per-patient adapter starts from the shared adapter and is trained on that patient's clinician
examples only. Trained on those alone it over-generalises (results/patient_steering/README_score.md:
after 3-8 examples of "meaning hint failed -> first sound" it retried with the first sound in every
situation that allowed a retry). Anchors fix that: the shared adapter's own choices on training-split
situations that do NOT match the patient's confirmed rules are added as examples, so the update says
"do this here, and keep doing what you did elsewhere".
"""
from __future__ import annotations

import math

from core.config import load_clinical
from core.practice import DecisionContext, contract_options, format_action, messages, parse_action

# ---- rules (contract section 7, "Rule format")


def failed_on_picture(ctx: DecisionContext) -> int:
    return sum(1 for a in ctx.history if a.exercise_id == ctx.current.exercise_id and not a.correct)


def missed_pictures_in_a_row(ctx: DecisionContext) -> int:
    """Pictures before the current one that ended without a correct answer, counted backwards."""
    last_by_picture: list[tuple[str, bool]] = []
    for a in ctx.history:
        if a.exercise_id == ctx.current.exercise_id:
            continue
        if last_by_picture and last_by_picture[-1][0] == a.exercise_id:
            last_by_picture[-1] = (a.exercise_id, a.correct)
        else:
            last_by_picture.append((a.exercise_id, a.correct))
    n = 0
    for _, ok in reversed(last_by_picture):
        if ok:
            break
        n += 1
    return n


def _in_range(value: int, spec: dict | None) -> bool:
    return spec is None or (value >= spec.get("min", -math.inf) and value <= spec.get("max", math.inf))


def matches(ctx: DecisionContext, when: dict) -> bool:
    """Every given part of a contract rule's `when` holds for this decision context."""
    latest = "correct" if ctx.latest.correct else ctx.latest.error
    shown = set(ctx.current.hints_shown)
    return ((not when.get("last_error") or latest in when["last_error"])
            and _in_range(failed_on_picture(ctx), when.get("failed_attempts_on_picture"))
            and set(when.get("hints_tried_include", [])) <= shown
            and not set(when.get("hints_tried_exclude", [])) & shown
            and _in_range(missed_pictures_in_a_row(ctx), when.get("missed_pictures_in_a_row"))
            and (not when.get("level") or ctx.current.tier in when["level"]))


def rule_action(ctx: DecisionContext, then: dict) -> str:
    """A rule's `then` as a canonical action for this context (relative level -> absolute tier)."""
    if then["op"] == "retry":
        return format_action(parse_action(f'{{"op":"retry","hint":"{then["hint"]}"}}'))
    t = ctx.current.tier + {"easier": -1, "same": 0, "harder": 1}[then["level"]]
    return format_action(parse_action(f'{{"op":"next","tier":{min(3, max(1, t))}}}'))


# ---- anchors


def options_for(ctx: DecisionContext, max_attempts: int = 3) -> list[str]:
    c = load_clinical()
    return [format_action(o) for o in contract_options(ctx, c.approved_cues, c.max_tier_step, max_attempts)]


def anchor_rows(runtime, adapter: str, contexts: list[DecisionContext], rules: list[dict], history_in_prompt: int,
                limit: int | None = None, fmt: str = "v2") -> list[dict]:
    """The shared adapter's choice (most likely valid option) on each context that matches none of the
    patient's rules, as training rows with that choice preferred."""
    rows = []
    for k, ctx in enumerate(contexts):
        if any(matches(ctx, r["when"]) for r in rules):
            continue
        opts = options_for(ctx)
        lps, _ = runtime.score(messages(ctx, [parse_action(o) for o in opts], history_in_prompt, fmt=fmt), opts, adapter)
        choice = opts[max(range(len(opts)), key=lps.__getitem__)]
        rows.append({"example_id": f"anchor-{k:03d}", "split": "train", "family_id": f"anchor-{k:03d}",
                     "context": ctx.model_dump(), "options": opts, "preferred": choice, "accepted": [choice],
                     "rule": "anchor", "note": "shared adapter's own choice (keeps behaviour outside the rules)",
                     "scope": "patient", "provenance": {"source": "anchor: shared adapter choice"}})
        if limit and len(rows) >= limit:
            break
    return rows


def mix(patient_rows: list[dict], anchors: list[dict], patient_share: float = 1 / 3) -> list[dict]:
    """Patient examples repeated (with distinct ids) so they make up about `patient_share` of the rows."""
    if not anchors:
        return list(patient_rows)
    reps = max(1, math.ceil(patient_share * len(anchors) / ((1 - patient_share) * len(patient_rows))))
    out = [{**r, "example_id": f"{r['example_id']}-r{j}"} for j in range(reps) for r in patient_rows]
    return out + anchors
