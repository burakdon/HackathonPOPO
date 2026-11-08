"""Converter between the team's agreed JSON format and the DITTO trainer's format.

Team format (one decision):
  patient_state.json   {"patient_id", "session_id", "state_version",
                        "current_exercise": {"exercise_id", "target_word", "difficulty_tier"},
                        "recent_attempts": [{"attempt_id", "exercise_id", "transcript", "result", "hint_used"}],
                        "clinician_guidance": [],
                        "available_actions": [{"action_id", "operation", "exercise_id", "hint_type"}]}
  demonstration.json   {"example_id", "state_snapshot_id", "preferred_action_id", "clinician_note",
                        "scope", "patient_id", "provenance"}, joined with the saved patient_state that
                        `state_snapshot_id` points to
  model output         {"action_id": "<one of available_actions>"}

Trainer format (training/ditto.py, core/practice.py): a DecisionContext (current picture, attempt
history, session minutes), the valid options as canonical action strings ({"op":"retry","hint":...}
or {"op":"next","tier":...}), and for training rows the preferred and accepted actions.

The model input is built from the patient_state alone (render_context: current picture, attempts,
options). The demonstration's preferred action and clinician note are never part of it: every
available action is listed, in the order given, whichever one was preferred.

Field gaps (also listed in FIELD_GAPS and docs/loading_the_adapter.md):
  team -> trainer, not provided by the team format, filled as stated:
    attempt tier and target word for attempts on other pictures -> looked up in data/items.json
    response_time_s per attempt          -> optional extra field, else DEFAULT_RESPONSE_TIME_S
    session_minutes                      -> optional extra field, else 0
    tier of a next_exercise action       -> looked up in data/items.json by its exercise_id
    accepted actions (evaluation only)   -> [preferred action]
    split, family_id, rule               -> "train", state_snapshot_id, None
  trainer -> team, not used by the trainer (kept in provenance, not shown to the model):
    patient_id, session_id, state_version, attempt_id, transcript, clinician_guidance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import resolve
from core.practice import (DecisionContext, ItemBank, PracticeAction, format_action, load_items, messages,
                           next_ex, parse_action, retry)
from core.schema import CUES

# Rule A/B of the demonstrator splits on 4 s. A missing time is rendered as 5 s, i.e. "not a fast
# answer", so a missing field can never trigger a tier increase on its own (trade-off: a genuinely
# fast answer without a time is treated as slow).
DEFAULT_RESPONSE_TIME_S = 5.0
OPERATIONS = ("retry_current", "next_exercise")
# Team `result` values -> (correct, trainer error category).
RESULTS: dict[str, tuple[bool, str | None]] = {
    "correct": (True, None),
    "sound": (False, "sound"), "sound_error": (False, "sound"), "phonemic": (False, "sound"),
    "meaning": (False, "meaning"), "meaning_error": (False, "meaning"), "semantic": (False, "meaning"),
    "no_response": (False, "no_response"),
    "incorrect": (False, "unclassified"), "unclassified": (False, "unclassified"),
}
FIELD_GAPS = {
    "missing_in_team_format": ["attempt tier (items.json)", "attempt target word for other pictures (items.json)",
                               "response_time_s (optional extra, default 5 s)",
                               "session_minutes (optional extra, default 0)",
                               "tier of next_exercise actions (items.json)", "accepted actions (= preferred)",
                               "split (= train)", "family_id (= state_snapshot_id)", "rule (None)"],
    "unused_by_trainer": ["patient_id", "session_id", "state_version", "attempt_id", "transcript",
                          "clinician_guidance"],
}


def _hint(value: str | None) -> str:
    cue = value or "none"
    if cue not in CUES:
        raise ValueError(f"unknown hint {value!r}; expected one of {CUES}")
    return cue


def _tier_of(exercise_id: str, bank: ItemBank, fallback: int | None = None) -> int:
    if exercise_id in bank.by_id:
        return bank.by_id[exercise_id].tier
    if fallback is None:
        raise ValueError(f"exercise {exercise_id!r} is not in the item bank and has no tier")
    return fallback


def map_action(a: dict, bank: ItemBank) -> PracticeAction:
    """One team available_action -> trainer action."""
    op = a["operation"]
    if op == "retry_current":
        return retry(_hint(a.get("hint_type")))
    if op == "next_exercise":
        return next_ex(_tier_of(a["exercise_id"], bank))
    raise ValueError(f"unknown operation {op!r}; expected one of {OPERATIONS}")


def to_trainer(state: dict, bank: ItemBank | None = None) -> tuple[DecisionContext, list[PracticeAction], dict[str, str]]:
    """patient_state -> (DecisionContext, options in the given order, canonical action -> action_id).

    Several available actions can map to the same trainer action (two new pictures at the same
    tier); the model cannot tell them apart, so the first one listed is used."""
    bank = bank or load_items()
    cur = state["current_exercise"]
    cur_id, cur_tier = cur["exercise_id"], int(cur["difficulty_tier"])
    history, hints_on_current = [], []
    for att in state["recent_attempts"]:
        if att["result"] not in RESULTS:
            raise ValueError(f"unknown result {att['result']!r}; expected one of {sorted(RESULTS)}")
        correct, error = RESULTS[att["result"]]
        eid, hint = att["exercise_id"], _hint(att.get("hint_used"))
        on_current = eid == cur_id
        if on_current and hint != "none":
            hints_on_current.append(hint)
        history.append({
            "exercise_id": eid,
            "word": cur["target_word"] if on_current else (bank.by_id[eid].preferred_name if eid in bank.by_id else eid),
            "tier": cur_tier if on_current else _tier_of(eid, bank, cur_tier),
            "hint": hint, "correct": correct, "error": error,
            "response_time_s": float(att.get("response_time_s", DEFAULT_RESPONSE_TIME_S)),
        })
    ctx = DecisionContext(
        current={"exercise_id": cur_id, "word": cur["target_word"], "tier": cur_tier,
                 "hints_shown": list(dict.fromkeys(hints_on_current)),
                 "attempts": sum(1 for a in state["recent_attempts"] if a["exercise_id"] == cur_id)},
        history=history, session_minutes=float(state.get("session_minutes", 0.0)))
    options, ids = [], {}
    for a in state["available_actions"]:
        act = map_action(a, bank)
        key = format_action(act)
        if key not in ids:
            ids[key] = a["action_id"]
            options.append(act)
    return ctx, options, ids


def model_messages(state: dict, history_in_prompt: int, bank: ItemBank | None = None) -> list[dict[str, str]]:
    """The exact chat messages the trained model sees for this patient_state."""
    ctx, options, _ = to_trainer(state, bank)
    return messages(ctx, options, history_in_prompt)


def to_action_id(model_text: str, state: dict, bank: ItemBank | None = None) -> dict | None:
    """Model output text -> {"action_id": ...}, or None if it is not one of the available actions
    (the caller then falls back, as agents/llm_policy.py does)."""
    _, _, ids = to_trainer(state, bank)
    a = parse_action(model_text)
    if a is None or a.key() not in ids:
        return None
    return {"action_id": ids[a.key()]}


def demonstration_to_row(demo: dict, state: dict, bank: ItemBank | None = None) -> dict:
    """demonstration.json + the patient_state it points to -> one trainer JSONL row."""
    ctx, options, ids = to_trainer(state, bank)
    by_id = {a["action_id"]: a for a in state["available_actions"]}
    if demo["preferred_action_id"] not in by_id:
        raise ValueError(f"{demo['example_id']}: preferred_action_id {demo['preferred_action_id']!r} "
                         "is not one of the state's available_actions")
    preferred = format_action(map_action(by_id[demo["preferred_action_id"]], bank or load_items()))
    return {
        "example_id": demo["example_id"], "split": "train", "family_id": demo["state_snapshot_id"],
        "context": ctx.model_dump(), "options": [format_action(o) for o in options],
        "preferred": preferred, "accepted": [preferred], "rule": None,
        "note": demo.get("clinician_note"), "scope": demo.get("scope", "global"),
        "provenance": {**(demo.get("provenance") or {}), "patient_id": demo.get("patient_id"),
                       "session_id": state.get("session_id"), "state_version": state.get("state_version"),
                       "state_snapshot_id": demo["state_snapshot_id"],
                       "preferred_action_id": demo["preferred_action_id"]},
    }


def row_to_team(row: dict, patient_id: str = "unknown", session_id: str = "unknown",
                bank: ItemBank | None = None) -> tuple[dict, dict]:
    """Trainer row -> (patient_state, demonstration). Action ids are generated ("a1", "a2", ...); a
    next_exercise action points at the item bank's deterministic next picture of that tier."""
    bank = bank or load_items()
    ctx = DecisionContext(**row["context"])
    shown = [a.exercise_id for a in ctx.history]
    actions = []
    for k, text in enumerate(row["options"], 1):
        a = parse_action(text)
        actions.append({"action_id": f"a{k}", "operation": a.operation,
                        "exercise_id": ctx.current.exercise_id if a.operation == "retry_current"
                        else bank.next_item(a.tier, shown).exercise_id,
                        "hint_type": a.cue})
    snapshot = f"{row['example_id']}-state"
    state = {
        "patient_id": patient_id, "session_id": session_id, "state_version": 1,
        "current_exercise": {"exercise_id": ctx.current.exercise_id, "target_word": ctx.current.word,
                             "difficulty_tier": ctx.current.tier},
        "recent_attempts": [{"attempt_id": f"t{k}", "exercise_id": a.exercise_id, "transcript": None,
                             "result": "correct" if a.correct else ("incorrect" if a.error == "unclassified" else a.error),
                             "hint_used": a.hint, "response_time_s": a.response_time_s}
                            for k, a in enumerate(ctx.history, 1)],
        "session_minutes": ctx.session_minutes, "clinician_guidance": [], "available_actions": actions,
    }
    preferred_id = next(x["action_id"] for x, t in zip(actions, row["options"]) if t == row["preferred"])
    demo = {"example_id": row["example_id"], "state_snapshot_id": snapshot, "preferred_action_id": preferred_id,
            "clinician_note": row.get("note"), "scope": row.get("scope", "global"), "patient_id": patient_id,
            "provenance": row.get("provenance", {})}
    return state, demo


def main() -> None:
    ap = argparse.ArgumentParser(description="team demonstration.json files -> trainer JSONL (for --extra-demos)")
    ap.add_argument("demonstrations", nargs="+", help="demonstration.json files")
    ap.add_argument("--states", required=True, help="directory with <state_snapshot_id>.json patient states")
    ap.add_argument("--out", default="runs/demonstrations/team.jsonl")
    args = ap.parse_args()
    rows = []
    for path in args.demonstrations:
        demo = json.loads(Path(path).read_text())
        state = json.loads((Path(args.states) / f"{demo['state_snapshot_id']}.json").read_text())
        rows.append(demonstration_to_row(demo, state))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
