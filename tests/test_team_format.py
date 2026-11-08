"""Team JSON format <-> trainer format (training/team_format.py). No model needed."""
import json

import pytest

from core.config import resolve
from core.practice import format_action
from training.ditto import example_messages, load_examples
from training.team_format import (demonstration_to_row, model_messages, row_to_team, to_action_id,
                                  to_trainer)

STATE = {
    "patient_id": "p-001", "session_id": "s-001", "state_version": 3,
    "current_exercise": {"exercise_id": "ex-banana", "target_word": "banana", "difficulty_tier": 2},
    "recent_attempts": [
        {"attempt_id": "t1", "exercise_id": "ex-cat", "transcript": "cat", "result": "correct", "hint_used": None},
        {"attempt_id": "t2", "exercise_id": "ex-banana", "transcript": "bandana", "result": "sound", "hint_used": "none"},
    ],
    "clinician_guidance": [],
    "available_actions": [
        {"action_id": "act-fs", "operation": "retry_current", "exercise_id": "ex-banana", "hint_type": "first_sound"},
        {"action_id": "act-sem", "operation": "retry_current", "exercise_id": "ex-banana", "hint_type": "semantic"},
        {"action_id": "act-n1", "operation": "next_exercise", "exercise_id": "ex-dog", "hint_type": None},
        {"action_id": "act-n2", "operation": "next_exercise", "exercise_id": "ex-chair", "hint_type": None},
        {"action_id": "act-n2b", "operation": "next_exercise", "exercise_id": "ex-rabbit", "hint_type": None},
    ],
}
DEMO = {"example_id": "team-001", "state_snapshot_id": "snap-001", "preferred_action_id": "act-fs",
        "clinician_note": "sound error, so give the first sound", "scope": "global", "patient_id": "p-001",
        "provenance": {"source": "dashboard override"}}


def test_state_maps_to_trainer_context_and_options():
    ctx, options, ids = to_trainer(STATE)
    assert ctx.current.word == "banana" and ctx.current.tier == 2 and ctx.current.attempts == 1
    assert ctx.history[0].word == "cat" and ctx.history[0].tier == 1          # looked up in items.json
    assert ctx.latest.error == "sound" and ctx.latest.response_time_s == 5.0
    keys = [format_action(o) for o in options]
    assert keys == ['{"op":"retry","hint":"first_sound"}', '{"op":"retry","hint":"semantic"}',
                    '{"op":"next","tier":1}', '{"op":"next","tier":2}']
    assert ids['{"op":"next","tier":2}'] == "act-n2"                          # first of two same-tier actions


def test_model_output_maps_back_to_an_available_action_id():
    assert to_action_id('{"op":"retry","hint":"semantic"}', STATE) == {"action_id": "act-sem"}
    assert to_action_id('{"op":"next","tier":1}', STATE) == {"action_id": "act-n1"}
    assert to_action_id('{"op":"next","tier":3}', STATE) is None              # not offered
    assert to_action_id("no idea", STATE) is None


def test_demonstration_becomes_a_trainer_row_without_leaking_the_preference():
    row = demonstration_to_row(DEMO, STATE)
    assert row["preferred"] == '{"op":"retry","hint":"first_sound"}' and row["split"] == "train"
    assert row["scope"] == "global" and row["provenance"]["preferred_action_id"] == "act-fs"
    # The model input is identical whichever action was preferred, and never carries the note.
    other = demonstration_to_row({**DEMO, "preferred_action_id": "act-n1"}, STATE)
    prompt = json.dumps(example_messages(row, 4))
    assert prompt == json.dumps(example_messages(other, 4)) == json.dumps(model_messages(STATE, 4))
    assert DEMO["clinician_note"] not in prompt and "act-fs" not in prompt


def test_unknown_preferred_action_or_result_is_rejected():
    with pytest.raises(ValueError, match="not one of"):
        demonstration_to_row({**DEMO, "preferred_action_id": "nope"}, STATE)
    bad = {**STATE, "recent_attempts": [{**STATE["recent_attempts"][1], "result": "maybe"}]}
    with pytest.raises(ValueError, match="unknown result"):
        to_trainer(bad)


def test_every_shipped_demonstration_round_trips_through_the_team_format():
    rows = load_examples(resolve("data/demonstrations/demonstrator_a_v1.jsonl"), None)
    for row in rows:
        state, demo = row_to_team(row)
        back = demonstration_to_row(demo, state)
        assert back["preferred"] == row["preferred"], row["example_id"]
        assert back["options"] == row["options"], row["example_id"]
        assert example_messages(back, 4) == example_messages(row, 4), row["example_id"]


def test_decide_action_id_returns_the_most_likely_available_action():
    from training.team_format import decide_action_id

    class Runtime:
        def score(self, msgs, keys, adapter=None):
            self.keys, self.text = keys, msgs[-1]["content"]
            return [-5.0, -4.0, -3.0, -1.0], 0.01            # next tier 2 is the most likely

    rt = Runtime()
    assert decide_action_id(rt, STATE, "ditto", 4) == {"action_id": "act-n2"}
    assert " = new picture, same level (tier 2)" in rt.text            # prompt format v2
