"""Shared serialization, valid options, deterministic exercise resolution, speech normalisation."""
import numpy as np
import pytest

from core.observation import ObservationBuilder
from core.config import load_clinical
from core.practice import (AttemptSummary, DecisionContext, format_action, from_tier_cue, load_items, next_ex,
                           parse_action, retry, valid_options)
from core.schema import CurrentExercise, State
from speech.normalize import normalize, suggest_match


def ctx(correct=False, attempts=1, tier=1, hints=(), minutes=5.0, error="sound"):
    a = AttemptSummary(exercise_id="ex-cat", word="cat", tier=tier, hint=hints[-1] if hints else "none",
                       correct=correct, error=None if correct else error, response_time_s=3.0)
    return DecisionContext(current=CurrentExercise(exercise_id="ex-cat", word="cat", tier=tier, hints_shown=list(hints),
                                                   attempts=attempts), history=[a], session_minutes=minutes)


def test_action_roundtrip_and_strict_parsing():
    for a in (retry("first_sound"), retry("none"), next_ex(2)):
        assert parse_action(format_action(a)) == a
    assert parse_action('<think>x</think>{"op": "next", "tier": 3}') == next_ex(3)
    for bad in ('{"op":"next"}', '{"op":"retry","hint":"gesture"}', '{"op":"next","tier":2,"cue":"semantic"}',
                "next tier 2", '{"op":"jump","tier":1}'):
        assert parse_action(bad) is None


def test_valid_options_enforce_constraints():
    cl = load_clinical()
    ok = valid_options(ctx(correct=True), cl.approved_cues, cl.max_tier_step, 3)
    assert all(o.operation == "next_exercise" for o in ok) and {o.tier for o in ok} == {1, 2}
    wrong = valid_options(ctx(correct=False), cl.approved_cues, cl.max_tier_step, 3)
    assert retry("first_sound") in wrong and next_ex(3) not in wrong
    at_limit = valid_options(ctx(correct=False, attempts=3), cl.approved_cues, cl.max_tier_step, 3)
    assert all(o.operation == "next_exercise" for o in at_limit)
    restricted = valid_options(ctx(correct=False), ["none", "first_sound"], 1, 3)
    assert retry("semantic") not in restricted


def test_next_exercise_is_deterministic_unseen_first():
    bank = load_items()
    first = bank.next_item(1, [])
    assert bank.next_item(1, []) == first
    tier1 = [i.exercise_id for i in bank.items if i.tier == 1]
    second = bank.next_item(1, [first.exercise_id])
    assert second.exercise_id != first.exercise_id
    after_all = bank.next_item(1, tier1)          # all shown: least recently shown
    assert after_all.exercise_id == tier1[0]


def test_tier_cue_mapping_for_baseline_agents():
    cl = load_clinical()
    opts = valid_options(ctx(correct=False), cl.approved_cues, cl.max_tier_step, 3)
    assert from_tier_cue(1, "semantic", opts) == retry("semantic")
    assert from_tier_cue(2, "none", opts) == next_ex(2)
    assert from_tier_cue(3, "none", opts) == next_ex(2)     # clamped to an allowed tier
    after_ok = valid_options(ctx(correct=True), cl.approved_cues, cl.max_tier_step, 3)
    assert from_tier_cue(1, "first_sound", after_ok) == next_ex(1)


def test_unclassified_error_is_valid_and_has_no_observation_slot():
    s = State(last_trials=[{"word": "cat", "tier": 1, "cue": "none", "outcome": False, "error_type": "unclassified",
                            "response_time": 2.0}], current_tier=1, trial_count=1, session_minutes=1.0)
    ob = ObservationBuilder(load_clinical(), history=8, nominal_trials=40, rt_cap_s=20.0)(s)
    assert ob.shape == (ob.size,) and np.isfinite(ob).all()


def test_context_requires_latest_attempt_on_current_exercise():
    with pytest.raises(ValueError):
        DecisionContext(current=CurrentExercise(exercise_id="ex-dog", word="dog", tier=1),
                        history=ctx().history, session_minutes=1.0)


def test_normalisation_is_conservative_and_only_a_suggestion():
    assert normalize("Um... it's a CAT!") == ["cat"]
    assert suggest_match("a kitten", "cat", ["kitty", "kitten"]).suggestion == "matches"
    assert suggest_match("bicycles", "bicycle", []).suggestion == "matches"
    assert suggest_match("dog", "cat", []).suggestion == "no_match"
    assert suggest_match("tat", "cat", []).suggestion == "no_match"     # not labelled as a sound error
    assert suggest_match("uh", "cat", []).suggestion == "empty"
