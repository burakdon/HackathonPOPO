"""Rule matching in the contract's rule format and the patient/anchor mix (training/patient_adapter.py)."""
from core.practice import DecisionContext
from training.patient_adapter import matches, missed_pictures_in_a_row, mix, rule_action


def ctx(hints, attempts, correct=False, tier=2, before=()):
    hist = [{"exercise_id": e, "word": e[3:], "tier": 1, "hint": "none", "correct": ok,
             "error": None if ok else "meaning", "response_time_s": 4.0} for e, ok in before]
    hist += [{"exercise_id": "ex-chair", "word": "chair", "tier": tier, "hint": h, "correct": False, "error": "meaning",
              "response_time_s": 5.0} for h in (["none"] + hints)[:attempts - 1]]
    hist.append({"exercise_id": "ex-chair", "word": "chair", "tier": tier, "hint": hints[-1] if hints else "none",
                 "correct": correct, "error": None if correct else "meaning", "response_time_s": 5.0})
    return DecisionContext(current={"exercise_id": "ex-chair", "word": "chair", "tier": tier, "hints_shown": hints,
                                    "attempts": attempts}, history=hist, session_minutes=5)


def test_rule_matching_follows_the_contract_fields():
    when = {"hints_tried_include": ["semantic"], "failed_attempts_on_picture": {"min": 2}}
    assert matches(ctx(["semantic"], 2), when)
    assert not matches(ctx([], 1), when)
    assert not matches(ctx(["semantic"], 2), {**when, "hints_tried_exclude": ["semantic"]})
    assert matches(ctx([], 1, correct=True), {"last_error": ["correct"], "level": [2]})
    c = ctx([], 1, before=[("ex-dog", False), ("ex-dog", False), ("ex-car", False), ("ex-key", True)])
    assert missed_pictures_in_a_row(c) == 0
    c = ctx([], 1, before=[("ex-key", True), ("ex-dog", False), ("ex-car", False)])
    assert missed_pictures_in_a_row(c) == 2 and matches(c, {"missed_pictures_in_a_row": {"min": 2}})


def test_rule_action_resolves_relative_levels():
    assert rule_action(ctx([], 1), {"op": "retry", "hint": "first_letter"}) == '{"op":"retry","hint":"first_letter"}'
    assert rule_action(ctx([], 1, tier=1), {"op": "next", "level": "easier"}) == '{"op":"next","tier":1}'
    assert rule_action(ctx([], 1, tier=2), {"op": "next", "level": "harder"}) == '{"op":"next","tier":3}'


def test_mix_repeats_patient_rows_to_about_a_third():
    rows = mix([{"example_id": "x1"}], [{"example_id": f"a{i}"} for i in range(10)])
    assert sum(r["example_id"].startswith("x1") for r in rows) == 5 and len({r["example_id"] for r in rows}) == len(rows)
