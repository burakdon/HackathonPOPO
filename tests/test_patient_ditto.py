"""Per-patient DITTO entry point: anchors never pin situations of the clinician's kind; mixing; app options."""
from core.practice import DecisionContext
from training.patient_ditto import app_options, mix, same_kind


def ctx(hints, attempts, correct=False, minutes=5):
    hist = [{"exercise_id": "ex-glasses", "word": "glasses", "tier": 2, "hint": h, "correct": False,
             "error": "meaning", "response_time_s": 5.0} for h in (["none"] + hints)[:attempts - 1]]
    hist.append({"exercise_id": "ex-glasses", "word": "glasses", "tier": 2, "hint": hints[-1] if hints else "none",
                 "correct": correct, "error": None if correct else "meaning", "response_time_s": 5.0})
    return DecisionContext(current={"exercise_id": "ex-glasses", "word": "glasses", "tier": 2, "hints_shown": hints,
                                    "attempts": attempts}, history=hist, session_minutes=minutes)


def test_same_kind_covers_the_clinicians_situation_and_nothing_else():
    example = ctx(["semantic"], 2)                                 # meaning hint failed
    assert same_kind(ctx(["semantic"], 2), example)
    assert same_kind(ctx(["semantic", "first_sound"], 3), example)  # still a failed meaning hint
    assert not same_kind(ctx([], 1), example)                     # first try, no hint yet
    assert not same_kind(ctx(["sentence_frame"], 2), example)     # a different hint failed
    assert not same_kind(ctx(["semantic"], 2, correct=True), example)


def test_mix_makes_patient_rows_about_a_third_with_distinct_ids():
    patient = [{"example_id": f"x{i}"} for i in range(3)]
    anchors = [{"example_id": f"a{i}"} for i in range(30)]
    rows, reps = mix(patient, anchors, 0.34)
    n_patient = sum(r["example_id"].startswith("x") for r in rows)
    assert reps == 6 and n_patient == 18 and 0.3 < n_patient / len(rows) < 0.4
    assert len({r["example_id"] for r in rows}) == len(rows)
    assert mix(patient, [], 0.34) == (patient, 1)


def test_app_options_are_hints_only_while_tries_are_left():
    assert app_options(ctx(["semantic"], 2)) == ['{"op":"retry","hint":"first_sound"}', '{"op":"retry","hint":"first_letter"}',
                                                 '{"op":"retry","hint":"sentence_frame"}']
    assert app_options(ctx([], 1, correct=True)) == ['{"op":"next","tier":1}', '{"op":"next","tier":2}', '{"op":"next","tier":3}']
