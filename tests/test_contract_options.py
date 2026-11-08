"""Contract option rule: a hint already shown on the picture is never offered again; no plain retry."""
from core.practice import DecisionContext, contract_options, format_action

APPROVED = ["none", "first_sound", "first_letter", "sentence_frame", "semantic"]


def ctx(hints, attempts, correct=False, tier=2):
    hist = [{"exercise_id": "ex-banana", "word": "banana", "tier": tier, "hint": h, "correct": False,
             "error": "meaning", "response_time_s": 5.0} for h in (["none"] + hints)[:attempts - 1]]
    hist.append({"exercise_id": "ex-banana", "word": "banana", "tier": tier, "hint": hints[-1] if hints else "none",
                 "correct": correct, "error": None if correct else "meaning", "response_time_s": 5.0})
    return DecisionContext(current={"exercise_id": "ex-banana", "word": "banana", "tier": tier,
                                    "hints_shown": hints, "attempts": attempts}, history=hist, session_minutes=5)


def keys(c, max_attempts=3):
    return [format_action(o) for o in contract_options(c, APPROVED, 1, max_attempts)]


def test_shown_hints_and_plain_retry_are_never_offered():
    k = keys(ctx(["semantic"], 2))
    assert '{"op":"retry","hint":"semantic"}' not in k and '{"op":"retry","hint":"none"}' not in k
    assert k == ['{"op":"retry","hint":"first_sound"}', '{"op":"retry","hint":"first_letter"}',
                 '{"op":"retry","hint":"sentence_frame"}', '{"op":"next","tier":1}', '{"op":"next","tier":2}',
                 '{"op":"next","tier":3}']


def test_only_next_after_a_correct_answer_or_when_no_tries_are_left():
    assert keys(ctx([], 1, correct=True)) == ['{"op":"next","tier":1}', '{"op":"next","tier":2}', '{"op":"next","tier":3}']
    assert all('"next"' in x for x in keys(ctx(["semantic", "first_sound"], 3)))
    assert keys(ctx([], 1, correct=True, tier=1)) == ['{"op":"next","tier":1}', '{"op":"next","tier":2}']
