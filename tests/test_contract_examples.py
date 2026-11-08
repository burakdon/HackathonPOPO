"""The committed contract samples are current, match their models, and the mock server serves them."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

import core.contract as C
from core.config import resolve
from scripts.build_contract_examples import build

EXAMPLES = resolve("docs/contract_examples")


def _files(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.json"))}


def test_committed_samples_are_current(tmp_path):
    build(tmp_path)
    assert _files(tmp_path) == _files(EXAMPLES), "rerun: uv run python -m scripts.build_contract_examples"


def _model(name: str):
    if name.startswith("list["):
        return TypeAdapter(list[getattr(C, name[5:-1])])
    return TypeAdapter(getattr(C, name))


@pytest.mark.parametrize("entry", json.loads((EXAMPLES / "index.json").read_text())["endpoints"],
                         ids=lambda e: e["file"])
def test_every_sample_matches_its_model(entry):
    _model(entry["model"]).validate_python(json.loads((EXAMPLES / entry["file"]).read_text()))
    if "request_file" in entry:
        _model(entry["request_model"]).validate_python(json.loads((EXAMPLES / entry["request_file"]).read_text()))


def test_scripted_session_walks_the_whole_flow():
    index = json.loads((EXAMPLES / "index.json").read_text())
    steps = [C.AttemptResult(**json.loads((EXAMPLES / f).read_text())) for f in index["practice_script"]]
    kinds = [s.next.kind for s in steps]
    assert set(kinds) == {"correct", "hint", "try_again", "not_assessed", "show_word"}
    assert steps[-1].next.session_complete and steps[-1].next.summary.counts.not_assessed == 1
    assert all(not s.counted for s in steps if s.score.outcome == "unclear")


def test_summary_is_per_picture():
    review = C.SessionReview(**json.loads((EXAMPLES / "review/GET_session_review.json").read_text()))
    c = review.summary.counts
    assert (c.independent, c.assisted, c.unsuccessful, c.not_assessed) == (1, 3, 1, 1)
    assert review.summary.named_with_hint == 0.6          # 3 of 5 assessed pictures


def test_contract_rejects_inconsistent_shapes():
    sig = C.Signals()
    with pytest.raises(ValidationError):                  # incorrect needs an error type
        C.Score(outcome="incorrect", error_type=None, heard_word="x", matched_name=None, reason="", signals=sig)
    with pytest.raises(ValidationError):                  # unclear must not be counted
        C.AttemptResult(attempt_id="a", presentation_id="p", counted=True, transcript="",
                        score=C.Score(outcome="unclear", error_type=None, heard_word=None, matched_name=None,
                                      reason="", signals=sig),
                        next=C.NextStep(kind="try_again", message=""))
    with pytest.raises(ValidationError):                  # a hint step needs its decision
        C.NextStep(kind="hint", message="", hint=C.ShownHint(type="semantic", text="t", audio_url=None),
                   attempt_number=2)


def test_rule_matching_and_relative_levels():
    review = C.SessionReview(**json.loads((EXAMPLES / "review/GET_session_review.json").read_text()))
    ds = [d for p in review.pictures for d in p.decisions]
    rule = C.Rule(when=C.RuleWhen(hints_tried_include=["semantic"], failed_attempts_on_picture=C.Range(min=2)),
                  then=C.RetryThen(op="retry", hint="first_sound"), summary="")
    assert [d.decision_id for d in ds if C.rule_matches(rule, d.context)] == ["s-demo-01-d07", "s-demo-01-d08"]
    easier = C.Rule(when=C.RuleWhen(), then=C.NextThen(op="next", level="easier"), summary="")
    assert C.rule_action(easier, ds[0].context) == C.NextAction(op="next", tier=1)   # never below level 1


def test_mock_server_serves_samples_and_steps_through_the_script():
    from service.contract_mock import create_mock_app
    c = TestClient(create_mock_app())
    assert len(c.get("/pictures").json()) == 24
    assert c.get("/media/images/cat.svg").status_code == 200
    assert c.get("/media/hints/ex-cat/first_sound.m4a").headers["content-type"] == "audio/wav"
    started = c.post("/patients/anyone/sessions").json()
    assert started["picture"]["word"] == "cat"
    first = c.post("/sessions/x/recordings")
    assert first.json()["next"]["kind"] == "correct" and first.headers["X-Mock-Step"] == "1/13"
    for _ in range(12):
        last = c.post("/sessions/x/recordings")
    assert last.json()["next"]["session_complete"] is True
    assert c.post("/sessions/x/recordings").status_code == 409
    c.post("/patients/anyone/sessions")                   # rewinds
    assert c.post("/sessions/x/recordings").headers["X-Mock-Step"] == "1/13"
    assert c.get("/nothing/here").status_code == 404
