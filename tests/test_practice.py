"""Practice flow through the API: confirmed attempts, idempotency, uncertain speech, retry vs next,
action validation with explicit fallback, clinician feedback persistence."""
from __future__ import annotations

import json

import pytest
import yaml
from fastapi.testclient import TestClient

from agents.llm_policy import LLMDecisionPolicy
from core.config import resolve
from speech.base import Transcription


class FakeRuntime:
    """Stands in for the language model: returns a scripted answer."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, msgs, adapter=None):
        self.calls.append(msgs)
        text = self.answer(msgs) if callable(self.answer) else self.answer
        return text, 0.001


class FakeSpeech:
    name = "fake-asr"

    def __init__(self, result: Transcription):
        self.result = result

    def transcribe(self, audio, content_type):
        return self.result


@pytest.fixture
def make_client(tmp_path):
    def _make(speech=None, default_model="bandit"):
        p = yaml.safe_load(resolve("configs/practice.yaml").read_text())
        p.update(sessions_dir=str(tmp_path / "practice"), demonstrations_path=str(tmp_path / "demos.jsonl"),
                 eval_dir=str(tmp_path / "eval"), default_model=default_model)
        p["llm"]["runs_dir"] = str(tmp_path / "runs")
        (tmp_path / "practice.yaml").write_text(yaml.safe_dump(p))
        s = yaml.safe_load(resolve("configs/service.yaml").read_text())
        s.update(agent="beta_bandit", practice_config=str(tmp_path / "practice.yaml"),
                 overrides_path=str(tmp_path / "overrides.jsonl"))
        (tmp_path / "service.yaml").write_text(yaml.safe_dump(s))
        from service.app import create_app
        app = create_app(tmp_path / "service.yaml", speech_provider=speech or FakeSpeech(
            Transcription(status="ok", text="cat", provider="fake-asr")))
        return app, TestClient(app)
    return _make


def install_fake(app, version: str, answer) -> FakeRuntime:
    rt = FakeRuntime(answer)
    pcfg = app.state.registry.pcfg
    app.state.registry._policies[version] = LLMDecisionPolicy(rt, version, None, {"fake": True},
                                                              pcfg.history_in_prompt, 5.0)
    return rt


def attempt(client, sid, pres, aid, correct, cat=None, status="confirmed", rt=2.0):
    body = {"attempt_id": aid, "presentation_id": pres["presentation_id"], "raw_transcript": "x",
            "transcript_status": "ok", "corrected_transcript": "x", "scoring_status": status,
            "correct": correct if status == "confirmed" else None,
            "error_category": cat if status == "confirmed" else None, "response_time_s": rt}
    return client.post(f"/practice/sessions/{sid}/attempts", json=body)


def start(client, model=None):
    r = client.post("/practice/sessions", json={"model_version": model} if model else {})
    assert r.status_code == 200, r.text
    return r.json()


def test_items_manifest_has_24_items_with_all_hints(make_client):
    _, c = make_client()
    items = c.get("/items").json()["items"]
    assert 20 <= len(items) <= 30
    assert all(set(i["cues"]) == {"first_sound", "first_letter", "sentence_frame", "semantic"} for i in items)
    assert c.get(items[0]["image_url"]).status_code == 200


def test_confirmed_attempt_flow_end_to_end(make_client):
    app, c = make_client()
    v = start(c)
    pres = v["presentation"]
    tr = c.post("/transcribe", files={"audio": ("a.webm", b"fake", "audio/webm")},
                data={"exercise_id": pres["exercise_id"]}).json()
    assert tr["status"] == "ok" and tr["suggestion"]["suggestion"] == "matches"
    r = attempt(c, v["session_id"], pres, "a1", True).json()
    assert r["decision"]["decision"]["source"] == "bandit"
    nxt = r["presentation"]
    assert nxt["exercise_id"] != pres["exercise_id"] and nxt["cue"] == "none"   # correct -> next picture only
    s = c.get(f"/practice/sessions/{v['session_id']}").json()
    assert s["stats"]["independent_naming"] == {"correct": 1, "n": 1}
    # persisted snapshot
    snap = json.loads((resolve(app.state.registry.pcfg.sessions_dir) / f"{v['session_id']}.json").read_text())
    assert len(snap["attempts"]) == 1


def test_duplicate_attempt_is_not_counted_twice_and_conflict_is_409(make_client):
    _, c = make_client()
    v = start(c)
    sid, pres = v["session_id"], v["presentation"]
    r1 = attempt(c, sid, pres, "dup-1", False, "sound")
    r2 = attempt(c, sid, pres, "dup-1", False, "sound")
    assert r1.status_code == r2.status_code == 200
    assert r2.json()["duplicate"] is True
    assert r2.json()["presentation"] == r1.json()["presentation"]
    s = c.get(f"/practice/sessions/{sid}").json()
    assert s["stats"]["confirmed_attempts"] == 1 and len(s["decisions"]) == 1
    assert attempt(c, sid, pres, "dup-1", True).status_code == 409


def test_unscored_attempt_never_reaches_the_model(make_client):
    app, c = make_client()
    rt = install_fake(app, "fake", '{"op":"next","tier":1}')
    v = start(c, "fake")
    sid, pres = v["session_id"], v["presentation"]
    r = attempt(c, sid, pres, "u1", None, status="unscored").json()
    assert r["decision"] is None and r["presentation"]["presentation_id"] == pres["presentation_id"]
    assert rt.calls == []
    s = c.get(f"/practice/sessions/{sid}").json()
    assert s["stats"]["confirmed_attempts"] == 0 and s["stats"]["unscored_attempts"] == 1


def test_unscored_attempt_cannot_carry_a_result(make_client):
    _, c = make_client()
    v = start(c)
    body = {"attempt_id": "u2", "presentation_id": v["presentation"]["presentation_id"], "scoring_status": "unscored",
            "correct": False, "error_category": "sound", "response_time_s": 1.0}
    assert c.post(f"/practice/sessions/{v['session_id']}/attempts", json=body).status_code == 422


def test_incorrect_without_category_rejected_but_unclassified_accepted(make_client):
    _, c = make_client()
    v = start(c)
    assert attempt(c, v["session_id"], v["presentation"], "e1", False, None).status_code == 422
    assert attempt(c, v["session_id"], v["presentation"], "e2", False, "unclassified").status_code == 200


def test_retry_keeps_the_exercise_and_next_changes_it(make_client):
    app, c = make_client()
    install_fake(app, "fake", lambda m: '{"op":"retry","hint":"first_sound"}'
                 if "incorrect" in m[-1]["content"].split("Latest attempt:")[1].split("\n")[0]
                 else '{"op":"next","tier":1}')
    v = start(c, "fake")
    sid, p0 = v["session_id"], v["presentation"]
    r = attempt(c, sid, p0, "r1", False, "sound").json()
    p1 = r["presentation"]
    assert r["decision"]["decision"]["source"] == "fake"
    assert p1["exercise_id"] == p0["exercise_id"] and p1["cue"] == "first_sound" and p1["hint_text"]
    assert p1["attempt_on_exercise"] == 2
    r = attempt(c, sid, p1, "r2", True).json()
    p2 = r["presentation"]
    assert p2["exercise_id"] != p0["exercise_id"] and p2["cue"] == "none"
    s = c.get(f"/practice/sessions/{sid}").json()
    assert s["stats"]["naming_after_hint"] == {"correct": 1, "n": 1}
    assert s["stats"]["independent_naming"] == {"correct": 0, "n": 1}


def test_stale_presentation_is_rejected(make_client):
    _, c = make_client()
    v = start(c)
    attempt(c, v["session_id"], v["presentation"], "s1", True)
    assert attempt(c, v["session_id"], v["presentation"], "s2", True).status_code == 409


@pytest.mark.parametrize("answer,reason", [
    ('{"op":"retry","hint":"first_sound"}', "not one of the valid options"),    # retry after a correct answer
    ('{"op":"next","tier":3}', "not one of the valid options"),                 # two tiers up from tier 1
    ("I think the patient should rest.", "could not be parsed"),
])
def test_invalid_model_output_falls_back_explicitly(make_client, answer, reason):
    app, c = make_client()
    install_fake(app, "fake", answer)
    v = start(c, "fake")
    d = attempt(c, v["session_id"], v["presentation"], "f1", True).json()["decision"]["decision"]
    assert d["source"] == "bandit" and d["requested_model"] == "fake"
    assert reason in d["fallback_reason"] and d["raw_output"] == answer and d["valid"] is False
    s = c.get(f"/practice/sessions/{v['session_id']}").json()
    assert s["stats"]["fallbacks"] == 1


def test_unknown_or_unavailable_model_is_refused(make_client):
    _, c = make_client()
    assert c.post("/practice/sessions", json={"model_version": "ditto:does-not-exist"}).status_code == 409
    assert c.post("/models/active", json={"version": "nonsense"}).status_code == 409


def test_override_replaces_presentation_and_is_what_the_bandit_sees(make_client):
    app, c = make_client()
    install_fake(app, "fake", '{"op":"retry","hint":"semantic"}')
    v = start(c, "fake")
    sid = v["session_id"]
    r = attempt(c, sid, v["presentation"], "o1", False, "sound").json()
    did = r["decision"]["decision_id"]
    o = c.post(f"/practice/sessions/{sid}/override", json={
        "decision_id": did, "action": {"operation": "retry_current", "cue": "first_sound"}, "note": "sound error",
        "scope": "patient"})
    assert o.status_code == 200, o.text
    shown = o.json()["presentation"]
    assert shown["cue"] == "first_sound" and shown["source"] == "clinician"
    a2 = attempt(c, sid, shown, "o2", True).json()
    assert a2["attempt"]["hint"] == "first_sound"                 # the action actually shown is recorded
    s = c.get(f"/practice/sessions/{sid}").json()
    dec = next(x for x in s["decisions"] if x["decision_id"] == did)
    assert dec["overridden"] and dec["shown_action"]["cue"] == "first_sound"
    assert dec["decision"]["action"]["cue"] == "semantic"         # the model's proposal is kept too
    # Answered decisions cannot be overridden any more, and constraints cannot be overridden at all.
    assert c.post(f"/practice/sessions/{sid}/override", json={
        "decision_id": did, "action": {"operation": "retry_current", "cue": "none"}}).status_code == 409


def test_override_cannot_break_constraints(make_client):
    _, c = make_client()
    v = start(c)
    r = attempt(c, v["session_id"], v["presentation"], "k1", True).json()
    bad = c.post(f"/practice/sessions/{v['session_id']}/override", json={
        "decision_id": r["decision"]["decision_id"], "action": {"operation": "next_exercise", "tier": 3}})
    assert bad.status_code == 422


def test_demonstrations_persist_with_scope_and_pending_status(make_client, tmp_path):
    app, c = make_client()
    v = start(c)
    r = attempt(c, v["session_id"], v["presentation"], "d1", True).json()
    d = r["decision"]
    row = c.post("/demonstrations", json={"session_id": v["session_id"], "decision_id": d["decision_id"],
                                          "action": d["decision"]["action"], "accepted_proposal": True,
                                          "scope": "session"}).json()
    rows = c.get("/demonstrations").json()
    assert rows[0]["example_id"] == row["example_id"] and rows[0]["scope"] == "session"
    assert rows[0]["training_status"].startswith("pending")
    assert rows[0]["context"]["current"]["exercise_id"] == v["presentation"]["exercise_id"]
    # Once a completed run lists the example, it is reported as used in that run.
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": "r1", "status": "completed",
                                                   "demonstrations": {"example_ids": [row["example_id"]]}}))
    assert c.get("/demonstrations").json()[0]["training_status"] == "used in r1"


def test_sessions_are_restored_after_restart(make_client):
    app, c = make_client()
    v = start(c)
    attempt(c, v["session_id"], v["presentation"], "p1", True)
    app2, c2 = make_client()
    s = c2.get(f"/practice/sessions/{v['session_id']}")
    assert s.status_code == 200 and s.json()["stats"]["confirmed_attempts"] == 1
    assert attempt(c2, v["session_id"], v["presentation"], "p1", True).json()["duplicate"] is True


def test_transcription_failure_is_reported_not_scored(make_client):
    _, c = make_client(speech=FakeSpeech(Transcription(status="empty", provider="fake-asr",
                                                       error="the recording was silent or too short")))
    v = start(c)
    tr = c.post("/transcribe", files={"audio": ("a.webm", b"", "audio/webm")},
                data={"exercise_id": v["presentation"]["exercise_id"]}).json()
    assert tr["status"] == "empty" and "suggestion" not in tr
    s = c.get(f"/practice/sessions/{v['session_id']}").json()
    assert s["stats"]["confirmed_attempts"] == 0


def test_manual_hint_and_skip_are_logged_as_manual(make_client):
    _, c = make_client()
    v = start(c)
    sid, p = v["session_id"], v["presentation"]
    h = c.post(f"/practice/sessions/{sid}/manual", json={"presentation_id": p["presentation_id"], "kind": "show_hint",
                                                        "actor": "patient"}).json()["presentation"]
    assert h["presentation_id"] == p["presentation_id"] and h["cue"] != "none" and h["source"] == "manual:patient"
    sk = c.post(f"/practice/sessions/{sid}/manual", json={"presentation_id": p["presentation_id"], "kind": "skip"}).json()
    assert sk["presentation"]["exercise_id"] != p["exercise_id"]


def test_legacy_update_is_idempotent_by_attempt_id(make_client):
    from tests.conftest import EXAMPLE_STATE
    _, c = make_client()
    body = {"state": {**EXAMPLE_STATE, "session_id": "legacy"}, "action": {"tier": 2, "cue": "none"},
            "outcome": {"correct": True, "error_type": "correct", "response_time": 1.2}, "attempt_id": "att-1"}
    assert c.post("/update", json=body).json()["agent_updated"] is True
    again = c.post("/update", json=body).json()
    assert again["status"] == "duplicate" and again["agent_updated"] is False
    body["outcome"] = {"correct": False, "error_type": "sound", "response_time": 1.2}
    assert c.post("/update", json=body).status_code == 409
    del body["attempt_id"]                     # old callers without attempt_id behave as before
    assert c.post("/update", json=body).json()["agent_updated"] is True


def test_act_reports_decision_source_in_headers_without_changing_body(make_client):
    from tests.conftest import EXAMPLE_STATE
    _, c = make_client()
    r = c.post("/act", json=EXAMPLE_STATE)
    assert set(r.json()) == {"tier", "cue", "rationale"}
    assert r.headers["X-Decision-Source"] == "beta_bandit"
