"""Clinician steering (contract §7): corrections, notes, rules, matches and examples, through the real endpoints
with a fake recognizer, the rules decider and a fake rule drafter (no network, no models)."""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("cmudict")

from core.contract import Example, Note, Range, RetryThen, Rule, RuleMatch, RuleWhen, SessionReview  # noqa: E402
from service.app import create_app  # noqa: E402
from service.engine import rules_decider  # noqa: E402
from service.rule_drafter import NOT_CONFIGURED, ClaudeRuleDrafter, RuleDraft, to_rule  # noqa: E402
from tests.test_patient_api import FakeSpeech  # noqa: E402

NOTE = "When meaning hints have repeatedly failed, try a first-sound hint before moving to another picture."
RULE = Rule(when=RuleWhen(hints_tried_include=["semantic"], failed_attempts_on_picture=Range(min=2)),
            then=RetryThen(op="retry", hint="first_sound"), summary="After 2+ failed tries with a meaning hint → first sound.")


class FakeDrafter:
    def __init__(self, rule=RULE, problem=None):
        self.rule, self.problem, self.calls = rule, problem, []

    def draft(self, text):
        self.calls.append(text)
        return self.rule, self.problem


def make(tmp_path, drafter=None):
    return TestClient(create_app(speech_provider=FakeSpeech(), cueagain_root=tmp_path, cueagain_decider=rules_decider,
                                 rule_drafter=drafter or FakeDrafter()))


def say(c, sid, pid, text):
    r = c.post(f"/sessions/{sid}/recordings", files={"audio": ("a.webm", text.encode(), "audio/webm")},
               data={"presentation_id": pid, "client_attempt_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    return r.json()


def demo_session(c):
    """Picture 'cat', three wrong words: d01 (1 failed try), d02 = the demo situation (2 failed tries, a meaning
    hint shown; the rules decider picks the sentence hint), d03 = the last try (levels only)."""
    s = c.post("/patients/demo-ana/sessions").json()
    sid, pid = s["session_id"], s["picture"]["presentation_id"]
    for w in ("bone", "truck", "zebra"):
        say(c, sid, pid, w)
    return sid


@pytest.fixture
def client(tmp_path):
    return make(tmp_path)


def test_correction_flow(client):
    sid = demo_session(client)
    d01, d02, d03 = [d for p in SessionReview(**client.get(f"/sessions/{sid}/review").json()).pictures[:1] for d in p.decisions]
    assert (d02.action.hint, [o.model_dump() for o in d02.options][0]["op"]) == ("sentence_frame", "retry")
    body = {"preferred": {"op": "retry", "hint": "first_sound"}, "note": "sound cue first"}
    r = client.post(f"/decisions/{d02.decision_id}/correction", json=body)
    assert r.status_code == 201
    ex = Example(**r.json())
    assert (ex.example_id, ex.source, ex.status, ex.model_chose.hint, ex.preferred.hint) == ("x-0001", "correction", "pending", "sentence_frame", "first_sound")
    rev = SessionReview(**client.get(f"/sessions/{sid}/review").json())
    shown = {d.decision_id: d for d in rev.pictures[0].decisions}
    assert shown[d02.decision_id].correction == ex and shown[d01.decision_id].correction is None
    assert [Example(**e) for e in client.get("/patients/demo-ana/examples").json()] == [ex]
    assert client.get("/patients/demo-ana").json()["pending_examples"] == 1


def test_correction_errors_and_replacement(client):
    sid = demo_session(client)
    d02 = client.get(f"/sessions/{sid}/review").json()["pictures"][0]["decisions"][1]
    did = d02["decision_id"]
    assert client.post("/decisions/nope/correction", json={"preferred": {"op": "retry", "hint": "first_sound"}}).status_code == 404
    bad = client.post(f"/decisions/{did}/correction", json={"preferred": {"op": "retry", "hint": "semantic"}})   # already shown
    assert bad.status_code == 422 and "is not one of this decision's options" in bad.json()["detail"]
    assert client.post(f"/decisions/{did}/correction", json={"preferred": {"op": "next", "tier": 2}}).status_code == 422
    client.post(f"/decisions/{did}/correction", json={"preferred": {"op": "retry", "hint": "first_sound"}})
    r = client.post(f"/decisions/{did}/correction", json={"preferred": {"op": "retry", "hint": "first_letter"}, "note": "again"})
    assert r.status_code == 201 and r.json()["example_id"] == "x-0001"
    exs = client.get("/patients/demo-ana/examples").json()
    assert len(exs) == 1 and exs[0]["preferred"]["hint"] == "first_letter" and exs[0]["note"] == "again"
    assert client.get("/patients/demo-ana").json()["pending_examples"] == 1
    assert client.get("/patients/nobody/examples").status_code == 404


def test_examples_survive_a_restart(tmp_path):
    c = make(tmp_path)
    sid = demo_session(c)
    did = c.get(f"/sessions/{sid}/review").json()["pictures"][0]["decisions"][1]["decision_id"]
    c.post(f"/decisions/{did}/correction", json={"preferred": {"op": "retry", "hint": "first_sound"}})
    note = c.post("/patients/demo-ana/notes", json={"text": NOTE}).json()
    c2 = make(tmp_path)
    assert [e["example_id"] for e in c2.get("/patients/demo-ana/examples").json()] == ["x-0001"]
    assert c2.get("/patients/demo-ana").json()["pending_examples"] == 1
    assert c2.get(f"/sessions/{sid}/review").json()["pictures"][0]["decisions"][1]["correction"]["example_id"] == "x-0001"
    again = c2.post("/patients/demo-ana/notes", json={"text": "second"}).json()
    assert (note["note_id"], again["note_id"]) == ("n-0001", "n-0002")


def test_note_draft_confirm_matches_and_examples(tmp_path):
    drafter = FakeDrafter()
    c = make(tmp_path, drafter)
    sid = demo_session(c)
    r = c.post("/patients/demo-ana/notes", json={"text": NOTE})
    assert r.status_code == 201 and drafter.calls == [NOTE]
    note = Note(**r.json())
    assert (note.rule_source, note.rule_status, note.rule) == ("claude", "draft", RULE)
    assert c.get(f"/notes/{note.note_id}/matches").status_code == 409            # not confirmed yet

    r = c.put(f"/notes/{note.note_id}/rule", json={"rule": RULE.model_dump(mode="json", exclude_none=True)})
    assert r.status_code == 200 and (r.json()["rule_status"], r.json()["rule_source"]) == ("confirmed", "claude")

    ms = [RuleMatch(**m) for m in c.get(f"/notes/{note.note_id}/matches").json()]
    d = {m.decision_id: m for m in ms}
    assert len(ms) == 2                                                      # d01 has only one failed try
    demo, last = d[f"{sid}-d02"], d[f"{sid}-d03"]
    assert demo.allowed and demo.rule_says.hint == "first_sound" and demo.model_chose.hint == "sentence_frame"
    assert not demo.already_agrees and demo.word == "cat" and "2 failed tries" in demo.situation
    assert not last.allowed and last.rule_says is None                       # last try: only levels are offered

    assert c.post(f"/notes/{note.note_id}/examples", json={"decision_ids": [last.decision_id]}).status_code == 422
    assert c.post(f"/notes/{note.note_id}/examples", json={"decision_ids": [f"{sid}-d01"]}).status_code == 422
    r = c.post(f"/notes/{note.note_id}/examples", json={"decision_ids": [demo.decision_id]})
    assert r.status_code == 201
    ex = Example(**r.json()[0])
    assert (ex.source, ex.note_id, ex.preferred.hint, ex.status) == ("note", note.note_id, "first_sound", "pending")
    assert [e["example_id"] for e in c.get("/patients/demo-ana/examples").json()] == [ex.example_id]
    assert c.get("/patients/demo-ana").json()["pending_examples"] == 1


def test_edited_rule_is_the_clinicians(client):
    note = client.post("/patients/demo-ana/notes", json={"text": NOTE}).json()
    edited = RULE.model_copy(update={"then": RetryThen(op="retry", hint="first_letter")})
    r = client.put(f"/notes/{note['note_id']}/rule", json={"rule": edited.model_dump(mode="json", exclude_none=True)})
    assert r.json()["rule_source"] == "clinician" and r.json()["rule_status"] == "confirmed"
    bad = Rule(when=RuleWhen(failed_attempts_on_picture=Range(min=3, max=1)), then=RetryThen(op="retry", hint="semantic"), summary="x")
    assert client.put(f"/notes/{note['note_id']}/rule", json={"rule": bad.model_dump(mode="json", exclude_none=True)}).status_code == 422
    assert client.put("/notes/n-9999/rule", json={"rule": edited.model_dump(mode="json", exclude_none=True)}).status_code == 404


def test_no_draft_means_a_hand_written_rule(tmp_path):
    c = make(tmp_path, FakeDrafter(rule=None, problem="Notes can steer hints and difficulty only."))
    note = c.post("/patients/demo-ana/notes", json={"text": "She seems tired on Mondays."}).json()
    assert (note["rule"], note["rule_source"], note["rule_status"]) == (None, "none", "none")
    assert note["problem"] == "Notes can steer hints and difficulty only."
    r = c.put(f"/notes/{note['note_id']}/rule", json={"rule": RULE.model_dump(mode="json", exclude_none=True)})
    assert (r.json()["rule_source"], r.json()["rule_status"], r.json()["problem"]) == ("clinician", "confirmed", None)


def test_not_configured_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("service.rule_drafter.load_dotenv", lambda *a, **k: None)
    c = TestClient(create_app(speech_provider=FakeSpeech(), cueagain_root=tmp_path, cueagain_decider=rules_decider,
                              rule_drafter=ClaudeRuleDrafter()))
    note = c.post("/patients/demo-ana/notes", json={"text": NOTE}).json()
    assert note["rule"] is None and note["problem"] == NOT_CONFIGURED


def test_matches_use_missed_pictures_from_the_history(tmp_path):
    """missed_pictures_in_a_row is computed from the session's earlier pictures."""
    c = make(tmp_path)
    s = c.post("/patients/demo-ana/sessions").json()
    sid, pid = s["session_id"], s["picture"]["presentation_id"]
    for _ in range(3):
        r = say(c, sid, pid, "zebra")                       # picture 1 missed
    pid = r["next"]["picture"]["presentation_id"]
    say(c, sid, pid, "zebra")                               # picture 2, first failed try
    rule = Rule(when=RuleWhen(missed_pictures_in_a_row=Range(min=1)), then=RetryThen(op="retry", hint="first_sound"), summary="after a miss")
    note = c.post("/patients/demo-ana/notes", json={"text": "x"}).json()
    c.put(f"/notes/{note['note_id']}/rule", json={"rule": rule.model_dump(mode="json", exclude_none=True)})
    ids = [m["decision_id"] for m in c.get(f"/notes/{note['note_id']}/matches").json()]
    assert ids == [f"{sid}-d04"]                            # only the decision on picture 2 follows a missed picture


def test_rule_draft_conversion():
    d = RuleDraft(understood=True, last_error=["meaning"], min_failed_attempts_on_picture=2, hints_tried_include=["semantic"],
                  then_op="retry", then_hint="first_sound", summary="s")
    rule, problem = to_rule(d)
    assert problem is None and rule.then.hint == "first_sound"
    assert rule.when.failed_attempts_on_picture == Range(min=2, max=None) and rule.when.last_error == ["meaning"]
    # a relative level
    rule, problem = to_rule(RuleDraft(understood=True, min_missed_pictures_in_a_row=2, then_op="next", then_level="easier", summary="go easier"))
    assert problem is None and rule.then.level == "easier" and rule.when.missed_pictures_in_a_row.min == 2
    from core.contract import rule_action
    from core.practice import DecisionContext
    from core.schema import CurrentExercise
    from core.practice import AttemptSummary
    ctx = DecisionContext(current=CurrentExercise(exercise_id="ex-x", word="x", tier=3, hints_shown=[], attempts=1),
                          history=[AttemptSummary(exercise_id="ex-x", word="x", tier=3, hint="none", correct=False, error="sound", response_time_s=1)],
                          session_minutes=1)
    assert rule_action(rule, ctx).tier == 2                  # "easier" from level 3
    # failures
    assert to_rule(RuleDraft(understood=False, problem="Notes can steer hints and difficulty only"))[1].startswith("Notes can steer")
    assert to_rule(RuleDraft(understood=True, then_op="retry", summary="s"))[0] is None          # no hint
    assert to_rule(RuleDraft(understood=True, then_op="retry", then_hint="semantic", summary="s"))[0] is None   # no condition
    assert to_rule(RuleDraft(understood=True, level=[4], then_op="retry", then_hint="semantic", summary="s"))[0] is None
    assert to_rule(RuleDraft(understood=True, last_error=["meaning"], then_op="retry", then_hint="semantic", summary=" "))[0] is None


def test_claude_drafter_with_a_fake_client(monkeypatch):
    from types import SimpleNamespace

    class Client:
        def __init__(self, resp):
            self.messages = SimpleNamespace(parse=lambda **kw: resp(kw))

    ok = SimpleNamespace(stop_reason="end_turn", parsed_output=RuleDraft(understood=True, last_error=["meaning"], then_op="retry",
                                                                         then_hint="first_sound", summary="s"))
    seen = {}

    def respond(kw):
        seen.update(kw)
        return ok
    rule, problem = ClaudeRuleDrafter(client=Client(respond)).draft("a note")
    assert problem is None and rule.then.hint == "first_sound"
    assert seen["output_format"] is RuleDraft and seen["messages"] == [{"role": "user", "content": "a note"}]
    monkeypatch.setenv("CUEAGAIN_RULE_MODEL", "some-model")
    ClaudeRuleDrafter(client=Client(respond)).draft("a note")
    assert seen["model"] == "some-model"
    refusal = SimpleNamespace(stop_reason="refusal", parsed_output=None)
    assert ClaudeRuleDrafter(client=Client(lambda kw: refusal)).draft("x")[0] is None

    def boom(kw):
        raise RuntimeError("network down")
    rule, problem = ClaudeRuleDrafter(client=Client(boom)).draft("x")
    assert rule is None and "network down" in problem
