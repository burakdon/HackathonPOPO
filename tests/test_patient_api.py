"""The patient practice API (service/patient_api.py + service/engine.py) against docs/API_CONTRACT.md:
a full session through the real endpoints, with a fake recognizer (the transcript comes from the
uploaded bytes) and the rules decider, so no audio model or language model is needed."""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("cmudict")

from core.contract import AttemptResult, ModelDecision, NextAction, RetryAction, SessionReview  # noqa: E402
from service.app import create_app  # noqa: E402
from service.engine import rules_decider  # noqa: E402
from speech.base import Transcription  # noqa: E402


class FakeSpeech:
    """The 'audio' is the transcript as text; b"" means silence."""
    name = "fake"

    def transcribe(self, audio: bytes, content_type):
        text = audio.decode()
        if not text:
            return Transcription(status="empty", provider=self.name, error="silent")
        return Transcription(status="ok", text=text, provider=self.name, audio_seconds=1.0, no_speech_prob=0.01)


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(speech_provider=FakeSpeech(), cueagain_root=tmp_path, cueagain_decider=rules_decider))


def say(c, sid, pid, text):
    r = c.post(f"/sessions/{sid}/recordings", files={"audio": ("a.webm", text.encode(), "audio/webm")},
               data={"presentation_id": pid, "client_attempt_id": uuid.uuid4().hex})
    assert r.status_code == 200, r.text
    return AttemptResult(**r.json())


def test_demo_patient_and_pictures(client):
    assert [p["id"] for p in client.get("/patients").json()] == ["demo-ana"]
    pics = client.get("/pictures").json()
    assert len(pics) == 24 and pics[0]["image_url"] == "/media/images/cat.svg"
    assert client.get("/media/images/cat.svg").status_code == 200


def test_full_session_through_the_api(client):
    s = client.post("/patients/demo-ana/sessions").json()
    sid, pic = s["session_id"], s["picture"]
    assert pic["word"] == "cat" and pic["attempt_number"] == 1

    r = say(client, sid, pic["presentation_id"], "uuummm cat")               # correct first try
    assert (r.score.outcome, r.next.kind) == ("correct", "correct") and r.next.decision_id
    pic = r.next.picture.model_dump()
    assert pic["word"] == "dog"

    r = say(client, sid, pic["presentation_id"], "bone")                      # wrong word -> hint
    assert (r.score.error_type, r.next.kind, r.next.hint.type) == ("meaning", "hint", "semantic")
    r = say(client, sid, pic["presentation_id"], "it's a dog")                # unclear, not counted
    assert (r.counted, r.next.kind) == (False, "try_again")
    r = say(client, sid, pic["presentation_id"], "dog")
    assert r.next.kind == "correct"
    pic = r.next.picture.model_dump()

    r = say(client, sid, pic["presentation_id"], "")                          # silence twice -> not assessed
    assert r.next.kind == "try_again"
    r = say(client, sid, pic["presentation_id"], "")
    assert r.next.kind == "not_assessed" and r.next.decision_id is None
    pic = r.next.picture.model_dump()

    nr = client.post(f"/sessions/{sid}/no-answer", json={"presentation_id": pic["presentation_id"],
                                                         "client_attempt_id": uuid.uuid4().hex}).json()
    assert nr["score"]["error_type"] == "no_response" and nr["next"]["kind"] == "hint"
    r = say(client, sid, pic["presentation_id"], "zebra")
    assert r.next.kind == "hint"
    r = say(client, sid, pic["presentation_id"], "zebra")                     # third counted try -> show the word
    assert r.next.kind == "show_word" and r.next.word == pic["word"]

    review = SessionReview(**client.get(f"/sessions/{sid}/review").json())
    assert [p.result for p in review.pictures] == ["independent", "assisted", "not_assessed", "unsuccessful"]
    for p in review.pictures:
        shown = [d.action.hint for d in p.decisions if isinstance(d.action, RetryAction)]
        assert len(shown) == len(set(shown)), "a hint was repeated on one picture"
    audio = review.pictures[0].attempts[0].audio_url
    assert client.get(audio).content == b"uuummm cat"
    row = client.get("/patients/demo-ana/sessions").json()[-1]
    assert row["counts"] == {"assessed": 3, "independent": 1, "assisted": 1, "unsuccessful": 1, "not_assessed": 1}

    end = client.post(f"/sessions/{sid}/end").json()
    assert end["ended_early"] is True
    assert client.get(f"/sessions/{sid}").json()["status"] == "ended"


def test_idempotent_and_stale(client):
    s = client.post("/patients/demo-ana/sessions").json()
    sid, pid = s["session_id"], s["picture"]["presentation_id"]
    form = {"presentation_id": pid, "client_attempt_id": "same-id-123"}
    a = client.post(f"/sessions/{sid}/recordings", files={"audio": ("a.webm", b"cat", "audio/webm")}, data=form)
    b = client.post(f"/sessions/{sid}/recordings", files={"audio": ("a.webm", b"cat", "audio/webm")}, data=form)
    assert a.json() == b.json()                                               # retry after a dropped connection
    c = client.post(f"/sessions/{sid}/recordings", files={"audio": ("a.webm", b"dog", "audio/webm")}, data=form)
    assert c.status_code == 409                                               # same id, different answer
    d = client.post(f"/sessions/{sid}/recordings", files={"audio": ("a.webm", b"cat", "audio/webm")},
                    data={"presentation_id": pid, "client_attempt_id": "another-id-1"})
    assert d.status_code == 409 and "no longer the current picture" in d.json()["detail"]
    assert client.post("/sessions/nope/no-answer", json={"presentation_id": "p", "client_attempt_id": "x" * 10}).status_code == 404


def test_speech_unavailable_is_503(tmp_path):
    from speech.base import UnavailableProvider
    c = TestClient(create_app(speech_provider=UnavailableProvider("not installed"), cueagain_root=tmp_path, cueagain_decider=rules_decider))
    s = c.post("/patients/demo-ana/sessions").json()
    r = c.post(f"/sessions/{s['session_id']}/recordings", files={"audio": ("a.webm", b"x", "audio/webm")},
               data={"presentation_id": s["picture"]["presentation_id"], "client_attempt_id": uuid.uuid4().hex})
    assert r.status_code == 503


def test_invalid_model_output_falls_back_to_the_plan(tmp_path):
    def bad(ctx, options, s):
        return ModelDecision(action=NextAction(op="next", tier=3) if ctx.current.tier == 1 else options[0],
                             source="broken", fallback_reason=None, raw_output="x", latency_ms=1)
    c = TestClient(create_app(speech_provider=FakeSpeech(), cueagain_root=tmp_path, cueagain_decider=bad))
    s = c.post("/patients/demo-ana/sessions").json()
    r = say(c, s["session_id"], s["picture"]["presentation_id"], "dog")       # wrong word on "cat"
    assert r.next.kind == "hint" and r.next.hint.type == "semantic"           # plan order: meaning first after a wrong word
    d = c.app.state.engine.sessions[s["session_id"]].pictures[0].decisions[0]   # picture still on screen
    assert d.source == "fallback:plan-order" and "not one of the options" in d.fallback_reason


def test_state_survives_a_restart(tmp_path):
    c = TestClient(create_app(speech_provider=FakeSpeech(), cueagain_root=tmp_path, cueagain_decider=rules_decider))
    s = c.post("/patients/demo-ana/sessions").json()
    say(c, s["session_id"], s["picture"]["presentation_id"], "dog")
    c2 = TestClient(create_app(speech_provider=FakeSpeech(), cueagain_root=tmp_path, cueagain_decider=rules_decider))
    st = c2.get(f"/sessions/{s['session_id']}").json()
    assert st["status"] == "active" and st["picture"]["attempt_number"] == 2 and st["picture"]["hint"]


def test_hint_only_while_tries_are_left(client):
    """After a wrong answer with tries left the model is offered hints only; after the last try, levels only."""
    s = client.post("/patients/demo-ana/sessions").json()
    sid, pid = s["session_id"], s["picture"]["presentation_id"]
    for _ in range(3):
        say(client, sid, pid, "zebra")
    ds = client.app.state.engine.review(sid).pictures[0].decisions
    assert [{o.op for o in d.options} for d in ds] == [{"retry"}, {"retry"}, {"next"}]
    assert ds[1].options and all(o.hint != ds[0].action.hint for o in ds[1].options)   # no repeated hint
