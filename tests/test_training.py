"""Training runs, preview and activation (contract §8) with a fake trainer and a fake previewer: no subprocess,
no language model. Also the pieces behind the real ones: the rows written for DITTO, the trainer's manifest
reading, the preview's situation generator and its counting."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("cmudict")

from core.contract import (  # noqa: E402
    Agreement, ModelDecision, Preview, Progress, RetryAction, TrainingRun, Unchanged, matches_rule, rule_action,
)
from core.practice import DecisionContext, load_items, parse_action  # noqa: E402
from service.app import create_app  # noqa: E402
from service.engine import rules_decider  # noqa: E402
from service.preview import ModelPreviewer, generate_situations, held_out_situations  # noqa: E402
from service.training import SubprocessTrainer, TrainerStatus  # noqa: E402
from tests.test_patient_api import FakeSpeech  # noqa: E402
from tests.test_steering import NOTE, RULE, FakeDrafter, demo_session  # noqa: E402


class FakeTrainer:
    """Stands in for the subprocess: the test moves each run queued -> running -> completed."""

    def __init__(self):
        self.state, self.started, self.cancelled = {}, [], []

    def start(self, run_id, patient_id, examples_jsonl_path):
        self.started.append((run_id, patient_id, [json.loads(line) for line in examples_jsonl_path.read_text().splitlines()]))
        self.state[run_id] = TrainerStatus("queued", Progress(stage="queued"))

    def status(self, run_id):
        return self.state[run_id]

    def cancel(self, run_id):
        self.cancelled.append(run_id)

    def set(self, run_id, status, **progress):
        self.state[run_id] = TrainerStatus(status, Progress(**progress), "boom" if status == "failed" else None)


PREVIEW = Preview(follows_your_examples=Agreement(new=2, current=0, total=2, label="fake"),
                  follows_your_rules_new_situations=Agreement(new=7, current=1, total=10),
                  unchanged_elsewhere=Unchanged(same=9, total=10))


def make(tmp_path, previewer=None):
    trainer = FakeTrainer()
    c = TestClient(create_app(speech_provider=FakeSpeech(), cueagain_root=tmp_path, cueagain_decider=rules_decider,
                              rule_drafter=FakeDrafter(), cueagain_trainer=trainer,
                              cueagain_previewer=previewer or (lambda engine, rec: PREVIEW)))
    c.app.state.engine.training.spawn = lambda fn: fn()          # run the "background" preview in place
    return c, trainer


def two_examples(c):
    """A correction on d02 and a note example on d03's twin in a second session: two pending examples."""
    sid = demo_session(c)
    d02 = f"{sid}-d02"
    c.post(f"/decisions/{d02}/correction", json={"preferred": {"op": "retry", "hint": "first_sound"}, "note": "n"})
    sid2 = demo_session(c)
    note = c.post("/patients/demo-ana/notes", json={"text": NOTE}).json()
    c.put(f"/notes/{note['note_id']}/rule", json={"rule": RULE.model_dump(mode="json", exclude_none=True)})
    r = c.post(f"/notes/{note['note_id']}/examples", json={"decision_ids": [f"{sid2}-d02"]})
    assert r.status_code == 201
    return sid, sid2


def test_run_lifecycle(tmp_path):
    c, trainer = make(tmp_path)
    assert c.post("/patients/demo-ana/training-runs").status_code == 422             # nothing pending
    assert c.post("/patients/nobody/training-runs").status_code == 404
    two_examples(c)
    assert c.get("/patients/demo-ana").json()["pending_examples"] == 2

    r = c.post("/patients/demo-ana/training-runs")
    assert r.status_code == 202
    run = TrainingRun(**r.json())
    assert (run.run_id, run.status, run.progress.stage, run.example_ids, run.base) == ("demo-ana-r1", "queued", "queued", ["x-0001", "x-0002"], "rules")
    run_id, _, rows = trainer.started[0]
    assert len(rows) == 2 and rows[0]["example_id"] == "cue-demo-ana-x-0001"
    assert c.post("/patients/demo-ana/training-runs").status_code == 409             # one at a time
    assert c.get("/training-runs/nope").status_code == 404

    trainer.set(run_id, "running", stage="sft")
    assert c.get(f"/training-runs/{run_id}").json()["progress"] == {"stage": "sft", "round": None, "of": None}
    trainer.set(run_id, "running", stage="ditto", round=2, of=3)
    assert c.get(f"/training-runs/{run_id}").json()["progress"] == {"stage": "ditto", "round": 2, "of": 3}
    assert all(e["status"] == "pending" for e in c.get("/patients/demo-ana/examples").json())

    trainer.set(run_id, "completed", stage="ditto", round=3, of=3)
    done = TrainingRun(**c.get(f"/training-runs/{run_id}").json())
    assert (done.status, done.progress.stage, done.preview) == ("completed", "done", PREVIEW)
    assert {e["status"] for e in c.get("/patients/demo-ana/examples").json()} == {f"used in {run_id}"}
    assert c.get("/patients/demo-ana").json()["pending_examples"] == 0
    assert c.post("/patients/demo-ana/training-runs").status_code == 422             # nothing new pending

    ids = [m["id"] for m in c.get("/patients/demo-ana/models").json()]
    assert "rules" in ids and f"ditto:{run_id}" in ids
    # a used example can't be changed
    sid = c.app.state.engine.sessions and next(iter(c.app.state.engine.sessions))
    assert c.post(f"/decisions/{sid}-d02/correction", json={"preferred": {"op": "retry", "hint": "first_letter"}}).status_code == 409


def test_preview_runs_in_the_background(tmp_path):
    seen = {}

    def slow_previewer(engine, rec):
        seen["run"] = rec.run_id
        return PREVIEW
    c, trainer = make(tmp_path, slow_previewer)
    queue = []
    c.app.state.engine.training.spawn = queue.append                             # hold the background job
    two_examples(c)
    run_id = c.post("/patients/demo-ana/training-runs").json()["run_id"]
    trainer.set(run_id, "completed", stage="ditto", round=3, of=3)
    r = c.get(f"/training-runs/{run_id}").json()
    assert (r["status"], r["progress"]["stage"], r["preview"]) == ("running", "preview", None)
    c.get(f"/training-runs/{run_id}")
    assert len(queue) == 1                                                       # started once
    queue[0]()
    assert c.get(f"/training-runs/{run_id}").json()["status"] == "completed" and seen["run"] == run_id


def test_failed_runs_and_failed_previews(tmp_path):
    c, trainer = make(tmp_path)
    two_examples(c)
    run_id = c.post("/patients/demo-ana/training-runs").json()["run_id"]
    trainer.set(run_id, "failed", stage="sft")
    r = c.get(f"/training-runs/{run_id}").json()
    assert (r["status"], r["error"]) == ("failed", "boom")
    assert all(e["status"] == "pending" for e in c.get("/patients/demo-ana/examples").json())
    assert [m["id"] for m in c.get("/patients/demo-ana/models").json() if m["kind"] == "patient"] == []
    again = c.post("/patients/demo-ana/training-runs")                           # a failed run doesn't block
    assert again.status_code == 202 and again.json()["run_id"] == "demo-ana-r2"

    def broken(engine, rec):
        raise RuntimeError("no model")
    c2, trainer2 = make(tmp_path / "b", broken)
    two_examples(c2)
    rid = c2.post("/patients/demo-ana/training-runs").json()["run_id"]
    trainer2.set(rid, "completed", stage="ditto", round=3, of=3)
    r = c2.get(f"/training-runs/{rid}").json()
    assert r["status"] == "failed" and "preview failed" in r["error"] and r["preview"] is None


def test_runs_survive_a_restart(tmp_path):
    c, trainer = make(tmp_path)
    two_examples(c)
    run_id = c.post("/patients/demo-ana/training-runs").json()["run_id"]
    trainer.set(run_id, "completed", stage="ditto", round=3, of=3)
    c.get(f"/training-runs/{run_id}")
    c2, _ = make(tmp_path)
    r = c2.get(f"/training-runs/{run_id}").json()
    assert r["status"] == "completed" and r["preview"]["unchanged_elsewhere"]["total"] == 10


def test_use_prepared(tmp_path):
    c, trainer = make(tmp_path)
    two_examples(c)
    prep = c.post("/patients/demo-ana/training-runs").json()["run_id"]
    trainer.set(prep, "completed", stage="ditto", round=3, of=3)
    assert c.get(f"/training-runs/{prep}").json()["status"] == "completed"
    # the same two examples again, as a live run: make them pending again by using a second patient is not
    # possible, so build the live run by hand with the same example ids
    eng = c.app.state.engine
    from service.training import TrainingRec
    live = TrainingRec(run_id="demo-ana-r2", patient_id="demo-ana", base="rules", example_ids=["x-0001", "x-0002"],
                       status="running", progress=Progress(stage="ditto", round=1, of=3), started="2026-11-07T10:00:00Z")
    other = TrainingRec(run_id="demo-ana-r3", patient_id="demo-ana", base="rules", example_ids=["x-0001"],
                        status="running", progress=Progress(stage="sft"), started="2026-11-07T10:00:00Z")
    eng.training.recs.update({live.run_id: live, other.run_id: other})
    trainer.set("demo-ana-r2", "running", stage="ditto", round=1, of=3)
    trainer.set("demo-ana-r3", "running", stage="sft")

    assert c.post("/training-runs/demo-ana-r3/use-prepared", json={"prepared_run_id": prep}).status_code == 409   # other examples
    assert c.post("/training-runs/demo-ana-r2/use-prepared", json={"prepared_run_id": "nope"}).status_code == 404
    assert c.post("/training-runs/demo-ana-r2/use-prepared", json={"prepared_run_id": "demo-ana-r3"}).status_code == 409   # not completed
    r = c.post("/training-runs/demo-ana-r2/use-prepared", json={"prepared_run_id": prep})
    assert r.status_code == 200
    run = TrainingRun(**r.json())
    assert (run.prepared, run.prepared_run_id, run.status, run.preview) == (True, prep, "completed", PREVIEW)
    assert trainer.cancelled == ["demo-ana-r2"]
    assert c.post("/training-runs/demo-ana-r2/use-prepared", json={"prepared_run_id": prep}).status_code == 409   # already completed
    assert c.get("/training-runs/demo-ana-r2").json()["prepared_run_id"] == prep
    # the prepared result is served by the prepared run's adapter
    assert eng.adapter_version("ditto:demo-ana-r2") == f"ditto:{prep}"


def test_models_and_activation(tmp_path):
    c, trainer = make(tmp_path)
    two_examples(c)
    run_id = c.post("/patients/demo-ana/training-runs").json()["run_id"]
    assert c.post("/patients/demo-ana/active-model", json={"version": f"ditto:{run_id}"}).status_code == 422   # not completed
    trainer.set(run_id, "completed", stage="ditto", round=3, of=3)
    models = {m["id"]: m for m in c.get("/patients/demo-ana/models").json()}
    assert models[f"ditto:{run_id}"]["kind"] == "patient" and models["rules"]["active"] and not models[f"ditto:{run_id}"]["active"]

    running = c.post("/patients/demo-ana/sessions").json()                      # started before the switch
    assert c.post("/patients/demo-ana/active-model", json={"version": "nonsense"}).status_code == 422
    assert c.post("/patients/nobody/active-model", json={"version": "rules"}).status_code == 404
    r = c.post("/patients/demo-ana/active-model", json={"version": f"ditto:{run_id}"})
    assert r.status_code == 200
    assert r.json()["active_model"] == {"version": f"ditto:{run_id}", "patient_adapter": run_id}
    assert c.get("/patients/demo-ana").json()["active_model"]["version"] == f"ditto:{run_id}"
    assert {m["id"]: m["active"] for m in c.get("/patients/demo-ana/models").json()}[f"ditto:{run_id}"]
    assert running["model_version"] == "rules" and c.app.state.engine.sessions[running["session_id"]].model_version == "rules"
    nxt = c.post("/patients/demo-ana/sessions").json()
    assert nxt["model_version"] == f"ditto:{run_id}"                            # the next session uses it
    assert any("Active model set" in e["what"] for e in c.get("/patients/demo-ana/log").json())
    c.post("/patients/demo-ana/active-model", json={"version": "rules"})        # undo
    assert c.get("/patients/demo-ana").json()["active_model"]["patient_adapter"] is None


# ---------------------------------------------------------------- the real pieces

def test_example_rows_have_the_demonstration_format(tmp_path):
    c, trainer = make(tmp_path)
    two_examples(c)
    c.post("/patients/demo-ana/training-runs")
    seed = json.loads(open("data/demonstrations/demonstrator_a_v1.jsonl").readline())
    for row in trainer.started[0][2]:
        assert set(row) == set(seed)
        assert (row["split"], row["scope"]) == ("train", "global") and "patient_id" not in row["context"]
        assert row["provenance"]["source"] == "clinician example (CueAgain app)"
        DecisionContext(**row["context"])
        assert parse_action(row["preferred"]).key() == row["preferred"] and row["preferred"] in row["options"]
        assert row["accepted"] == [row["preferred"]]
    assert json.loads(trainer.started[0][2][0]["preferred"]) == {"op": "retry", "hint": "first_sound"}


def test_subprocess_trainer_reads_the_manifest(tmp_path):
    runs, logs = tmp_path / "runs", tmp_path / "logs"
    t = SubprocessTrainer(runs, logs)
    assert t.status("r1").status == "queued"                                   # nothing yet
    alive = SimpleNamespace(poll=lambda: None, pid=1)
    dead = SimpleNamespace(poll=lambda: 1, pid=1)

    def manifest(**kw):
        (runs / "r1").mkdir(parents=True, exist_ok=True)
        (runs / "r1" / "manifest.json").write_text(json.dumps({"run_id": "r1", "config": {"ditto": {"rounds": 3}}, **kw}))
    t._procs["r1"] = alive
    manifest(status="running", rounds=[])
    assert t.status("r1").progress == Progress(stage="sft")
    manifest(status="running", sft={"steps": 3}, rounds=[])
    assert t.status("r1").progress == Progress(stage="ditto", round=1, of=3)
    manifest(status="running", sft={"steps": 3}, rounds=[{}, {}])
    assert t.status("r1").progress == Progress(stage="ditto", round=3, of=3)
    manifest(status="completed", sft={}, rounds=[{}, {}, {}])
    assert t.status("r1").status == "completed"
    manifest(status="failed", error="OutOfMemory", rounds=[])
    assert (t.status("r1").status, t.status("r1").error) == ("failed", "OutOfMemory")
    manifest(status="running", sft={}, rounds=[])
    t._procs["r1"] = dead
    assert t.status("r1").status == "failed" and "ended while" in t.status("r1").error
    (runs / "r1" / "manifest.json").unlink()
    t._pidfile("r1").parent.mkdir(parents=True, exist_ok=True)
    t._pidfile("r1").write_text("1")
    assert t.status("r1").status == "failed" and "before it wrote a manifest" in t.status("r1").error


def test_generated_situations_satisfy_the_rule():
    bank = load_items()
    rng = random.Random(0)
    sits = generate_situations(RULE, bank, 3, {"ex-car"}, 10, rng)
    assert len(sits) == 10
    assert len({(c.current.exercise_id, tuple(c.current.hints_shown), c.latest.error, len(c.history)) for c, _ in sits}) == 10
    for ctx, opts in sits:
        assert matches_rule(RULE, ctx) and rule_action(RULE, ctx) in opts
        assert "semantic" in ctx.current.hints_shown and ctx.current.exercise_id != "ex-car"
        assert all(isinstance(o, RetryAction) for o in opts)
    # a rule that can never hold (needs a hint but allows only the first failed try) gives nothing
    from core.contract import Range, RetryThen, Rule, RuleWhen
    impossible = Rule(when=RuleWhen(hints_tried_include=["semantic"], failed_attempts_on_picture=Range(max=1)),
                      then=RetryThen(op="retry", hint="first_sound"), summary="x")
    assert generate_situations(impossible, bank, 3, set(), 10, random.Random(0)) == []
    # relative level rule after missed pictures
    from core.contract import NextThen
    easier = Rule(when=RuleWhen(missed_pictures_in_a_row=Range(min=2), failed_attempts_on_picture=Range(min=3)),
                  then=NextThen(op="next", level="easier"), summary="x")
    got = generate_situations(easier, bank, 3, set(), 5, random.Random(1))
    assert got and all(matches_rule(easier, c) and rule_action(easier, c) in o for c, o in got)


def test_held_out_situations_use_the_live_option_rule():
    sits = held_out_situations("demo-ana")
    assert len(sits) == 10
    for ctx, opts in sits:
        retries = [o for o in opts if isinstance(o, RetryAction)]
        assert not retries or len(retries) == len(opts)
        assert all(o.hint not in ctx.current.hints_shown for o in retries)


def test_previewer_counts_real_choices(tmp_path):
    """The previewer's numbers are counts of what the two models choose; here the models are scripted."""
    c, _ = make(tmp_path)
    two_examples(c)
    eng = c.app.state.engine

    def decide(ctx, options, s):
        want = RetryAction(op="retry", hint="first_sound")
        a = want if (s.model_version == "ditto:demo-ana-r1" and want in options) else options[-1]
        return ModelDecision(action=a, source="fake", fallback_reason=None, raw_output=None, latency_ms=0)

    from service.training import TrainingRec
    rec = TrainingRec(run_id="demo-ana-r1", patient_id="demo-ana", base="rules", example_ids=["x-0001", "x-0002"])
    p = ModelPreviewer(decide)(eng, rec)
    assert (p.follows_your_examples.new, p.follows_your_examples.current, p.follows_your_examples.total) == (2, 0, 2)
    a = p.follows_your_rules_new_situations
    assert (a.new, a.current, a.total) == (10, 0, 10) and "generated" in a.label
    assert p.unchanged_elsewhere.total == 10 and 0 <= p.unchanged_elsewhere.same <= 10
