"""Model updates (docs/API_CONTRACT.md §8): training runs per patient, their preview and which model is active.

The engine owns the records (JSON under <root>/training/<run_id>.json); a Trainer does the training:

  Trainer.start(run_id, patient_id, examples_jsonl_path)   begin a run (never in the server process)
  Trainer.status(run_id) -> TrainerStatus                  where it is now
  Trainer.cancel(run_id)                                   stop it (when a prepared run replaces it)

`SubprocessTrainer` is the real one: it writes the patient's examples in the demonstration row format, runs
`python -m training.patient_ditto --run-id <id> --examples <file>` as a subprocess and reads
runs/ditto/<id>/manifest.json. The patient's adapter starts from the shared adapter (models/ditto_llm/
sft200-g2-s0) and trains on this patient's examples only, plus anchors that keep the shared adapter's choices
in other kinds of situations (configs/patient_ditto.yaml; about 2-3 minutes on an M5 laptop).

Status is synced when someone asks (GET /training-runs/{id}). When the trainer says completed, the preview is
computed in the background (stage "preview", status still "running"), and only then the run is "completed" and
its examples read "used in <run_id>".
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel

from core.config import REPO_ROOT
from core.contract import Example, Preview, Progress, TrainingRun, UsePreparedIn
from service.errors import Conflict, Invalid, NotFound

if TYPE_CHECKING:
    from service.engine import Engine


@dataclass
class TrainerStatus:
    status: Literal["queued", "running", "completed", "failed"]
    progress: Progress
    error: str | None = None


class Trainer(Protocol):
    def start(self, run_id: str, patient_id: str, examples_jsonl_path: Path) -> None: ...
    def status(self, run_id: str) -> TrainerStatus: ...
    def cancel(self, run_id: str) -> None: ...


Previewer = Callable[["Engine", "TrainingRec"], Preview]


class TrainingRec(BaseModel):
    run_id: str
    patient_id: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    progress: Progress = Progress(stage="queued")
    started: str | None = None
    finished: str | None = None
    base: str
    example_ids: list[str]
    preview: Preview | None = None
    prepared_run_id: str | None = None
    error: str | None = None

    @property
    def final(self) -> bool:
        return self.status in ("completed", "failed")

    def run(self) -> TrainingRun:
        return TrainingRun(run_id=self.run_id, patient_id=self.patient_id, status=self.status, progress=self.progress,
                           started=self.started, finished=self.finished, base=self.base, example_ids=self.example_ids,
                           preview=self.preview, prepared=self.prepared_run_id is not None,
                           prepared_run_id=self.prepared_run_id, error=self.error)


# ---------------------------------------------------------------- examples -> demonstration rows

def _fmt(a) -> str:
    return json.dumps(a.model_dump(), separators=(",", ":"))      # same text as core.practice.format_action


def example_rows(engine: "Engine", pid: str, examples: list[Example]) -> list[dict]:
    """Training rows in the format of data/demonstrations/*.jsonl: scope global, the decision's context without
    the patient ID, options and preferred as core.practice.format_action strings."""
    rows = []
    for ex in examples:
        _, _, d = engine.steering.find_decision(ex.decision_id)
        rows.append({
            "example_id": f"cue-{pid}-{ex.example_id}", "split": "train", "family_id": f"cueagain-{pid}",
            "context": d.context.model_dump(mode="json", exclude={"patient_id"}),
            "options": [_fmt(o) for o in d.options], "preferred": _fmt(ex.preferred), "accepted": [_fmt(ex.preferred)],
            "rule": ex.source, "note": ex.note or "", "scope": "global",
            "provenance": {"source": "clinician example (CueAgain app)", "patient_id": pid,
                           "example_id": ex.example_id, "decision_id": ex.decision_id, "origin": ex.source,
                           "note_id": ex.note_id, "model_chose": _fmt(ex.model_chose)}})
    return rows


# ---------------------------------------------------------------- the real trainer

class SubprocessTrainer:
    def __init__(self, runs_dir: str | Path, log_dir: str | Path) -> None:
        self.runs_dir, self.log_dir = Path(runs_dir), Path(log_dir)
        self._procs: dict[str, subprocess.Popen] = {}

    def _manifest(self, run_id: str) -> dict | None:
        p = self.runs_dir / run_id / "manifest.json"
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except json.JSONDecodeError:
            return None                          # being rewritten; read it again next time

    def _pidfile(self, run_id: str) -> Path:
        return self.log_dir / f"{run_id}.pid"

    def start(self, run_id: str, patient_id: str, examples_jsonl_path: Path) -> None:
        if (self.runs_dir / run_id).exists():
            raise Conflict(f"runs/ditto/{run_id} already exists")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["uv", "run", "--extra", "llm", "--extra", "speech", "python", "-m", "training.patient_ditto",
               "--run-id", run_id, "--examples", str(examples_jsonl_path)]
        log = open(self.log_dir / f"{run_id}.log", "wb")
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self._procs[run_id] = proc
        self._pidfile(run_id).write_text(str(proc.pid))

    def _alive(self, run_id: str) -> bool:
        proc = self._procs.get(run_id)
        if proc is not None:
            return proc.poll() is None
        try:
            os.kill(int(self._pidfile(run_id).read_text()), 0)
            return True
        except (OSError, ValueError):
            return False

    def _log_tail(self, run_id: str) -> str:
        try:
            lines = (self.log_dir / f"{run_id}.log").read_text(errors="replace").strip().splitlines()
            return " | ".join(lines[-3:])[-400:]
        except OSError:
            return ""

    def status(self, run_id: str) -> TrainerStatus:
        m = self._manifest(run_id)
        alive = self._alive(run_id)
        if m is None:
            if alive or not self._pidfile(run_id).exists():
                return TrainerStatus("queued", Progress(stage="queued"))
            return TrainerStatus("failed", Progress(stage="queued"),
                                 f"the training process ended before it wrote a manifest: {self._log_tail(run_id)}")
        total = int(m.get("config", {}).get("ditto", {}).get("rounds", 3))
        if m["status"] == "completed":
            return TrainerStatus("completed", Progress(stage="ditto", round=total, of=total))
        if m["status"] in ("failed", "cancelled"):
            return TrainerStatus("failed", Progress(stage="ditto"), m.get("error") or f"training {m['status']}")
        if not alive:
            return TrainerStatus("failed", Progress(stage="sft"),
                                 f"the training process ended while the run was still marked running: {self._log_tail(run_id)}")
        if "sft" not in m:
            return TrainerStatus("running", Progress(stage="sft"))
        return TrainerStatus("running", Progress(stage="ditto", round=min(len(m.get("rounds", [])) + 1, total), of=total))

    def cancel(self, run_id: str, reason: str = "cancelled: replaced by a run prepared earlier") -> None:
        """Stop the run's process and mark its manifest "cancelled" (so it is not left as "running").
        Only use-prepared cancels runs today, hence the default reason."""
        proc = self._procs.get(run_id)
        try:
            pid = proc.pid if proc else int(self._pidfile(run_id).read_text())
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ValueError):
            pass
        for _ in range(50):                      # up to 5 s: the process must not rewrite the manifest afterwards
            if not self._alive(run_id):
                break
            time.sleep(0.1)
        p = self.runs_dir / run_id / "manifest.json"
        m = self._manifest(run_id)
        if m is not None and m.get("status") == "running":
            m.update(status="cancelled", error=reason)
            p.write_text(json.dumps(m, indent=1))


# ---------------------------------------------------------------- the manager

def _thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, daemon=True).start()


class TrainingManager:
    def __init__(self, engine: "Engine", trainer: Trainer, previewer: Previewer,
                 spawn: Callable[[Callable[[], None]], None] = _thread) -> None:
        self.engine, self.trainer, self.previewer, self.spawn = engine, trainer, previewer, spawn
        self.recs: dict[str, TrainingRec] = {}
        self._previewing: set[str] = set()
        d = engine.root / "training"
        for p in sorted(d.glob("*.json")) if d.exists() else []:
            try:
                rec = TrainingRec.model_validate_json(p.read_text())
                self.recs[rec.run_id] = rec
            except Exception as e:
                print(f"engine: could not load {p}: {e}")

    def _save(self, rec: TrainingRec) -> None:
        self.engine._write("training", rec.run_id, rec)

    # ---- queries
    def runs(self, pid: str) -> list[TrainingRec]:
        return [r for r in self.recs.values() if r.patient_id == pid]

    def in_flight_examples(self) -> set[tuple[str, str]]:
        return {(r.patient_id, x) for r in self.recs.values() if not r.final for x in r.example_ids}

    def adapter_version(self, version: str) -> str:
        """The registry version behind a model version: a run that took a prepared run's result serves that
        run's adapter."""
        if version.startswith("ditto:"):
            rec = self.recs.get(version[len("ditto:"):])
            if rec is not None and rec.prepared_run_id:
                return f"ditto:{rec.prepared_run_id}"
        return version

    def completed(self, pid: str) -> list[TrainingRec]:
        self.sync_all()
        return [r for r in self.runs(pid) if r.status == "completed"]

    # ---- start
    def start(self, pid: str) -> TrainingRun:
        eng = self.engine
        with eng.lock:
            patient = eng.patient(pid)
            self.sync_all()
            busy = next((r for r in self.recs.values() if not r.final), None)
            if busy is not None:
                raise Conflict(f"a training run is already in progress ({busy.run_id})")
            examples = eng.steering.examples(pid)
            if not any(e.status == "pending" for e in examples):
                raise Invalid("there are no pending examples to train on")
            n = 1
            while f"{pid}-r{n}" in self.recs:
                n += 1
            run_id = f"{pid}-r{n}"
            path = eng.root / "training" / f"{run_id}.examples.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in example_rows(eng, pid, examples)))
            rec = TrainingRec(run_id=run_id, patient_id=pid, base=patient.active_model.version,
                              example_ids=[e.example_id for e in examples], started=_iso(eng))
            try:
                self.trainer.start(run_id, pid, path)
            except Exception as e:
                rec.status, rec.error, rec.finished = "failed", f"could not start training: {type(e).__name__}: {e}", _iso(eng)
            self.recs[run_id] = rec
            self._save(rec)
            eng._log(pid, "clinician", f"Training run {run_id} started on {len(rec.example_ids)} examples.")
            return rec.run()

    # ---- progress
    def sync_all(self) -> None:
        for rec in list(self.recs.values()):
            if not rec.final:
                self._sync(rec)

    def get(self, run_id: str) -> TrainingRun:
        with self.engine.lock:
            rec = self.recs.get(run_id)
            if rec is None:
                raise NotFound(f"no training run {run_id!r}")
            if not rec.final:
                self._sync(rec)
            return rec.run()

    def _sync(self, rec: TrainingRec) -> None:
        with self.engine.lock:
            if rec.final:
                return
            try:
                st = self.trainer.status(rec.run_id)
            except Exception as e:
                st = TrainerStatus("failed", rec.progress, f"could not read the training status: {type(e).__name__}: {e}")
            before = rec.model_dump()
            if st.status == "failed":
                rec.status, rec.error, rec.finished = "failed", st.error or "training failed", _iso(self.engine)
            elif st.status == "completed":
                rec.status, rec.progress = "running", Progress(stage="preview")
                if rec.run_id not in self._previewing:
                    self._previewing.add(rec.run_id)
                    self.spawn(lambda r=rec: self._finish(r))
            else:
                rec.status, rec.progress = st.status, st.progress
            if rec.model_dump() != before:
                self._save(rec)

    def _finish(self, rec: TrainingRec) -> None:
        """Background: compute the preview once, then complete the run and mark its examples used."""
        try:
            preview = self.previewer(self.engine, rec)
        except Exception as e:
            with self.engine.lock:
                rec.status, rec.error, rec.finished = "failed", f"preview failed: {type(e).__name__}: {str(e)[:300]}", _iso(self.engine)
                self._previewing.discard(rec.run_id)
                self._save(rec)
            return
        with self.engine.lock:
            if rec.final:                              # replaced by a prepared run meanwhile
                self._previewing.discard(rec.run_id)
                return
            rec.preview, rec.status, rec.progress, rec.finished = preview, "completed", Progress(stage="done"), _iso(self.engine)
            self.engine.steering.mark_used(rec.patient_id, rec.example_ids, rec.run_id)
            self._previewing.discard(rec.run_id)
            self._save(rec)
            self.engine._log(rec.patient_id, "system", f"Training run {rec.run_id} finished; its adapter is ready to activate.")

    # ---- a run trained earlier (contract §10.5)
    def use_prepared(self, run_id: str, body: UsePreparedIn) -> TrainingRun:
        with self.engine.lock:
            rec = self.recs.get(run_id)
            if rec is None:
                raise NotFound(f"no training run {run_id!r}")
            prep = self.recs.get(body.prepared_run_id)
            if prep is None:
                raise NotFound(f"no training run {body.prepared_run_id!r}")
            self._sync(rec)
            self._sync(prep)
            if rec.final:
                raise Conflict(f"run {run_id} is already {rec.status}")
            if prep.status != "completed" or prep.preview is None or prep.prepared_run_id:
                raise Conflict(f"run {prep.run_id} is not a completed run trained on its own")
            if prep.patient_id != rec.patient_id or sorted(prep.example_ids) != sorted(rec.example_ids):
                raise Conflict(f"run {prep.run_id} was trained on different examples than {run_id}")
            self.trainer.cancel(run_id)
            rec.preview, rec.prepared_run_id = prep.preview, prep.run_id
            rec.status, rec.progress, rec.finished, rec.error = "completed", Progress(stage="done"), _iso(self.engine), None
            self.engine.steering.mark_used(rec.patient_id, rec.example_ids, rec.run_id)
            self._previewing.discard(rec.run_id)
            self._save(rec)
            self.engine._log(rec.patient_id, "clinician", f"Training run {run_id} took the result of the prepared run {prep.run_id}.")
            return rec.run()


def _iso(engine: "Engine") -> str:
    from service.engine import iso
    return iso(engine.clock())
