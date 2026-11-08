"""Practice-flow API: sessions, transcription, confirmed attempts, decisions, clinician feedback and
model versions. Mounted on the agent service (service/app.py).

Flow per picture:
  1. The app shows the current presentation (picture + hint, if any).
  2. Audio -> POST /transcribe -> a *proposed* transcript and a suggested match.
  3. The helper confirms or corrects -> POST /practice/sessions/{id}/attempts with scoring_status:
       confirmed  the result is known: the bandit observes it, the session's model decides what
                  happens next (retry with a hint, or the next picture), and the next presentation
                  is resolved deterministically from the manifest;
       unscored   recording failed / the helper could not tell: stored for the log, never scored,
                  never sent to a model; the same presentation stays on screen.
  4. A clinician may replace a decision whose presentation has not been attempted yet; the
     replacement is what the patient sees and what the bandit later learns from.

Attempts are idempotent by attempt_id: the same request again returns the stored result; different
content under an existing id is a 409. One service worker; state is in memory and snapshotted to
runs/practice/<session_id>.json after every change (restored at startup).
"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator

from agents.llm_policy import BanditDecisionPolicy, Decision
from agents.registry import ModelRegistry
from core.config import BetaBanditConfig, ClinicalConfig, resolve
from core.practice import (AttemptSummary, DecisionContext, ErrorCategory, PracticeAction, PracticeConfig,
                           describe, format_action, load_items, to_state, valid_options)
from core.schema import Cue, CurrentExercise, Outcome
from speech.base import Transcription, UnavailableProvider
from speech.normalize import suggest_match


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------------------------
# Request bodies

class StartSession(BaseModel):
    model_version: str | None = None       # default: the active model
    patient_label: str | None = Field(default=None, max_length=64)


class AttemptIn(BaseModel):
    attempt_id: str = Field(min_length=1, max_length=128)
    presentation_id: str
    raw_transcript: str | None = Field(default=None, max_length=500)       # recogniser proposal
    transcript_status: str | None = None                                   # ok / empty / failed / timeout / manual
    transcript_provider: str | None = None
    corrected_transcript: str | None = Field(default=None, max_length=500)  # after the helper's correction
    scoring_status: Literal["confirmed", "unscored"]
    correct: bool | None = None
    error_category: ErrorCategory | None = None
    unscored_reason: str | None = Field(default=None, max_length=200)
    response_time_s: float = Field(ge=0, le=600)

    @model_validator(mode="after")
    def _check(self) -> "AttemptIn":
        if self.scoring_status == "confirmed":
            if self.correct is None:
                raise ValueError("a confirmed attempt needs correct = true/false")
            if self.correct and self.error_category is not None:
                raise ValueError("a correct attempt has no error category")
            if not self.correct and self.error_category is None:
                raise ValueError("an incorrect attempt needs an error category ('unclassified' if unknown)")
        elif self.correct is not None or self.error_category is not None:
            raise ValueError("an unscored attempt cannot carry a result")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json(exclude={"attempt_id"}).encode()).hexdigest()


class ActionIn(BaseModel):
    operation: Literal["retry_current", "next_exercise"]
    cue: Cue = "none"
    tier: int | None = Field(default=None, ge=1, le=3)

    def to_action(self) -> PracticeAction:
        return PracticeAction(operation=self.operation, cue=self.cue, tier=self.tier)


class OverrideIn(BaseModel):
    decision_id: str
    action: ActionIn
    note: str | None = Field(default=None, max_length=1000)
    scope: Literal["session", "patient", "global"] = "session"
    save_example: bool = True
    demonstrator_id: str = Field(default="clinician-demo", max_length=64)


class DemonstrationIn(BaseModel):
    session_id: str
    decision_id: str
    action: ActionIn                 # the preferred action (= the proposal when accepting it)
    accepted_proposal: bool = False
    note: str | None = Field(default=None, max_length=1000)
    scope: Literal["session", "patient", "global"] = "session"
    demonstrator_id: str = Field(default="clinician-demo", max_length=64)


class ManualIn(BaseModel):
    presentation_id: str
    kind: Literal["show_hint", "skip"]
    cue: Cue | None = None
    actor: Literal["patient", "helper"] = "helper"


class VersionIn(BaseModel):
    version: str


# --------------------------------------------------------------------------------------------
# Session state

class Session:
    def __init__(self, sid: str, model_version: str, patient_label: str | None, bandit: BanditDecisionPolicy) -> None:
        self.id, self.model_version, self.patient_label = sid, model_version, patient_label
        self.created = now()
        self.bandit = bandit
        self.presentations: list[dict] = []
        self.attempts: dict[str, dict] = {}         # attempt_id -> record (insertion-ordered)
        self.responses: dict[str, dict] = {}        # attempt_id -> stored response (idempotency)
        self.decisions: dict[str, dict] = {}
        self.history: list[AttemptSummary] = []     # confirmed attempts only
        self.events: list[dict] = []
        self.paused = False
        self.started_ts = datetime.now(timezone.utc)

    @property
    def current(self) -> dict:
        return self.presentations[-1]

    def minutes(self) -> float:
        return (datetime.now(timezone.utc) - self.started_ts).total_seconds() / 60

    def shown_ids(self) -> list[str]:
        return [p["exercise_id"] for p in self.presentations if p["attempt_on_exercise"] == 1]

    def current_exercise(self) -> CurrentExercise:
        p = self.current
        on_it = [a for a in self.attempts.values()
                 if a["exercise_id"] == p["exercise_id"] and a["exercise_visit"] == p["exercise_visit"]
                 and a["scoring_status"] == "confirmed"]
        hints = [x["cue"] for x in self.presentations
                 if x["exercise_id"] == p["exercise_id"] and x["exercise_visit"] == p["exercise_visit"]
                 and x["cue"] != "none"]
        return CurrentExercise(exercise_id=p["exercise_id"], word=p["word"], tier=p["tier"],
                               hints_shown=list(dict.fromkeys(hints)), attempts=len(on_it))

    def stats(self) -> dict:
        conf = [a for a in self.attempts.values() if a["scoring_status"] == "confirmed"]
        indep = [a for a in conf if a["hint"] == "none" and a["attempt_on_exercise"] == 1]
        hinted = [a for a in conf if a["hint"] != "none"]
        return {"confirmed_attempts": len(conf),
                "unscored_attempts": sum(a["scoring_status"] == "unscored" for a in self.attempts.values()),
                "independent_naming": {"correct": sum(a["correct"] for a in indep), "n": len(indep)},
                "naming_after_hint": {"correct": sum(a["correct"] for a in hinted), "n": len(hinted)},
                "fallbacks": sum(d["decision"]["fallback_reason"] is not None for d in self.decisions.values()),
                "decisions": len(self.decisions)}

    def snapshot(self) -> dict:
        return {"session_id": self.id, "model_version": self.model_version, "patient_label": self.patient_label,
                "created": self.created, "started_ts": self.started_ts.isoformat(), "paused": self.paused,
                "presentations": self.presentations, "attempts": list(self.attempts.values()),
                "responses": self.responses, "decisions": list(self.decisions.values()),
                "history": [h.model_dump() for h in self.history], "events": self.events}


# --------------------------------------------------------------------------------------------

def practice_router(pcfg: PracticeConfig, clinical: ClinicalConfig, bcfg: BetaBanditConfig,
                    registry: ModelRegistry, speech_provider=None) -> APIRouter:
    r = APIRouter()
    bank = load_items(pcfg.items_path)
    sessions: dict[str, Session] = {}
    lock = threading.RLock()
    sessions_dir = resolve(pcfg.sessions_dir)
    demos_path = resolve(pcfg.demonstrations_path)
    state = {"active": pcfg.default_model}
    speech = speech_provider

    def get_speech():
        nonlocal speech
        if speech is None:
            try:
                from speech.whisper_local import WhisperLocalProvider
                speech = WhisperLocalProvider()
            except ImportError as e:
                speech = UnavailableProvider(f"speech extra not installed ({e.name}); run `uv sync --extra speech`. "
                                             "Type the answer instead (labelled as manual entry).")
        return speech

    # ---- persistence
    def save(s: Session) -> None:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        tmp = sessions_dir / f"{s.id}.json.tmp"
        tmp.write_text(json.dumps(s.snapshot()))
        tmp.replace(sessions_dir / f"{s.id}.json")

    def new_bandit(sid: str) -> BanditDecisionPolicy:
        return BanditDecisionPolicy(bcfg, clinical, seed=stable_seed(sid))

    def restore() -> None:
        if not sessions_dir.exists():
            return
        for p in sessions_dir.glob("*.json"):
            try:
                d = json.loads(p.read_text())
                if not (isinstance(d, dict) and "session_id" in d and "presentations" in d):
                    continue          # not a session snapshot (for example active_model.json)
                s = Session(d["session_id"], d["model_version"], d.get("patient_label"), new_bandit(d["session_id"]))
                s.created, s.paused = d["created"], d.get("paused", False)
                s.started_ts = datetime.fromisoformat(d["started_ts"])
                s.presentations, s.responses, s.events = d["presentations"], d["responses"], d["events"]
                s.attempts = {a["attempt_id"]: a for a in d["attempts"]}
                s.decisions = {x["decision_id"]: x for x in d["decisions"]}
                s.history = [AttemptSummary(**h) for h in d["history"]]
                for a in s.attempts.values():     # the bandit's state is rebuilt from confirmed attempts
                    if a["scoring_status"] == "confirmed":
                        s.bandit.observe(to_state(DecisionContext(**a["context"]), s.id, 0), a["tier"], a["hint"],
                                         Outcome(correct=a["correct"],
                                                 error_type="correct" if a["correct"] else a["error_category"],
                                                 response_time=a["response_time_s"]))
                sessions[s.id] = s
            except Exception as e:   # a corrupt snapshot must not take the service down
                print(f"could not restore {p.name}: {e}")

    restore()

    def session(sid: str) -> Session:
        if sid not in sessions:
            raise HTTPException(404, f"no practice session {sid!r}")
        return sessions[sid]

    def event(s: Session, kind: str, **data) -> None:
        s.events.append({"t": now(), "kind": kind, **data})

    def present(s: Session, exercise_id: str, cue: str, attempt_on_exercise: int, visit: int, source: str,
                decision_id: str | None) -> dict:
        item = bank.by_id[exercise_id]
        p = {"presentation_id": f"pr-{uuid.uuid4().hex[:10]}", "exercise_id": exercise_id,
             "word": item.preferred_name, "tier": item.tier, "image_url": f"/app/{item.image_path}",
             "cue": cue, "hint_text": bank.hint_text(exercise_id, cue), "attempt_on_exercise": attempt_on_exercise,
             "exercise_visit": visit, "source": source, "decision_id": decision_id, "created": now()}
        s.presentations.append(p)
        event(s, "presentation", presentation_id=p["presentation_id"], exercise_id=exercise_id, cue=cue, source=source)
        return p

    def visits(s: Session, exercise_id: str) -> int:
        return len({p["exercise_visit"] for p in s.presentations if p["exercise_id"] == exercise_id})

    def apply(s: Session, action: PracticeAction, source: str, decision_id: str | None) -> dict:
        cur = s.current
        if action.operation == "retry_current":
            return present(s, cur["exercise_id"], action.cue, s.current_exercise().attempts + 1,
                           cur["exercise_visit"], source, decision_id)
        item = bank.next_item(action.tier, s.shown_ids())
        return present(s, item.exercise_id, "none", 1, visits(s, item.exercise_id) + 1, source, decision_id)

    def options_for(ctx: DecisionContext) -> list[PracticeAction]:
        return valid_options(ctx, clinical.approved_cues, clinical.max_tier_step, pcfg.max_attempts_per_exercise)

    def decide(s: Session, ctx: DecisionContext, options: list[PracticeAction]) -> Decision:
        n = len(s.history)
        if s.model_version == "bandit":
            return s.bandit.decide(ctx, options, s.id, n)
        try:
            pol = registry.policy(s.model_version)
        except Exception as e:
            return s.bandit.decide(ctx, options, s.id, n, requested=s.model_version,
                                   reason=f"model unavailable: {type(e).__name__}: {str(e)[:200]}")
        return pol.decide(ctx, options, s.bandit, s.id, n)

    def view(s: Session) -> dict:
        cur = s.current
        cur_dec = s.decisions.get(cur["decision_id"]) if cur["decision_id"] else None
        return {"session_id": s.id, "model_version": s.model_version, "patient_label": s.patient_label,
                "paused": s.paused, "created": s.created, "presentation": cur,
                "current_decision": cur_dec, "stats": s.stats()}

    # ---- items, speech, models
    @r.get("/items")
    def items() -> dict:
        return {"version": bank.version, "items": [i.model_dump() | {"image_url": f"/app/{i.image_path}"}
                                                   for i in bank.items]}

    @r.post("/transcribe")
    async def transcribe(audio: UploadFile = File(...), exercise_id: str | None = Form(None)) -> dict:
        data = await audio.read()
        if len(data) > 10_000_000:
            raise HTTPException(413, "recording too large (max 10 MB)")
        res: Transcription = get_speech().transcribe(data, audio.content_type)
        out = res.model_dump()
        if exercise_id and exercise_id in bank.by_id and res.status == "ok":
            it = bank.by_id[exercise_id]
            out["suggestion"] = suggest_match(res.text, it.preferred_name, it.accepted_alternatives).model_dump()
        return out

    @r.get("/speech/status")
    def speech_status() -> dict:
        sp = get_speech()
        return {"provider": sp.name, "available": sp.name != "unavailable", "reason": getattr(sp, "reason", None)}

    @r.get("/models")
    def models() -> dict:
        return {"active": state["active"], "versions": registry.versions()}

    @r.post("/models/active")
    def activate(body: VersionIn) -> dict:
        check_version(body.version)
        state["active"] = body.version
        return {"active": state["active"]}

    def check_version(v: str) -> None:
        if v == "bandit":
            return
        try:
            registry.policy(v)
        except (KeyError, ValueError, FileNotFoundError, RuntimeError) as e:
            raise HTTPException(409, f"cannot use {v}: {e}") from e

    @r.get("/runs")
    def runs() -> list[dict]:
        out = []
        for m in registry.runs():
            out.append({k: m.get(k) for k in ("run_id", "status", "started", "finished", "base_model", "base_revision",
                                             "smoke", "seconds", "final_adapter")}
                       | {"n_demonstrations": m.get("demonstrations", {}).get("n")})
        return out

    @r.get("/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        try:
            m = registry.run(run_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        m.pop("traceback", None)
        evals = {}
        for split in ("test", "dev"):
            p = resolve(pcfg.eval_dir) / f"{run_id}-{split}.json"
            if p.exists():
                evals[split] = json.loads(p.read_text())
        return {"manifest": m, "evaluations": evals}

    # ---- sessions
    @r.post("/practice/sessions")
    def start(body: StartSession) -> dict:
        version = body.model_version or state["active"]
        check_version(version)
        with lock:
            sid = f"ps-{datetime.now().strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
            s = Session(sid, version, body.patient_label, new_bandit(sid))
            event(s, "session_started", model_version=version)
            first = bank.next_item(pcfg.start_tier, [])
            present(s, first.exercise_id, "none", 1, 1, "start", None)
            sessions[sid] = s
            save(s)
            return view(s)

    @r.get("/practice/sessions")
    def list_sessions() -> list[dict]:
        return [{"session_id": s.id, "created": s.created, "model_version": s.model_version,
                 "patient_label": s.patient_label, **s.stats()} for s in sessions.values()][::-1]

    @r.get("/practice/sessions/{sid}")
    def get_session(sid: str) -> dict:
        s = session(sid)
        return view(s) | {"presentations": s.presentations, "attempts": list(s.attempts.values()),
                          "decisions": list(s.decisions.values()), "events": s.events[-50:]}

    @r.post("/practice/sessions/{sid}/model")
    def set_session_model(sid: str, body: VersionIn) -> dict:
        check_version(body.version)
        with lock:
            s = session(sid)
            event(s, "model_switched", old=s.model_version, new=body.version)
            s.model_version = body.version
            save(s)
            return view(s)

    @r.post("/practice/sessions/{sid}/pause")
    def pause(sid: str, paused: bool = True) -> dict:
        with lock:
            s = session(sid)
            s.paused = paused
            event(s, "paused" if paused else "resumed")
            save(s)
            return view(s)

    @r.post("/practice/sessions/{sid}/attempts")
    def attempt(sid: str, body: AttemptIn) -> dict:
        with lock:
            s = session(sid)
            fp = body.fingerprint()
            if body.attempt_id in s.attempts:                 # idempotency by attempt id
                if s.attempts[body.attempt_id]["fingerprint"] != fp:
                    raise HTTPException(409, f"attempt {body.attempt_id} already exists with different content")
                return s.responses[body.attempt_id] | {"duplicate": True}
            if s.paused:
                raise HTTPException(409, "session is paused")
            cur = s.current
            if body.presentation_id != cur["presentation_id"]:
                raise HTTPException(409, "this attempt is for a picture that is no longer on screen "
                                         f"(current presentation {cur['presentation_id']})")
            rec = {**body.model_dump(), "fingerprint": fp, "created": now(), "exercise_id": cur["exercise_id"],
                   "exercise_visit": cur["exercise_visit"], "word": cur["word"], "tier": cur["tier"], "hint": cur["cue"],
                   "attempt_on_exercise": cur["attempt_on_exercise"], "independent": cur["cue"] == "none"
                   and cur["attempt_on_exercise"] == 1}
            if body.scoring_status == "unscored":
                s.attempts[body.attempt_id] = rec
                event(s, "attempt_unscored", attempt_id=body.attempt_id, reason=body.unscored_reason)
                resp = {"attempt": rec, "decision": None, "presentation": cur,
                        "message": "Not scored. The same picture stays on screen."}
                s.responses[body.attempt_id] = resp
                save(s)
                return resp
            summary = AttemptSummary(exercise_id=cur["exercise_id"], word=cur["word"], tier=cur["tier"],
                                     hint=cur["cue"], correct=body.correct, error=body.error_category,
                                     response_time_s=body.response_time_s)
            s.history.append(summary)
            s.attempts[body.attempt_id] = rec
            ctx = DecisionContext(current=s.current_exercise(), history=s.history, session_minutes=s.minutes())
            rec["context"] = ctx.model_dump()
            outcome = Outcome(correct=body.correct, error_type="correct" if body.correct else body.error_category,
                              response_time=body.response_time_s)
            # The bandit (baseline and fallback) observes what was actually shown: this presentation's hint.
            s.bandit.observe(to_state(ctx, s.id, len(s.history)), cur["tier"], cur["cue"], outcome)
            options = options_for(ctx)
            d = decide(s, ctx, options)
            did = f"de-{uuid.uuid4().hex[:10]}"
            drec = {"decision_id": did, "after_attempt_id": body.attempt_id, "created": now(),
                    "context": ctx.model_dump(), "decision": d.model_dump(),
                    "options": [{"key": format_action(o), "label": describe(o, cur["tier"]),
                                 **o.model_dump()} for o in options],
                    "proposal_label": describe(d.action, cur["tier"]),
                    "shown_action": d.action.model_dump(), "overridden": False}
            s.decisions[did] = drec
            event(s, "decision", decision_id=did, source=d.source, requested=d.requested_model,
                  fallback_reason=d.fallback_reason, action=d.action.key())
            nxt = apply(s, d.action, "decision", did)
            resp = {"attempt": rec, "decision": drec, "presentation": nxt}
            s.responses[body.attempt_id] = resp
            save(s)
            return resp

    @r.post("/practice/sessions/{sid}/manual")
    def manual(sid: str, body: ManualIn) -> dict:
        """Patient/helper buttons: 'show a hint' on the current picture (before or between attempts),
        or 'skip' to the next picture at the same tier. Logged as manual, not as a model decision."""
        with lock:
            s = session(sid)
            cur = s.current
            if body.presentation_id != cur["presentation_id"]:
                raise HTTPException(409, "stale presentation")
            if body.kind == "show_hint":
                cue = body.cue or next((c for c in ("first_sound", "sentence_frame", "semantic", "first_letter")
                                        if c in clinical.approved_cues and c not in s.current_exercise().hints_shown),
                                       None)
                if cue is None or cue == "none" or cue not in clinical.approved_cues:
                    raise HTTPException(409, "no further approved hint for this picture")
                cur.update({"cue": cue, "hint_text": bank.hint_text(cur["exercise_id"], cue),
                            "source": f"manual:{body.actor}"})
                event(s, "manual_hint", actor=body.actor, cue=cue, presentation_id=cur["presentation_id"])
                p = cur
            else:
                item = bank.next_item(cur["tier"], s.shown_ids())
                p = present(s, item.exercise_id, "none", 1, visits(s, item.exercise_id) + 1, f"manual:{body.actor}", None)
                event(s, "manual_skip", actor=body.actor)
            save(s)
            return view(s) | {"presentation": p}

    # ---- clinician feedback
    def store_demo(s: Session, drec: dict, action: PracticeAction, accepted_proposal: bool, note: str | None,
                   scope: str, demonstrator_id: str, kind: str) -> dict:
        row = {"example_id": f"cap-{uuid.uuid4().hex[:10]}", "split": "captured", "family_id": None,
               "context": drec["context"], "options": [o["key"] for o in drec["options"]],
               "preferred": action.key(), "accepted": [action.key()], "rule": None, "note": note, "scope": scope,
               "provenance": {"source": f"clinician view ({kind})", "demonstrator_id": demonstrator_id,
                              "review_status": "captured in app", "session_id": s.id,
                              "decision_id": drec["decision_id"], "model_proposal": drec["decision"]["action"],
                              "model_source": drec["decision"]["source"], "accepted_proposal": accepted_proposal},
               "created": now()}
        demos_path.parent.mkdir(parents=True, exist_ok=True)
        with open(demos_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        event(s, "demonstration_saved", example_id=row["example_id"], scope=scope, feedback=kind)
        return row

    @r.post("/practice/sessions/{sid}/override")
    def override(sid: str, body: OverrideIn) -> dict:
        with lock:
            s = session(sid)
            drec = s.decisions.get(body.decision_id)
            if drec is None:
                raise HTTPException(404, "unknown decision")
            action = body.action.to_action()
            if action.key() not in [o["key"] for o in drec["options"]]:
                raise HTTPException(422, f"{action.key()} is not a valid option here (constraints are not changed "
                                         "by a preference)")
            cur = s.current
            if cur["decision_id"] != body.decision_id:
                raise HTTPException(409, "this decision's picture has already been answered; save it as an "
                                         "example instead")
            s.presentations.pop()                     # replace the not-yet-attempted presentation
            drec["shown_action"] = action.model_dump()
            drec["overridden"] = True
            drec["override_note"] = body.note
            event(s, "clinician_override", decision_id=body.decision_id, action=action.key())
            p = apply(s, action, "clinician", body.decision_id)
            demo = store_demo(s, drec, action, False, body.note, body.scope, body.demonstrator_id, "override") \
                if body.save_example else None
            save(s)
            return view(s) | {"presentation": p, "demonstration": demo}

    @r.post("/demonstrations")
    def save_demonstration(body: DemonstrationIn) -> dict:
        with lock:
            s = session(body.session_id)
            drec = s.decisions.get(body.decision_id)
            if drec is None:
                raise HTTPException(404, "unknown decision")
            action = body.action.to_action()
            if action.key() not in [o["key"] for o in drec["options"]]:
                raise HTTPException(422, f"{action.key()} is not a valid option here")
            row = store_demo(s, drec, action, body.accepted_proposal, body.note, body.scope, body.demonstrator_id,
                             "accepted proposal" if body.accepted_proposal else "preferred action")
            save(s)
            return row

    @r.get("/demonstrations")
    def list_demonstrations() -> list[dict]:
        if not demos_path.exists():
            return []
        used: dict[str, list[str]] = {}
        for m in registry.runs():
            if m.get("status") == "completed":
                for eid in m.get("demonstrations", {}).get("example_ids", []):
                    used.setdefault(eid, []).append(m["run_id"])
        rows = [json.loads(line) for line in demos_path.read_text().splitlines() if line.strip()]
        for row in rows:
            runs_ = used.get(row["example_id"], [])
            row["training_status"] = (f"used in {', '.join(runs_)}" if runs_
                                      else "pending: not in any trained model yet")
        return rows[::-1]

    return r
