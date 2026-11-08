"""The practice engine behind the patient API (docs/API_CONTRACT.md §3, §4, §6).

Holds patients and sessions, applies the fixed flow rules to each scored attempt, asks the patient's
decision model what comes next, and stores every attempt, recording and decision as JSON under one
directory (default runs/cueagain/). Scoring and transcription happen before `submit` (service/
patient_api.py); the engine receives a contract `Score`, so the same engine also runs scripted
sessions (scripts/build_contract_examples.py).

Flow rules (D1, D6), applied per picture:
  unclear, 1st in a row     -> try_again          (not counted, the model isn't asked)
  unclear, 2nd in a row     -> not_assessed       (same level, the model isn't asked)
  correct                   -> the model picks the next level             -> correct
  incorrect, tries left     -> the model picks which hint                 -> hint
  incorrect, last try       -> the model picks the next level             -> show_word
Hints already shown on a picture are never offered again. While tries are left the model is offered
hints only ("hint only", Option A): it chooses which hint, never whether to move on early. move_on
remains only for a plan with more tries than there are hints. On the session's last picture, a correct
answer ends the session without asking the model.
"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from core.contract import (
    HINTS, Action, ActiveModel, AttemptResult, ContractContext, LogEntry, ModelDecision, NewPatient, NextAction,
    NextStep, Patient, PatientRow, Picture, HintClip, Plan, Presentation, RetryAction, ReviewAttempt, ReviewPicture,
    Score, SessionReview, SessionRow, SessionStarted, SessionState, ShownHint, StoredDecision, picture_result,
    summarize,
)
from core.practice import AttemptSummary, ItemBank, valid_options
from core.schema import CurrentExercise
from service.errors import Conflict, Invalid, NotFound  # noqa: F401  (re-exported)

MESSAGES = {"correct": "Yes! {word}.", "hint": "Here's a hint.", "show_word": "This is a {word}. Say it with me.",
            "try_again": "I didn't catch that. Please try again.", "not_assessed": "Let's try another one.",
            "move_on": "Let's try another one."}

# Fallback hint order per plan (D1): used only when the model's output is invalid or it fails.
HINT_ORDERS: dict[str, dict[str, list[str]]] = {
    "match_error": {"sound": ["first_sound", "first_letter", "sentence_frame", "semantic"],
                    "meaning": ["semantic", "sentence_frame", "first_sound", "first_letter"],
                    "_": ["sentence_frame", "semantic", "first_sound", "first_letter"]},
    "meaning_first": {"_": ["semantic", "sentence_frame", "first_sound", "first_letter"]},
    "first_sound_first": {"_": ["first_sound", "first_letter", "sentence_frame", "semantic"]},
    "least_help": {"_": ["sentence_frame", "first_letter", "first_sound", "semantic"]},
}

DEFAULT_PLAN = Plan(level=1, pictures_per_session=8, max_attempts=3, break_after_misses=3, hint_order="match_error",
                    session_minutes=10, days_per_week=5, personal_words=[])


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def plan_order_action(plan: Plan, ctx: ContractContext, options: list[Action]) -> Action:
    """The fallback: the plan's hint order for the latest error, else the next picture at the same level."""
    retry_hints = [o.hint for o in options if isinstance(o, RetryAction)]
    if retry_hints:
        order = HINT_ORDERS[plan.hint_order]
        err = ctx.latest.error or "_"
        for h in order.get(err, order["_"]):
            if h in retry_hints:
                return RetryAction(op="retry", hint=h)
    nexts = [o for o in options if isinstance(o, NextAction)]
    same = [o for o in nexts if o.tier == ctx.current.tier]
    return (same or nexts)[0]


def decision_options(ctx: ContractContext, hints_shown: list[str], max_attempts: int) -> list[Action]:
    """The actions the model may choose from after a counted attempt (what the live app offers)."""
    allowed = [h for h in HINTS if h not in hints_shown]          # never repeat a hint on a picture
    opts = valid_options(ctx, allowed, max_tier_step=1, max_attempts=max_attempts)
    # Hint only (Option A, 2026-11-07): after a wrong answer with tries left, the model chooses which
    # hint, not whether to move on. Moving on happens after the last try or a correct answer.
    retries = [o for o in opts if o.operation == "retry_current"]
    if retries:
        opts = retries
    return [RetryAction(op="retry", hint=o.cue) if o.operation == "retry_current" else NextAction(op="next", tier=o.tier)
            for o in opts]


class Decider(Protocol):
    def __call__(self, ctx: ContractContext, options: list[Action], session: "SessionRec") -> ModelDecision: ...


def rules_decider(ctx: ContractContext, options: list[Action], session: "SessionRec") -> ModelDecision:
    return ModelDecision(action=plan_order_action(session.plan, ctx, options), source="rules",
                         fallback_reason=None, raw_output=None, latency_ms=0)


# ---------------------------------------------------------------- stored records

class PresRec(BaseModel):
    presentation_id: str
    number: int
    exercise_id: str
    word: str
    level: int
    image_url: str
    hints: list[str] = Field(default_factory=list)
    counted: int = 0
    unclear_streak: int = 0
    attempts: list[ReviewAttempt] = Field(default_factory=list)
    decisions: list[StoredDecision] = Field(default_factory=list)
    result: str | None = None              # set when the picture is left
    shown_at: str                          # when the current prompt (picture or hint) appeared


class SessionRec(BaseModel):
    session_id: str
    patient_id: str
    simulated: bool
    model_version: str
    plan: Plan
    started: str
    finished: str | None = None
    status: Literal["active", "complete", "ended"] = "active"
    pictures: list[PresRec] = Field(default_factory=list)
    history: list[AttemptSummary] = Field(default_factory=list)
    shown: list[str] = Field(default_factory=list)
    misses_in_a_row: int = 0
    n_attempts: int = 0
    n_decisions: int = 0
    seen: dict[str, dict] = Field(default_factory=dict)     # client_attempt_id -> {fingerprint, response}

    @property
    def current(self) -> PresRec | None:
        return self.pictures[-1] if self.status == "active" and self.pictures and self.pictures[-1].result is None else None


class PatientRec(BaseModel):
    patient: Patient
    log: list[LogEntry] = Field(default_factory=list)


# ---------------------------------------------------------------- engine

class Engine:
    def __init__(self, root: str | Path, bank: ItemBank, decider: Decider = rules_decider,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 hint_audio_dir: str | Path | None = None, default_model: str = "rules") -> None:
        self.root, self.bank, self.decider, self.clock = Path(root), bank, decider, clock
        self.hint_audio_dir = Path(hint_audio_dir) if hint_audio_dir else None
        self.default_model = default_model
        self.lock = threading.RLock()
        self.patients: dict[str, PatientRec] = {}
        self.sessions: dict[str, SessionRec] = {}
        self._load()
        from service.steering import Steering
        self.steering = Steering(self)        # corrections, notes, rules, examples (contract §7)
        self.training = None                  # service.training.TrainingManager, set by service/app.py (contract §8)

    # ---- persistence
    def _dir(self, kind: str) -> Path:
        d = self.root / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write(self, kind: str, key: str, model: BaseModel) -> None:
        p = self._dir(kind) / f"{key}.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(model.model_dump_json(indent=1))
        tmp.replace(p)

    def _load(self) -> None:
        for kind, cls, store, key in (("patients", PatientRec, self.patients, lambda r: r.patient.id),
                                      ("sessions", SessionRec, self.sessions, lambda r: r.session_id)):
            d = self.root / kind
            for p in sorted(d.glob("*.json")) if d.exists() else []:
                try:
                    rec = cls.model_validate_json(p.read_text())
                    store[key(rec)] = rec
                except Exception as e:          # a corrupt file must not take the service down
                    print(f"engine: could not load {p}: {e}")

    # ---- media
    def hint_url(self, exercise_id: str, clip: str) -> str | None:
        if self.hint_audio_dir and (self.hint_audio_dir / exercise_id / f"{clip}.m4a").exists():
            return f"/media/hints/{exercise_id}/{clip}.m4a"
        return None

    def audio_path(self, attempt_id: str) -> Path | None:
        d = self.root / "audio"
        hits = sorted(d.glob(f"{attempt_id}.*")) if d.exists() else []
        return hits[0] if hits else None

    # ---- pictures
    def pictures(self) -> list[Picture]:
        return [Picture(exercise_id=i.exercise_id, word=i.preferred_name, accepted=i.accepted_alternatives,
                        level=i.tier, image_url="/media/" + i.image_path,
                        hints={h: HintClip(text=i.cues[h], audio_url=self.hint_url(i.exercise_id, h)) for h in HINTS},
                        whole_word_audio_url=self.hint_url(i.exercise_id, "whole_word"),
                        attribution=f"{i.source.get('name', '')}, {i.source.get('license', '')}".strip(", "))
                for i in self.bank.items]

    # ---- patients
    def _patient(self, pid: str) -> PatientRec:
        if pid not in self.patients:
            raise NotFound(f"no patient {pid!r}")
        return self.patients[pid]

    def add_patient(self, body: NewPatient, pid: str | None = None, simulated: bool = False,
                    plan: Plan | None = None, model: str | None = None, who: str = "system") -> Patient:
        with self.lock:
            pid = pid or f"p-{uuid.uuid4().hex[:8]}"
            if pid in self.patients:
                raise Conflict(f"patient {pid} already exists")
            p = Patient(id=pid, name=body.name, first=body.first, simulated=simulated, about=body.about,
                        plan=plan or DEFAULT_PLAN, active_model=ActiveModel(version=model or self.default_model,
                                                                            patient_adapter=None),
                        pending_examples=0)
            rec = PatientRec(patient=p, log=[LogEntry(when=iso(self.clock()), who=who, what="Patient added with the default plan.")])
            self.patients[pid] = rec
            self._write("patients", pid, rec)
            return p

    def _pub(self, rec: PatientRec) -> Patient:
        return rec.patient.model_copy(update={"pending_examples": self.steering.pending(rec.patient.id)})

    def patient(self, pid: str) -> Patient:
        return self._pub(self._patient(pid))

    def _log(self, pid: str, who: str, what: str) -> None:
        with self.lock:
            rec = self._patient(pid)
            rec.log.append(LogEntry(when=iso(self.clock()), who=who, what=what))
            self._write("patients", pid, rec)

    def set_plan(self, pid: str, plan: Plan, who: str) -> Plan:
        with self.lock:
            rec = self._patient(pid)
            old = rec.patient.plan
            changes = [f"{k}: {getattr(old, k)} → {getattr(plan, k)}" for k in Plan.model_fields if getattr(old, k) != getattr(plan, k)]
            rec.patient = rec.patient.model_copy(update={"plan": plan})
            if changes:
                rec.log.append(LogEntry(when=iso(self.clock()), who=who, what="Plan changed: " + "; ".join(changes) + "."))
            self._write("patients", pid, rec)
            return plan

    def set_model(self, pid: str, version: str, who: str, adapter: str | None = None) -> Patient:
        with self.lock:
            rec = self._patient(pid)
            rec.patient = rec.patient.model_copy(update={"active_model": ActiveModel(version=version, patient_adapter=adapter)})
            rec.log.append(LogEntry(when=iso(self.clock()), who=who, what=f"Active model set to {version}."))
            self._write("patients", pid, rec)
            return self._pub(rec)

    def adapter_version(self, version: str) -> str:
        return self.training.adapter_version(version) if self.training else version

    def log(self, pid: str) -> list[LogEntry]:
        return self._patient(pid).log

    def caseload(self) -> list[PatientRow]:
        rows = []
        for rec in self.patients.values():
            p = rec.patient
            done = [s for s in self.sessions.values() if s.patient_id == p.id and any(x.result for x in s.pictures)]
            last = max(done, key=lambda s: s.started, default=None)
            rows.append(PatientRow(id=p.id, name=p.name, first=p.first, simulated=p.simulated,
                                   last_session=self.summary(last) if last else None))
        return rows

    # ---- sessions
    def _session(self, sid: str) -> SessionRec:
        if sid not in self.sessions:
            raise NotFound(f"no session {sid!r}")
        return self.sessions[sid]

    def _present(self, s: SessionRec, tier: int) -> PresRec:
        item = self.bank.next_item(tier, s.shown)
        s.shown.append(item.exercise_id)
        pr = PresRec(presentation_id=f"p-{len(s.pictures) + 1:02d}", number=len(s.pictures) + 1,
                     exercise_id=item.exercise_id, word=item.preferred_name, level=item.tier,
                     image_url="/media/" + item.image_path, shown_at=iso(self.clock()))
        s.pictures.append(pr)
        return pr

    def _shown_hint(self, pr: PresRec) -> ShownHint | None:
        if not pr.hints:
            return None
        h = pr.hints[-1]
        return ShownHint(type=h, text=self.bank.by_id[pr.exercise_id].cues[h], audio_url=self.hint_url(pr.exercise_id, h))

    def presentation(self, s: SessionRec, pr: PresRec) -> Presentation:
        return Presentation(presentation_id=pr.presentation_id, number=pr.number, of=s.plan.pictures_per_session,
                            exercise_id=pr.exercise_id, word=pr.word, level=pr.level, image_url=pr.image_url,
                            attempt_number=pr.counted + 1, max_attempts=s.plan.max_attempts, hint=self._shown_hint(pr))

    def start(self, pid: str, session_id: str | None = None, simulated: bool = False) -> SessionStarted:
        """`simulated`: a scripted session (scripts/), not a person practising."""
        with self.lock:
            p = self.patient(pid)
            sid = session_id or f"s-{uuid.uuid4().hex[:8]}"
            if sid in self.sessions:
                raise Conflict(f"session {sid} already exists")
            s = SessionRec(session_id=sid, patient_id=pid, simulated=simulated, model_version=p.active_model.version,
                           plan=p.plan, started=iso(self.clock()))
            pr = self._present(s, p.plan.level)
            self.sessions[sid] = s
            self._write("sessions", sid, s)
            return SessionStarted(session_id=sid, patient_id=pid, model_version=s.model_version, plan=s.plan,
                                  picture=self.presentation(s, pr))

    @staticmethod
    def fingerprint(presentation_id: str, audio: bytes | None) -> str:
        """What a client_attempt_id stands for: the picture and the exact audio (None = "I'm not sure")."""
        return hashlib.sha256(json.dumps([presentation_id, hashlib.sha256(audio).hexdigest() if audio is not None
                                          else "no-answer"]).encode()).hexdigest()

    def replay(self, sid: str, client_attempt_id: str, presentation_id: str, audio: bytes | None) -> AttemptResult | None:
        """The stored result if this client_attempt_id was already answered with the same request; Conflict if
        it was used for a different one; None if it is new. Checked before transcribing and scoring."""
        with self.lock:
            prev = self._session(sid).seen.get(client_attempt_id)
            if prev is None:
                return None
            if prev["fingerprint"] != self.fingerprint(presentation_id, audio):
                raise Conflict(f"client_attempt_id {client_attempt_id} was already used for a different answer")
            return AttemptResult(**prev["response"])

    def submit(self, sid: str, presentation_id: str, client_attempt_id: str, score: Score, transcript: str | None,
               audio: tuple[bytes, str] | None = None) -> AttemptResult:
        """Record one answer (recording or "I'm not sure") and return what happens next."""
        fp = self.fingerprint(presentation_id, audio[0] if audio else None)
        with self.lock:
            s = self._session(sid)
            again = self.replay(sid, client_attempt_id, presentation_id, audio[0] if audio else None)
            if again is not None:
                return again
            pr = s.current
            if pr is None:
                raise Conflict(f"session {sid} is {s.status}; no picture is waiting for an answer")
            if pr.presentation_id != presentation_id:
                raise Conflict(f"presentation {presentation_id} is no longer the current picture (now {pr.presentation_id})")
            res = self._apply(s, pr, score, transcript, audio)
            s.seen[client_attempt_id] = {"fingerprint": fp, "response": res.model_dump(mode="json")}
            self._write("sessions", sid, s)
            return res

    def _apply(self, s: SessionRec, pr: PresRec, score: Score, transcript: str | None,
               audio: tuple[bytes, str] | None) -> AttemptResult:
        now = self.clock()
        s.n_attempts += 1
        aid = f"{s.session_id}-a{s.n_attempts:02d}"
        if audio:
            (self._dir("audio") / f"{aid}.{audio[1]}").write_bytes(audio[0])
        counted = score.outcome != "unclear"
        shown_hint = pr.hints[-1] if pr.hints else None
        if counted:
            pr.counted += 1
            pr.unclear_streak = 0
        pr.attempts.append(ReviewAttempt(
            attempt_id=aid, number=pr.counted if counted else None, hint_shown=shown_hint, transcript=transcript,
            outcome=score.outcome, error_type=score.error_type, counted=counted, reason=score.reason,
            audio_url=f"/attempts/{aid}/audio" if audio else None, at=iso(now)))
        nxt: dict = {}
        leave_tier: int | None = None
        decision_id = None
        if not counted:
            pr.unclear_streak += 1
            kind = "try_again" if pr.unclear_streak == 1 else "not_assessed"
            if kind == "not_assessed":
                leave_tier = pr.level
        else:
            rt = max(0.0, min(120.0, (now - datetime.fromisoformat(pr.shown_at.replace("Z", "+00:00"))).total_seconds()))
            s.history.append(AttemptSummary(exercise_id=pr.exercise_id, word=pr.word, tier=pr.level,
                                            hint=shown_hint or "none", correct=score.outcome == "correct",
                                            error=score.error_type, response_time_s=round(rt, 1)))
            last_picture = pr.number >= s.plan.pictures_per_session
            if score.outcome == "correct" and last_picture:
                kind = "correct"
            else:
                d = self._decide(s, pr, aid, now)
                decision_id = d.decision_id
                if score.outcome == "correct":
                    kind, leave_tier = "correct", d.action.tier
                elif pr.counted >= s.plan.max_attempts:
                    kind, leave_tier = "show_word", d.action.tier
                elif isinstance(d.action, RetryAction):
                    kind = "hint"
                    pr.hints.append(d.action.hint)
                else:
                    kind, leave_tier = "move_on", d.action.tier
        nxt.update(kind=kind, message=MESSAGES[kind].format(word=pr.word))
        if kind == "hint":
            nxt.update(hint=self._shown_hint(pr), attempt_number=pr.counted + 1, max_attempts=s.plan.max_attempts,
                       decision_id=decision_id)
        if kind == "show_word":
            nxt.update(word=pr.word, whole_word_audio_url=self.hint_url(pr.exercise_id, "whole_word"))
        if kind in ("correct", "show_word", "move_on"):
            nxt["decision_id"] = decision_id
        pr.shown_at = iso(now)
        if kind in ("correct", "show_word", "not_assessed", "move_on"):
            pr.result = picture_result(pr.attempts)
            if pr.result == "unsuccessful":
                s.misses_in_a_row += 1
            elif pr.result != "not_assessed":
                s.misses_in_a_row = 0
            if pr.number >= s.plan.pictures_per_session:
                s.status, s.finished = "complete", iso(now)
                nxt.update(session_complete=True, summary=self.summary(s))
            else:
                new = self._present(s, leave_tier if leave_tier is not None else pr.level)
                nxt.update(picture=self.presentation(s, new),
                           break_suggested=s.misses_in_a_row >= s.plan.break_after_misses)
                if nxt["break_suggested"]:
                    s.misses_in_a_row = 0
        return AttemptResult(attempt_id=aid, presentation_id=pr.presentation_id, counted=counted,
                             transcript=transcript, score=score, next=NextStep(**nxt))

    def _decide(self, s: SessionRec, pr: PresRec, aid: str, now: datetime) -> StoredDecision:
        s.n_decisions += 1
        cur = CurrentExercise(exercise_id=pr.exercise_id, word=pr.word, tier=pr.level, hints_shown=list(pr.hints),
                              attempts=pr.counted)
        started = datetime.fromisoformat(s.started.replace("Z", "+00:00"))
        ctx = ContractContext(patient_id=s.patient_id, current=cur, history=list(s.history),
                              session_minutes=round((now - started).total_seconds() / 60, 1))
        options = decision_options(ctx, pr.hints, s.plan.max_attempts)
        try:
            md = self.decider(ctx, options, s)
            if md.action not in options:
                raise ValueError(f"{md.action.model_dump()} is not one of the options")
        except Exception as e:                   # never let the model break practice: fall back, say why
            md = ModelDecision(action=plan_order_action(s.plan, ctx, options), source="fallback:plan-order",
                               fallback_reason=f"{type(e).__name__}: {str(e)[:200]}", raw_output=None, latency_ms=0)
        d = StoredDecision(decision_id=f"{s.session_id}-d{s.n_decisions:02d}", session_id=s.session_id,
                           presentation_id=pr.presentation_id, after_attempt_id=aid, context=ctx, options=options,
                           action=md.action, source=md.source, fallback_reason=md.fallback_reason,
                           latency_ms=md.latency_ms, at=iso(now))
        pr.decisions.append(d)
        return d

    def end(self, sid: str) -> SessionRow:
        with self.lock:
            s = self._session(sid)
            if s.status == "active":
                if s.pictures and s.pictures[-1].result is None:
                    s.pictures.pop()                      # the picture on screen was never finished
                s.status, s.finished = "ended", iso(self.clock())
                self._write("sessions", sid, s)
            return self.summary(s)

    def state(self, sid: str) -> SessionState:
        s = self._session(sid)
        pr = s.current
        done = sum(1 for p in s.pictures if p.result)
        return SessionState(session_id=sid, patient_id=s.patient_id, status=s.status,
                            picture=self.presentation(s, pr) if pr else None, pictures_done=done,
                            summary=None if s.status == "active" else self.summary(s))

    # ---- review
    def _with_correction(self, s: SessionRec, d: StoredDecision) -> StoredDecision:
        c = self.steering.correction_for(s.patient_id, d.decision_id)
        return d.model_copy(update={"correction": c}) if c else d

    def _review_pictures(self, s: SessionRec) -> list[ReviewPicture]:
        return [ReviewPicture(presentation_id=p.presentation_id, exercise_id=p.exercise_id, word=p.word, level=p.level,
                              image_url=p.image_url, result=p.result, attempts=p.attempts,
                              decisions=[self._with_correction(s, d) for d in p.decisions])
                for p in s.pictures if p.result]

    def summary(self, s: SessionRec) -> SessionRow:
        pics = self._review_pictures(s)
        last = s.finished or (pics[-1].attempts[-1].at if pics else s.started)
        minutes = (datetime.fromisoformat(last.replace("Z", "+00:00"))
                   - datetime.fromisoformat(s.started.replace("Z", "+00:00"))).total_seconds() / 60
        return summarize(s.session_id, s.started[:10], s.simulated, s.model_version, pics, minutes=round(minutes, 1),
                         ended_early=s.status == "ended" and len(pics) < s.plan.pictures_per_session)

    def patient_sessions(self, pid: str) -> list[SessionRow]:
        self._patient(pid)
        ss = sorted((s for s in self.sessions.values() if s.patient_id == pid), key=lambda s: s.started)
        return [self.summary(s) for s in ss]

    def review(self, sid: str) -> SessionReview:
        s = self._session(sid)
        return SessionReview(session_id=sid, summary=self.summary(s), pictures=self._review_pictures(s))
