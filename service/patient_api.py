"""Patient practice and review endpoints of docs/API_CONTRACT.md (§2, §3, §4, §6), on the engine in
service/engine.py. Mounted on the agent service (service/app.py) next to the older /practice routes.

Each recording is transcribed (local Whisper), scored by fixed rules (speech/scoring.py), and passed
to the engine, which asks the patient's decision model what happens next. Models:
  rules          the plan's hint order (no language model)
  ditto:<run>    a trained adapter on the local Qwen3-1.7B (agents/registry.py)
Invalid output, a timeout or a load failure falls back to the plan's hint order, recorded with the
reason. The language model loads in the background when a session starts.
"""
from __future__ import annotations

import threading

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from core.contract import (
    ActivateIn, AttemptResult, CorrectionIn, Example, ExamplesIn, LogEntry, ModelDecision, ModelVersion, NewPatient, NextAction, NoAnswerIn, Note,
    NoteIn, Patient, PatientRow, Picture, Plan, RetryAction, RuleIn, RuleMatch, Score, SessionReview, SessionRow,
    SessionStarted, SessionState, Signals, TrainingRun, UsePreparedIn,
)
from core.practice import PracticeAction
from service.engine import Conflict, Engine, Invalid, NotFound, plan_order_action, rules_decider
from speech.scoring import SpeechUnavailable, score

AUDIO_EXT = {"webm": "webm", "ogg": "ogg", "mp4": "mp4", "m4a": "m4a", "aac": "m4a", "mpeg": "mp3", "wav": "wav",
             "x-wav": "wav", "aiff": "aiff", "x-aiff": "aiff"}
AUDIO_TYPE = {"webm": "audio/webm", "ogg": "audio/ogg", "mp4": "audio/mp4", "m4a": "audio/mp4", "mp3": "audio/mpeg",
              "wav": "audio/wav", "aiff": "audio/aiff", "bin": "application/octet-stream"}


def audio_ext(content_type: str | None) -> str:
    sub = (content_type or "").split(";")[0].split("/")[-1].strip().lower()
    return AUDIO_EXT.get(sub, "bin")


class _PlanFallback:
    """The fallback object agents/llm_policy.LLMDecisionPolicy expects, answering with the plan's hint order."""

    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    def decide(self, ctx, options, session_id=None, trial_count=0, requested=None, reason=None, extra=None):
        from agents.llm_policy import Decision
        from core.practice import format_action
        contract_opts = [_to_contract(o) for o in options]
        a = plan_order_action(self.plan, ctx, contract_opts)
        return Decision(action=_to_practice(a), source="fallback:plan-order", requested_model=requested or "",
                        model_detail=extra or {}, fallback_reason=reason, latency_ms=0.0,
                        options=[format_action(o) for o in options])


def _to_practice(a) -> PracticeAction:
    if isinstance(a, RetryAction):
        return PracticeAction(operation="retry_current", cue=a.hint)
    return PracticeAction(operation="next_exercise", tier=a.tier)


def _to_contract(o: PracticeAction):
    if o.operation == "retry_current":
        return RetryAction(op="retry", hint=o.cue)
    return NextAction(op="next", tier=o.tier)


def model_decider(registry, resolve=lambda v: v):
    """Decider for the engine: the session's model version via the model registry. `resolve` maps a version to
    the registry version behind it (a run that took a prepared run's result serves that run's adapter)."""
    def decide(ctx, options, s) -> ModelDecision:
        if s.model_version == "rules":
            return rules_decider(ctx, options, s)
        pol = registry.policy(resolve(s.model_version))        # raises if unavailable -> engine falls back
        d = pol.decide(ctx, [_to_practice(o) for o in options], _PlanFallback(s.plan), s.session_id, len(s.history))
        return ModelDecision(action=_to_contract(d.action), source=d.source, fallback_reason=d.fallback_reason,
                             raw_output=d.raw_output, latency_ms=int(d.latency_ms))
    return decide


def default_model(registry) -> str:
    """The model configs/practice.yaml names as default (the delivered shared adapter, e.g.
    ditto:sft200-g2-s0) if it is available, else the newest available DITTO adapter, else the rules."""
    try:
        ds = [v["id"] for v in registry.versions() if v["id"].startswith("ditto:") and v.get("available")]
    except Exception:
        ds = []
    preferred = getattr(getattr(registry, "pcfg", None), "default_model", None)
    if preferred in ds:
        return preferred
    return ds[-1] if ds else "rules"


def patient_router(engine: Engine, get_speech, registry=None, rule_drafter=None) -> APIRouter:
    r = APIRouter(tags=["patient practice (contract)"])

    def http(e: Exception) -> HTTPException:
        if isinstance(e, NotFound):
            return HTTPException(404, str(e).strip("'\""))
        if isinstance(e, Conflict):
            return HTTPException(409, str(e))
        if isinstance(e, Invalid):
            return HTTPException(422, str(e))
        raise e

    def warm_up(version: str) -> None:
        if registry is None or version == "rules":
            return
        threading.Thread(target=lambda: _quiet(registry.policy, engine.adapter_version(version)), daemon=True).start()

    # §2
    @r.get("/pictures")
    def pictures() -> list[Picture]:
        return engine.pictures()

    # §3
    @r.get("/patients")
    def patients() -> list[PatientRow]:
        return engine.caseload()

    @r.get("/patients/{pid}")
    def patient(pid: str) -> Patient:
        try:
            return engine.patient(pid)
        except NotFound as e:
            raise http(e)

    @r.post("/patients", status_code=201)
    def add_patient(body: NewPatient) -> Patient:
        return engine.add_patient(body, who="clinician")

    @r.put("/patients/{pid}/plan")
    def set_plan(pid: str, body: Plan) -> Plan:
        try:
            return engine.set_plan(pid, body, who="clinician")
        except NotFound as e:
            raise http(e)

    @r.get("/patients/{pid}/log")
    def log(pid: str) -> list[LogEntry]:
        try:
            return engine.log(pid)
        except NotFound as e:
            raise http(e)

    # §4
    @r.post("/patients/{pid}/sessions", status_code=201)
    def start(pid: str) -> SessionStarted:
        try:
            s = engine.start(pid)
        except (NotFound, Conflict) as e:
            raise http(e)
        warm_up(s.model_version)
        return s

    @r.post("/sessions/{sid}/recordings")
    def recording(sid: str, audio: UploadFile = File(...), presentation_id: str = Form(...),
                  client_attempt_id: str = Form(..., min_length=8, max_length=64)) -> AttemptResult:
        data = audio.file.read()
        try:
            again = engine.replay(sid, client_attempt_id, presentation_id, data if data else None)
            if again is not None:                               # the same request again: the stored result
                return again
            target, alts = _names(engine, sid)                  # 404 / 409 before spending time on speech
        except (NotFound, Conflict) as e:
            raise http(e)
        t = get_speech().transcribe(data, audio.content_type)
        try:
            s = score(t, target, alts)
        except SpeechUnavailable as e:
            raise HTTPException(503, f"speech recognition unavailable: {e}")
        try:
            return engine.submit(sid, presentation_id, client_attempt_id, s, t.text if t.status == "ok" else "",
                                 (data, audio_ext(audio.content_type)) if data else None)
        except (NotFound, Conflict) as e:
            raise http(e)

    @r.post("/sessions/{sid}/no-answer")
    def no_answer(sid: str, body: NoAnswerIn) -> AttemptResult:
        s = Score(outcome="incorrect", error_type="no_response", heard_word=None, matched_name=None,
                  reason="The patient pressed “I'm not sure”.", signals=Signals())
        try:
            again = engine.replay(sid, body.client_attempt_id, body.presentation_id, None)
            return again if again is not None else engine.submit(sid, body.presentation_id, body.client_attempt_id, s, None)
        except (NotFound, Conflict) as e:
            raise http(e)

    @r.post("/sessions/{sid}/end")
    def end(sid: str) -> SessionRow:
        try:
            return engine.end(sid)
        except NotFound as e:
            raise http(e)

    @r.get("/sessions/{sid}")
    def state(sid: str) -> SessionState:
        try:
            return engine.state(sid)
        except NotFound as e:
            raise http(e)

    # §6
    @r.get("/patients/{pid}/sessions")
    def patient_sessions(pid: str) -> list[SessionRow]:
        try:
            return engine.patient_sessions(pid)
        except NotFound as e:
            raise http(e)

    @r.get("/sessions/{sid}/review")
    def review(sid: str) -> SessionReview:
        try:
            return engine.review(sid)
        except NotFound as e:
            raise http(e)

    @r.get("/attempts/{aid}/audio")
    def attempt_audio(aid: str) -> FileResponse:
        p = engine.audio_path(aid)
        if p is None:
            raise HTTPException(404, f"no recording for attempt {aid!r}")
        return FileResponse(p, media_type=AUDIO_TYPE.get(p.suffix.lstrip("."), "application/octet-stream"))

    # §7 steering
    @r.post("/decisions/{decision_id}/correction", status_code=201)
    def correction(decision_id: str, body: CorrectionIn) -> Example:
        try:
            return engine.steering.correct(decision_id, body)[0]
        except (NotFound, Conflict, Invalid) as e:
            raise http(e)

    @r.get("/patients/{pid}/examples")
    def examples(pid: str) -> list[Example]:
        try:
            return engine.steering.examples(pid)
        except NotFound as e:
            raise http(e)

    @r.post("/patients/{pid}/notes", status_code=201)
    def add_note(pid: str, body: NoteIn) -> Note:
        try:
            return engine.steering.add_note(pid, body.text, rule_drafter)
        except NotFound as e:
            raise http(e)

    @r.put("/notes/{note_id}/rule")
    def confirm_rule(note_id: str, body: RuleIn) -> Note:
        try:
            return engine.steering.confirm_rule(note_id, body.rule)
        except (NotFound, Conflict, Invalid) as e:
            raise http(e)

    @r.get("/notes/{note_id}/matches")
    def matches(note_id: str) -> list[RuleMatch]:
        try:
            return engine.steering.matches(note_id)
        except (NotFound, Conflict) as e:
            raise http(e)

    @r.post("/notes/{note_id}/examples", status_code=201)
    def note_examples(note_id: str, body: ExamplesIn) -> list[Example]:
        try:
            return engine.steering.examples_from_note(note_id, body.decision_ids)
        except (NotFound, Conflict, Invalid) as e:
            raise http(e)

    # §8 model updates
    @r.post("/patients/{pid}/training-runs", status_code=202)
    def start_training(pid: str) -> TrainingRun:
        try:
            return engine.training.start(pid)
        except (NotFound, Conflict, Invalid) as e:
            raise http(e)

    @r.get("/training-runs/{run_id}")
    def training_run(run_id: str) -> TrainingRun:
        try:
            return engine.training.get(run_id)
        except NotFound as e:
            raise http(e)

    @r.post("/training-runs/{run_id}/use-prepared")
    def use_prepared(run_id: str, body: UsePreparedIn) -> TrainingRun:
        try:
            return engine.training.use_prepared(run_id, body)
        except (NotFound, Conflict) as e:
            raise http(e)

    def model_versions(pid: str) -> list[ModelVersion]:
        p = engine.patient(pid)
        active = p.active_model.version
        out = [ModelVersion(id="rules", label="The plan's hint order (no language model)", kind="baseline",
                            available=True, active=active == "rules")]
        mine = {x.run_id for x in engine.training.recs.values()}
        try:
            shared = [v for v in (registry.versions() if registry else [])
                      if v["id"].startswith("ditto:") and v["id"][len("ditto:"):] not in mine]
        except Exception:
            shared = []
        out += [ModelVersion(id=v["id"], label=f"Shared starting adapter ({v['run_id']})", kind="shared",
                             available=bool(v.get("available")), active=active == v["id"]) for v in shared]
        for x in engine.training.completed(pid):
            how = f", result prepared earlier ({x.prepared_run_id})" if x.prepared_run_id else ""
            out.append(ModelVersion(id=f"ditto:{x.run_id}", label=f"{p.first}'s adapter (run {x.run_id}{how})",
                                    kind="patient", available=True, active=active == f"ditto:{x.run_id}"))
        return out

    @r.get("/patients/{pid}/models")
    def models(pid: str) -> list[ModelVersion]:
        try:
            return model_versions(pid)
        except NotFound as e:
            raise http(e)

    @r.post("/patients/{pid}/active-model")
    def activate(pid: str, body: ActivateIn) -> Patient:
        """Applies to sessions started afterwards; a session already running keeps the model it started with."""
        try:
            v = next((m for m in model_versions(pid) if m.id == body.version), None)
            if v is None or not v.available:
                raise Invalid(f"{body.version!r} is not a model this patient can use "
                              "(rules, an available shared adapter, or one of this patient's completed runs)")
            return engine.set_model(pid, v.id, "clinician", adapter=v.id[len("ditto:"):] if v.kind == "patient" else None)
        except (NotFound, Invalid) as e:
            raise http(e)

    return r


def _names(engine: Engine, sid: str) -> tuple[str, list[str]]:
    s = engine._session(sid)
    pr = s.current
    if pr is None:
        raise Conflict(f"session {sid} is {s.status}; no picture is waiting for an answer")
    item = engine.bank.by_id[pr.exercise_id]
    return item.preferred_name, item.accepted_alternatives


def _quiet(f, *a) -> None:
    try:
        f(*a)
    except Exception as e:          # loading failures surface as fallbacks with a reason at decision time
        print(f"model warm-up failed: {type(e).__name__}: {e}")
