"""Build the sample responses in docs/contract_examples/ from the contract models (core/contract.py).

The practice and review samples are produced by the real practice engine (service/engine.py): one
scripted session is played through it, with scripted scores and scripted model choices, so the
samples show exactly what the backend returns. The steering and training samples (contract §7, §8)
are built from the contract models directly, until those endpoints exist.

  1 cat    "cat"                         correct first try                    -> independent
  2 dog    "bone" -> meaning hint -> "uuummm dog"                             -> assisted
  3 apple  "it's an apple" (unclear) -> "I'm not sure" -> sentence hint -> "apple" -> assisted
  4 house  silence, silence (2 unclear)                                       -> not assessed
  5 car    "bus" -> meaning hint -> "truck" -> sentence hint -> "train" -> show the word -> unsuccessful
  6 tree   "free" (sound error) -> first-sound hint -> "tree"                -> assisted

Picture 5 is the demo situation: after two failed tries with a meaning hint already given, the model
chose the sentence hint; the clinician's note asks for the first-sound hint there.

Scores and the model's choices are scripted (the scorer and the model aren't called). Preview numbers
are illustrative placeholders, not measurements.

Run:  uv run python -m scripts.build_contract_examples          (writes docs/contract_examples/)
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.config import resolve
from core.contract import (
    HINTS, ActivateIn, ActiveModel, Agreement, CorrectionIn, ErrorBody, Example, ExamplesIn, LogEntry,
    ModelDecision, ModelVersion, NewPatient, NextAction, NoAnswerIn, Note, NoteIn, Plan, Preview, Progress, Range,
    RetryAction, RetryThen, Rule, RuleIn, RuleMatch, RuleWhen, Score, Signals, StoredDecision, TrainingRun,
    Unchanged, UsePreparedIn, rule_action, rule_matches, situation,
)
from core.practice import load_items
from service.engine import Engine

OUT = "docs/contract_examples"
PATIENT, FIRST, NAME = "demo-ana", "Ana", "Ana (demo)"
SID, DATE = "s-demo-01", "2026-11-05"
SHARED = "ditto:demoA-qwen3-1.7b-r1"
T0 = datetime(2026, 11, 5, 15, 0, tzinfo=UTC)
PLAN = Plan(level=1, pictures_per_session=6, max_attempts=3, break_after_misses=3, hint_order="match_error",
            session_minutes=10, days_per_week=5, personal_words=[])
NOTE_TEXT = "When meaning hints have repeatedly failed, try a first-sound hint before moving to another picture."
CORRECTION_NOTE = "For this patient, try a sound cue before another meaning cue."


def R(hint):
    return RetryAction(op="retry", hint=hint)


def N(tier):
    return NextAction(op="next", tier=tier)


# Each answer: (kind, transcript, outcome, error, heard, matched, reason, response_s, model_choice)
# kind "rec" = recording, "none" = the "I'm not sure" button. model_choice is the scripted model
# decision after a counted attempt (None where the engine doesn't ask the model).
SCRIPT = [
    [("rec", "cat", "correct", None, "cat", "cat", "One word, “cat”, matches “cat”.", 1.6, N(1))],
    [("rec", "bone", "incorrect", "meaning", "bone", None, "One word, “bone”, a different word from “dog”.", 3.1, R("semantic")),
     ("rec", "uuummm dog", "correct", None, "dog", "dog", "After removing fillers and repeats, one word, “dog”, matches “dog”.", 3.8, N(1))],
    [("rec", "it's an apple", "unclear", None, None, None, "More than one word (“it's an apple”); only single-word answers are scored.", 2.4, None),
     ("none", None, "incorrect", "no_response", None, None, "The patient pressed “I'm not sure”.", 5.0, R("sentence_frame")),
     ("rec", "apple", "correct", None, "apple", "apple", "One word, “apple”, matches “apple”.", 2.2, N(1))],
    [("rec", "", "unclear", None, None, None, "No speech detected.", 4.0, None),
     ("rec", "", "unclear", None, None, None, "No speech detected.", 4.0, None)],
    [("rec", "bus", "incorrect", "meaning", "bus", None, "One word, “bus”, a different word from “car”.", 2.9, R("semantic")),
     ("rec", "truck", "incorrect", "meaning", "truck", None, "One word, “truck”, a different word from “car”.", 3.5, R("sentence_frame")),
     ("rec", "train", "incorrect", "meaning", "train", None, "One word, “train”, a different word from “car”.", 3.9, N(1))],
    [("rec", "free", "incorrect", "sound", "free", None, "One word, “free”, sounds close to “tree”.", 2.7, R("first_sound")),
     ("rec", "tree", "correct", None, "tree", "tree", "One word, “tree”, matches “tree”.", 2.0, None)],
]


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def play(engine: Engine, clock: Clock, choices: list) -> tuple[list[tuple[str, object]], object, object]:
    """Play SCRIPT through the engine. Returns the (endpoint, AttemptResult) steps, the session start and the
    session state while picture 2's hint is on screen."""
    started = engine.start(PATIENT, session_id=SID, simulated=True)
    pid = started.picture.presentation_id
    steps, mid = [], None
    k = 0
    for answers in SCRIPT:
        for kind, transcript, outcome, error, heard, matched, reason, rt, choice in answers:
            k += 1
            clock.t += timedelta(seconds=rt)
            if choice is not None:
                choices.append(choice)
            sig = Signals() if kind == "none" else Signals(
                audio_seconds=round(rt * 0.6, 1), no_speech_prob=0.92 if transcript == "" else 0.03,
                word_confidence=None if not transcript else 0.84)
            score = Score(outcome=outcome, error_type=error, heard_word=heard, matched_name=matched, reason=reason,
                          signals=sig)
            res = engine.submit(SID, pid, f"c-{k:04d}-0000", score, transcript,
                                None if kind == "none" else (f"scripted audio {k}".encode(), "webm"))
            assert not choices, f"step {k}: the engine didn't ask the model where the script expected it"
            steps.append(("no-answer" if kind == "none" else "recordings", res))
            if res.next.picture:
                pid = res.next.picture.presentation_id
            if mid is None and res.next.kind == "hint" and res.presentation_id == "p-02":
                mid = engine.state(SID)
            clock.t += timedelta(seconds=6)
    return steps, started, mid


def build(out: Path) -> dict:
    bank = load_items()
    work = Path(tempfile.mkdtemp())
    hints = work / "hints"                       # every clip "exists", so samples show where audio will be
    for it in bank.items:
        (hints / it.exercise_id).mkdir(parents=True)
        for clip in [*HINTS, "whole_word"]:
            (hints / it.exercise_id / f"{clip}.m4a").write_bytes(b"")
    clock = Clock(T0 - timedelta(days=4))
    choices: list = []

    def scripted(ctx, options, s):
        if not choices:
            raise AssertionError("the engine asked the model where the script has no choice")
        a = choices.pop()
        assert a in options, f"scripted choice {a} is not one of the options {options}"
        return ModelDecision(action=a, source=SHARED, fallback_reason=None, raw_output=None, latency_ms=560)

    engine = Engine(work / "data", bank, decider=scripted, clock=clock, hint_audio_dir=hints, default_model=SHARED)
    engine.add_patient(NewPatient(name=NAME, first=FIRST,
                                  about="Simulated demo patient. Answers are scripted Mac-voice clips played through the real pipeline."),
                       pid=PATIENT, simulated=True, plan=PLAN, model=SHARED, who="Dr. Patel, SLP")
    clock.t = T0
    steps, started, mid = play(engine, clock, choices)
    review = engine.review(SID)
    summary = review.summary
    files: list[dict] = []

    def put(rel: str, obj, method: str, path: str, status: int = 200, note: str = "", request=None):
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        body = [o.model_dump(mode="json") for o in obj] if isinstance(obj, list) else obj.model_dump(mode="json")
        p.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
        entry = {"method": method, "path": path, "status": status, "file": rel,
                 "model": f"list[{type(obj[0]).__name__}]" if isinstance(obj, list) else type(obj).__name__}
        if request is not None:
            req_rel = rel.replace(".json", ".request.json")
            (out / req_rel).write_text(json.dumps(request.model_dump(mode="json", exclude_none=True), indent=2,
                                                  ensure_ascii=False) + "\n")
            entry.update(request_file=req_rel, request_model=type(request).__name__)
        if note:
            entry["note"] = note
        files.append(entry)

    # §2 pictures
    put("pictures/GET_pictures.json", engine.pictures(), "GET", "/pictures",
        note="audio_url values are where the clips will be; until they are generated the real server returns null "
             "and the mock server a placeholder tone")

    # §3 patients and plans
    patient = engine.patient(PATIENT)
    put("patients/GET_patients.json", engine.caseload(), "GET", "/patients")
    put("patients/GET_patient.json", patient, "GET", "/patients/{patient_id}")
    put("patients/POST_patients.json", patient.model_copy(update={"id": "p-0002", "name": "New patient", "first": "Sam",
                                                                  "simulated": False, "about": ""}),
        "POST", "/patients", status=201, request=NewPatient(name="New patient", first="Sam"))
    new_plan = PLAN.model_copy(update={"max_attempts": 4})
    put("patients/PUT_plan.json", new_plan, "PUT", "/patients/{patient_id}/plan", request=new_plan)

    # §4 practice
    put("practice/POST_session.json", started, "POST", "/patients/{patient_id}/sessions", status=201)
    script_files = []
    for k, (endpoint, res) in enumerate(steps, start=1):
        rel = f"practice/script/{k:02d}_{endpoint}_{res.next.kind}.json"
        put(rel, res, "POST", f"/sessions/{{session_id}}/{endpoint}",
            note="one step of the scripted session; the mock server returns these in order",
            request=NoAnswerIn(presentation_id=res.presentation_id, client_attempt_id=f"c-{k:04d}-0000")
            if endpoint == "no-answer" else None)
        script_files.append(rel)
    put("practice/GET_session_state.json", mid, "GET", "/sessions/{session_id}",
        note="mid-session: picture 2 with the meaning hint on screen")
    put("practice/POST_end.json", summary, "POST", "/sessions/{session_id}/end")
    put("practice/error_409_stale.json", ErrorBody(detail="presentation p-01 is no longer the current picture (now p-02)"),
        "POST", "/sessions/{session_id}/recordings", status=409, note="error example")
    put("practice/error_503_speech.json", ErrorBody(detail="speech recognition unavailable: not installed"),
        "POST", "/sessions/{session_id}/recordings", status=503, note="error example")

    # §6 review (with the correction of §7 attached to its decision)
    decisions: list[StoredDecision] = [d for p in review.pictures for d in p.decisions]
    target = next(d for d in decisions if d.presentation_id == "p-02" and d.action == R("semantic"))
    correction = Example(example_id="x-0001", patient_id=PATIENT, source="correction", decision_id=target.decision_id,
                         note_id=None, model_chose=target.action, preferred=R("first_sound"), note=CORRECTION_NOTE,
                         status="pending")
    target.correction = correction
    put("review/GET_patient_sessions.json", engine.patient_sessions(PATIENT), "GET", "/patients/{patient_id}/sessions")
    put("review/GET_session_review.json", review, "GET", "/sessions/{session_id}/review",
        note=f"decision {target.decision_id} already carries the clinician's correction")

    # §7 steering
    put("steering/POST_correction.json", correction, "POST", "/decisions/{decision_id}/correction", status=201,
        request=CorrectionIn(preferred=R("first_sound"), note=CORRECTION_NOTE))
    put("steering/error_422_correction.json", ErrorBody(detail="preferred action {\"op\":\"retry\",\"hint\":\"semantic\"} "
                                                              "is not one of this decision's options"),
        "POST", "/decisions/{decision_id}/correction", status=422, note="error example")
    rule = Rule(when=RuleWhen(hints_tried_include=["semantic"], failed_attempts_on_picture=Range(min=2)),
                then=RetryThen(op="retry", hint="first_sound"),
                summary="After 2 or more failed tries on a picture where a meaning hint was already given → give the first-sound hint.")
    draft = Note(note_id="n-0001", patient_id=PATIENT, text=NOTE_TEXT, rule=rule, rule_source="claude",
                 rule_status="draft", problem=None)
    put("steering/POST_note.json", draft, "POST", "/patients/{patient_id}/notes", status=201, request=NoteIn(text=NOTE_TEXT))
    put("steering/POST_note_no_rule.json",
        Note(note_id="n-0002", patient_id=PATIENT, text="She seems tired on Mondays.", rule=None, rule_source="none",
             rule_status="none", problem="Notes can steer hints and difficulty only; this one doesn't ask for either."),
        "POST", "/patients/{patient_id}/notes", status=201, request=NoteIn(text="She seems tired on Mondays."),
        note="drafting failed or nothing to steer: the UI shows the 'when ___ → do ___' form")
    confirmed = draft.model_copy(update={"rule_status": "confirmed"})
    put("steering/PUT_rule.json", confirmed, "PUT", "/notes/{note_id}/rule", request=RuleIn(rule=rule))
    matches = []
    for d in decisions:
        if not rule_matches(rule, d.context):
            continue
        want = rule_action(rule, d.context)
        ok = want in d.options
        matches.append(RuleMatch(decision_id=d.decision_id, date=DATE, word=d.context.current.word,
                                 situation=situation(d.context), model_chose=d.action, rule_says=want if ok else None,
                                 already_agrees=ok and want == d.action, allowed=ok))
    put("steering/GET_matches.json", matches, "GET", "/notes/{note_id}/matches")
    ticked = [m for m in matches if m.allowed and not m.already_agrees]
    note_examples = [Example(example_id=f"x-{k:04d}", patient_id=PATIENT, source="note", decision_id=m.decision_id,
                             note_id="n-0001", model_chose=m.model_chose, preferred=m.rule_says, note=NOTE_TEXT,
                             status="pending") for k, m in enumerate(ticked, start=2)]
    put("steering/POST_note_examples.json", note_examples, "POST", "/notes/{note_id}/examples", status=201,
        request=ExamplesIn(decision_ids=[m.decision_id for m in ticked]))
    all_examples = [correction, *note_examples]
    put("steering/GET_examples.json", all_examples, "GET", "/patients/{patient_id}/examples")

    # §8 model updates
    t_end = clock.t
    ids = [e.example_id for e in all_examples]
    run = {"run_id": "demo-ana-r1", "patient_id": PATIENT, "base": SHARED, "example_ids": ids, "error": None}
    put("training/POST_training_run.json",
        TrainingRun(**run, status="queued", progress=Progress(stage="queued"), started=None, finished=None, preview=None),
        "POST", "/patients/{patient_id}/training-runs", status=202)
    put("training/GET_training_run_running.json",
        TrainingRun(**run, status="running", progress=Progress(stage="ditto", round=2, of=3), started=iso(t_end),
                    finished=None, preview=None),
        "GET", "/training-runs/{run_id}")
    preview = Preview(
        follows_your_examples=Agreement(new=len(ids), current=0, total=len(ids),
                                        label="trained on these: shows the update took, not evidence"),
        follows_your_rules_new_situations=Agreement(new=7, current=1, total=10),
        unchanged_elsewhere=Unchanged(same=9, total=10))
    done = TrainingRun(**run, status="completed", progress=Progress(stage="done"), started=iso(t_end),
                       finished=iso(t_end + timedelta(minutes=5)), preview=preview, prepared=True,
                       prepared_run_id="demo-ana-r1-prepared")
    put("training/POST_use_prepared.json", done, "POST", "/training-runs/{run_id}/use-prepared",
        request=UsePreparedIn(prepared_run_id="demo-ana-r1-prepared"),
        note="ILLUSTRATIVE preview numbers (placeholders, not measured)")
    put("training/GET_training_run_completed.json", done, "GET", "/training-runs/{run_id}",
        note="ILLUSTRATIVE preview numbers (placeholders, not measured)")
    put("training/error_409_running.json", ErrorBody(detail="a training run is already in progress (demo-ana-r1)"),
        "POST", "/patients/{patient_id}/training-runs", status=409, note="error example")
    patient_version = "ditto:demo-ana-r1"      # a patient adapter is served by its own run id
    put("training/GET_models.json", [
        ModelVersion(id="rules", label="The plan's hint order (no language model)", kind="baseline", available=True, active=False),
        ModelVersion(id=SHARED, label="Shared starting adapter", kind="shared", available=True, active=True),
        ModelVersion(id=patient_version, label="Ana's adapter (run demo-ana-r1)", kind="patient", available=True, active=False),
    ], "GET", "/patients/{patient_id}/models")
    activated = patient.model_copy(update={"active_model": ActiveModel(version=patient_version, patient_adapter="demo-ana-r1")})
    put("training/POST_active_model.json", activated, "POST", "/patients/{patient_id}/active-model",
        request=ActivateIn(version=patient_version))
    put("patients/GET_log.json", [
        *engine.log(PATIENT),
        LogEntry(when=iso(t_end), who="Dr. Patel, SLP", what=f"Rule confirmed from note: “{NOTE_TEXT}”"),
        LogEntry(when=iso(t_end + timedelta(minutes=6)), who="Dr. Patel, SLP",
                 what=f"Activated Ana's adapter (run demo-ana-r1, trained on {len(ids)} examples)."),
    ], "GET", "/patients/{patient_id}/log")

    shutil.rmtree(work, ignore_errors=True)
    index = {"generated_by": "scripts/build_contract_examples.py", "contract": "docs/API_CONTRACT.md",
             "scripted": True, "note": "Scripted demo data: scores and model choices are scripted; preview numbers are "
                                       "illustrative placeholders.",
             "practice_script": script_files, "endpoints": files}
    (out / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    out = resolve(args.out)
    for sub in ("pictures", "patients", "practice", "review", "steering", "training"):
        shutil.rmtree(out / sub, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    index = build(out)
    print(f"wrote {len(index['endpoints'])} samples ({len(index['practice_script'])} practice steps) to {out}")


if __name__ == "__main__":
    main()
