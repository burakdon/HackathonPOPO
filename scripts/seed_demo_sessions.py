"""Fill the demo patient with three scripted sessions played through the real pipeline (contract §10.1).

Every answer is a macOS-voice clip (`say`) posted to the real endpoint of the app, in-process
(fastapi.testclient.TestClient(create_app())): real Whisper transcribes it, the real scorer scores it, the patient's
current real model (a DITTO adapter on Qwen3-1.7B, unless the patient uses the plan's rules) chooses each next step,
and every attempt, recording and decision is stored. Sessions are started with `engine.start(..., simulated=True)`,
so they are marked simulated. What is scripted is only what the "patient" says; transcripts, scores and model
decisions are whatever the pipeline really produced.

Each picture gets one of these behaviours (the script says the same kinds of things a person might):
  first      names it correctly at once
  helped     says a wrong real word, then names it after the hint
  stuck_sound  like stuck, but the first wrong answer is a near-miss
  stuck      three wrong real words in a row: the situation the demo note is about (a meaning hint failed twice),
             if the model chose a meaning hint first; the picture ends with "say it with me"
  sound      a near-miss with a different first sound, then correct
  unsure     presses "I'm not sure", then names it
  unclear    says "it's a <word>" (unclear, not counted), then names it
Before a clip is used, it is transcribed and scored once on its own (nothing recorded) to check that it really
scores as intended; if Whisper mishears it, another candidate word is tried. A transcript is never edited.

Idempotent: the IDs of the sessions this script created are kept in <data>/seed_demo_sessions.json. Running it
again does nothing; `--reset` deletes exactly those sessions (and their recordings) and seeds again. It refuses to
reset while training examples refer to their decisions.

Run:  uv run --extra llm --extra speech python -m scripts.seed_demo_sessions [--reset] [--data DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from core.config import resolve
from core.contract import Range, RetryThen, Rule, RuleWhen, matches_rule, rule_action
from scripts.scorer_report import NEAR, WRONG

PATIENT = "demo-ana"
# Whether the model starts with a meaning hint depends on the history in its prompt in ways we don't control (trial
# runs gave it on roughly one stuck picture in six), so most pictures are "stuck" ones: that is what makes
# "a meaning hint failed twice" happen at least four times.
BEHAVIOURS = (
    ["stuck", "stuck_sound", "stuck", "stuck_sound", "helped", "stuck", "stuck_sound", "first"],
    ["stuck_sound", "stuck", "unsure", "stuck", "stuck_sound", "stuck", "sound", "stuck"],
    ["unclear", "stuck", "stuck_sound", "stuck", "stuck_sound", "helped", "stuck", "stuck"],
)
STEPS = {"first": ["correct"], "helped": ["meaning", "correct"], "stuck": ["meaning", "meaning", "meaning"],
         "stuck_sound": ["sound", "meaning", "meaning"],
         "sound": ["sound", "correct"], "unsure": ["unsure", "correct"], "unclear": ["unclear", "correct"]}
FALLBACK_WRONG = ["table", "bird", "spoon", "ball", "book", "shoe", "lamp", "bread"]
DEMO_RULE = Rule(when=RuleWhen(hints_tried_include=["semantic"], failed_attempts_on_picture=Range(min=2)),
                 then=RetryThen(op="retry", hint="first_sound"), summary="demo rule (for the summary only)")


class Voice:
    def __init__(self, name: str, scorer, provider) -> None:
        self.name, self.scorer, self.provider = name, scorer, provider
        self.dir = Path(tempfile.mkdtemp(prefix="seed-clips-"))

    def clip(self, text: str) -> bytes:
        out = self.dir / f"{uuid.uuid4().hex}.wav"
        subprocess.run(["say", "-v", self.name, "-o", str(out), "--data-format=LEI16@16000", text], check=True)
        return out.read_bytes()

    def check(self, text: str, item, want: tuple[str, str | None]) -> bytes | None:
        """The clip for `text` if the real recognizer and scorer score it as `want`, else None."""
        data = self.clip(text)
        s = self.scorer(self.provider.transcribe(data, "audio/wav"), item.preferred_name, item.accepted_alternatives)
        return data if (s.outcome, s.error_type) == want else None


def candidates(kind: str, item) -> list[tuple[str, tuple[str, str | None]]]:
    name = item.preferred_name
    if kind == "correct":
        return [(name, ("correct", None)), (f"um, {name}", ("correct", None))]
    if kind == "meaning":
        words = [WRONG[name], *[w for w in FALLBACK_WRONG if w != WRONG[name]]]
        return [(w, ("incorrect", "meaning")) for w in words]
    if kind == "sound":
        return [(NEAR[name], ("incorrect", "sound")), (WRONG[name], ("incorrect", "meaning"))]
    if kind == "unclear":
        return [(f"it's {'an' if name[0] in 'aeiou' else 'a'} {name}", ("unclear", None))]
    raise ValueError(kind)


def seeded_ids(root: Path) -> list[str]:
    p = root / "seed_demo_sessions.json"
    return json.loads(p.read_text())["sessions"] if p.exists() else []


def reset(root: Path) -> None:
    ids = seeded_ids(root)
    for sid in ids:
        f = root / "sessions" / f"{sid}.json"
        decisions = {d["decision_id"] for p in json.loads(f.read_text())["pictures"] for d in p["decisions"]} if f.exists() else set()
        ex_file = root / "examples" / f"{PATIENT}.json"
        used = [e["example_id"] for e in json.loads(ex_file.read_text())["examples"] if e["decision_id"] in decisions] if ex_file.exists() else []
        if used:
            raise SystemExit(f"examples {used} refer to decisions of seeded session {sid}; remove them first, then reset")
    for sid in ids:
        (root / "sessions" / f"{sid}.json").unlink(missing_ok=True)
        for a in (root / "audio").glob(f"{sid}-a*"):
            a.unlink()
    (root / "seed_demo_sessions.json").unlink(missing_ok=True)
    print(f"reset: removed {len(ids)} seeded sessions")


def play(client, engine, voice: Voice, plan: list[str]) -> str:
    sid = engine.start(PATIENT, simulated=True).session_id
    for k, behaviour in enumerate(plan, start=1):
        cur = engine.sessions[sid].current
        if cur is None:
            break
        item = engine.bank.by_id[cur.exercise_id]
        steps = list(STEPS[behaviour])
        for spent in range(8):                                   # a picture ends within its tries; 8 is a safety stop
            pres_id = engine.sessions[sid].current.presentation_id
            kind = steps[min(spent, len(steps) - 1)]
            cid = uuid.uuid4().hex
            if kind == "unsure":
                r = client.post(f"/sessions/{sid}/no-answer", json={"presentation_id": pres_id, "client_attempt_id": cid})
            else:
                data = None
                for text, want in candidates(kind, item):
                    data = voice.check(text, item, want)
                    if data:
                        break
                if data is None:
                    print(f"  ! picture {k} ({item.preferred_name}): no clip scored as '{kind}'; using the last candidate as it is")
                    data = voice.clip(candidates(kind, item)[-1][0])
                r = client.post(f"/sessions/{sid}/recordings", files={"audio": ("a.wav", data, "audio/wav")},
                                data={"presentation_id": pres_id, "client_attempt_id": cid})
            assert r.status_code == 200, r.text
            if r.json()["next"]["kind"] in ("correct", "show_word", "not_assessed", "move_on"):
                break
    engine.end(sid)
    return sid


def summary(engine, sids: list[str]) -> None:
    total = tick = 0
    print(f"\n{'session':<14}{'pictures':>9}{'indep':>7}{'assist':>7}{'unsucc':>7}{'n/a':>5}{'decisions':>10}{'model':>34}{'demo rule':>10}{'tickable':>9}")
    for sid in sids:
        rev = engine.review(sid)
        ds = [d for p in rev.pictures for d in p.decisions]
        m = [d for d in ds if matches_rule(DEMO_RULE, d.context)]
        t = [d for d in m if rule_action(DEMO_RULE, d.context) in d.options]
        total, tick = total + len(m), tick + len(t)
        c = rev.summary.counts
        src = {d.source.split(":")[0] if not d.source.startswith("ditto") else "ditto" for d in ds}
        fb = sum(d.source.startswith("fallback") for d in ds)
        print(f"{sid:<14}{rev.summary.pictures:>9}{c.independent:>7}{c.assisted:>7}{c.unsuccessful:>7}{c.not_assessed:>5}"
              f"{len(ds):>10}{rev.summary.model_version:>34}{len(m):>10}{len(t):>9}   sources={sorted(src)} fallbacks={fb}")
        errs = rev.summary.errors
        print(f"{'':<14}errors: sound {errs.sound}, wrong word {errs.wrong_word}, no answer {errs.no_answer}, unclassified {errs.unclassified}; simulated={rev.summary.simulated}")
    print(f"\ndemo rule (meaning hint shown, 2+ failed tries -> first-sound hint): {total} matching decisions, {tick} tickable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true", help="delete the sessions this script created, then seed again")
    ap.add_argument("--data", default=None, help="data directory (default $CUEAGAIN_DATA or runs/cueagain)")
    ap.add_argument("--voice", default="Samantha")
    args = ap.parse_args()
    root = resolve(args.data or os.environ.get("CUEAGAIN_DATA", "runs/cueagain"))
    if args.reset:
        reset(root)
    if seeded_ids(root):
        from service.app import create_app
        engine = create_app(cueagain_root=root).state.engine
        print("already seeded (use --reset to start over)")
        summary(engine, seeded_ids(root))
        return

    from fastapi.testclient import TestClient

    from service.app import create_app
    from speech.scoring import score
    from speech.whisper_local import WhisperLocalProvider
    provider = WhisperLocalProvider()
    app = create_app(speech_provider=provider, cueagain_root=root)
    engine, client = app.state.engine, TestClient(app)
    version = engine.patient(PATIENT).active_model.version
    print(f"patient {PATIENT}, model {version}")
    if version != "rules":
        print("loading the model ...")
        app.state.registry.policy(engine.adapter_version(version))
    voice = Voice(args.voice, score, provider)
    sids = []
    for n, plan in enumerate(BEHAVIOURS, start=1):
        sids.append(play(client, engine, voice, plan))
        (root / "seed_demo_sessions.json").write_text(json.dumps({"patient": PATIENT, "sessions": sids}, indent=1))
        print(f"session {n}: {sids[-1]}")
    summary(engine, sids)


if __name__ == "__main__":
    main()
