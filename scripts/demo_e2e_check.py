"""End-to-end rehearsal of the DITTO demo against a running server (real HTTP, real Whisper, real models).

    CUEAGAIN_DATA=runs/cueagain_e2e uv run --extra llm --extra speech uvicorn service.app:app --port 8000
    uv run --extra llm --extra speech python -m scripts.demo_e2e_check [--base http://localhost:8000]

Use a copy of the demo data (the steps below create examples, a training run and switch the active model).
Needs runs/<data>/prepared_demo.json from scripts/prepare_demo_run.py. Steps, each timed:
  1 practice     a session where the patient gives two wrong words on the first picture: the shared adapter
                 chooses the hint after the failed meaning hint
  2 steering     note -> rule (entered by hand) -> matches -> tick the prepared decisions -> examples
  3 training     start a training run (real training starts)
  4 prepared     use the prepared run's result (same example IDs)
  5 activate     make the patient's adapter active
  6 practice     a new session, same answers: the patient's adapter chooses the hint
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import httpx

PATIENT = "demo-ana"
WRONG = ["table", "spoon"]


def clip(text: str, d: Path) -> bytes:
    out = d / f"{uuid.uuid4().hex}.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(out), "--data-format=LEI16@16000", text], check=True)
    return out.read_bytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--data", default="runs/cueagain_e2e")
    args = ap.parse_args()
    prep = json.loads((Path(args.data) / "prepared_demo.json").read_text())
    c = httpx.Client(base_url=args.base, timeout=600)
    tmp = Path(tempfile.mkdtemp())
    log = []

    def step(name, method, url, **kw):
        t0 = time.perf_counter()
        r = c.request(method, url, **kw)
        ms = (time.perf_counter() - t0) * 1000
        assert r.status_code < 300, f"{name}: {r.status_code} {r.text[:300]}"
        log.append((name, r.status_code, ms))
        return r.json(), ms

    def practice(label):
        s, _ = step(f"{label}: start session", "POST", f"/patients/{PATIENT}/sessions")
        sid, pres = s["session_id"], s["picture"]
        print(f"[{label}] session {sid}, model {s.get('model_version')}, picture '{pres['word']}'")
        hints = []
        for k, word in enumerate(WRONG, 1):
            res, ms = step(f"{label}: wrong answer {k} ('{word}')", "POST", f"/sessions/{sid}/recordings",
                           files={"audio": ("a.wav", clip(word, tmp), "audio/wav")},
                           data={"presentation_id": pres["presentation_id"], "client_attempt_id": uuid.uuid4().hex})
            nxt = res["next"]
            hints.append((nxt.get("hint") or {}).get("type"))
            print(f"[{label}]   heard '{res['transcript']}' -> {res['score']['outcome']}/{res['score']['error_type']}; "
                  f"next {nxt['kind']} {hints[-1]} (decision {nxt.get('decision_id')}) in {ms:.0f} ms")
        step(f"{label}: end session", "POST", f"/sessions/{sid}/end")
        return hints

    t_all = time.perf_counter()
    h, _ = step("health", "GET", "/health")
    before = practice("before")

    note, _ = step("note", "POST", f"/patients/{PATIENT}/notes", json={"text": prep["note"]})
    print(f"note {note['note_id']}: drafted rule {note['rule']} ({note.get('problem')})")
    step("confirm rule", "PUT", f"/notes/{note['note_id']}/rule", json={"rule": prep["rule"]})
    matches, ms = step("matches", "GET", f"/notes/{note['note_id']}/matches")
    print(f"{len(matches)} matches, {sum(m['allowed'] for m in matches)} tickable ({ms:.0f} ms)")
    ex, _ = step("tick examples", "POST", f"/notes/{note['note_id']}/examples",
                 json={"decision_ids": prep["tick_decision_ids"]})
    print(f"examples {[e['example_id'] for e in ex]} (prepared run used {prep['example_ids']})")

    run, _ = step("start training", "POST", f"/patients/{PATIENT}/training-runs")
    rid = run["run_id"]
    time.sleep(3)
    prog, _ = step("training progress", "GET", f"/training-runs/{rid}")
    print(f"live run {rid}: {prog['status']} {prog['progress']}")
    used, _ = step("use prepared", "POST", f"/training-runs/{rid}/use-prepared",
                   json={"prepared_run_id": prep["prepared_run_id"]})
    print(f"use-prepared: status {used['status']}, prepared {used['prepared']} ({used['prepared_run_id']}), "
          f"preview {json.dumps(used['preview'])}")
    models, _ = step("models", "GET", f"/patients/{PATIENT}/models")
    print("models:", [(m["id"], m["active"]) for m in models])
    pat, _ = step("activate", "POST", f"/patients/{PATIENT}/active-model", json={"version": f"ditto:{rid}"})
    print("active model:", pat["active_model"])
    after = practice("after")

    print(f"\nhint after the failed meaning hint: before {before[-1]}, after {after[-1]} "
          f"(first hint: before {before[0]}, after {after[0]})")
    print(f"\n{'step':42s} {'status':>6s} {'ms':>8s}")
    for name, code, ms in log:
        print(f"{name:42s} {code:>6d} {ms:8.0f}")
    print(f"total {time.perf_counter() - t_all:.0f} s")
    if after[-1] != prep["rule"]["then"]["hint"]:
        raise SystemExit("the patient's adapter did not choose the therapist's hint")


if __name__ == "__main__":
    main()
