"""Prepare the "trained earlier" per-patient run for the demo (contract §10.5), through the real endpoints.

The live demo writes the note, confirms the rule, ticks DECISIONS and presses Train; the run then takes this
prepared result (POST /training-runs/{live}/use-prepared), which the backend accepts only when both runs were
trained on the same example IDs. So the prepared run is made in a copy of the data folder with exactly those
steps (the same note, rule and ticks give the same example IDs, x-0001...), trained for real
(training.patient_ditto from the shared adapter) with the real preview, and only its run record is copied back.
The live data keeps no examples, so the demo creates them itself.

    uv run --extra llm --extra speech python -m scripts.prepare_demo_run      # after scripts.seed_demo_sessions

Writes runs/cueagain/training/<run_id>.json (+ .examples.jsonl), runs/ditto/<run_id>/ (the adapter) and
runs/cueagain/prepared_demo.json (what the demo must click). Needs the seeded sessions of demo-ana.
"""
from __future__ import annotations

import json
import shutil
import time

from core.config import resolve

PATIENT = "demo-ana"
NOTE = "When meaning hints have repeatedly failed, try a first-letter hint before moving to another picture."
RULE = {"when": {"hints_tried_include": ["semantic"], "failed_attempts_on_picture": {"min": 2}},
        "then": {"op": "retry", "hint": "first_letter"},
        "summary": "After 2 or more failed tries on a picture where a meaning hint was already given -> give the "
                   "first-letter hint."}
N_TICK = 5


def main() -> None:
    from fastapi.testclient import TestClient

    from service.app import create_app
    src, prep = resolve("runs/cueagain"), resolve("runs/cueagain_prepared")
    if (src / "prepared_demo.json").exists():
        raise SystemExit(f"{src / 'prepared_demo.json'} exists: the demo run is already prepared")
    if prep.exists():
        shutil.rmtree(prep)
    shutil.copytree(src, prep)
    c = TestClient(create_app(cueagain_root=prep))

    note = c.post(f"/patients/{PATIENT}/notes", json={"text": NOTE}).json()
    nid = note["note_id"]
    print(f"note {nid}: drafted rule = {note['rule']} ({note.get('problem')})")
    r = c.put(f"/notes/{nid}/rule", json={"rule": RULE})
    assert r.status_code == 200, r.text
    matches = c.get(f"/notes/{nid}/matches").json()
    allowed = [m for m in matches if m["allowed"]]
    print(f"{len(matches)} matches, {len(allowed)} tickable; model chose: "
          f"{sorted({json.dumps(m['model_chose']) for m in allowed})}")
    # one decision per picture word where possible, spread over the sessions
    picked, words = [], set()
    for m in allowed:
        if m["word"] not in words and len(picked) < N_TICK:
            picked.append(m["decision_id"])
            words.add(m["word"])
    r = c.post(f"/notes/{nid}/examples", json={"decision_ids": picked})
    assert r.status_code == 201, r.text
    example_ids = [e["example_id"] for e in r.json()]
    print(f"ticked {picked} -> examples {example_ids}")

    t0 = time.time()
    run = c.post(f"/patients/{PATIENT}/training-runs").json()
    rid = run["run_id"]
    last = None
    while True:
        run = c.get(f"/training-runs/{rid}").json()
        state = (run["status"], json.dumps(run["progress"]))
        if state != last:
            print(f"  {time.time() - t0:5.0f} s  {run['status']:9s} {run['progress']}", flush=True)
            last = state
        if run["status"] in ("completed", "failed"):
            break
        time.sleep(5)
    if run["status"] != "completed":
        raise SystemExit(f"training failed: {run['error']}")
    print(f"run {rid} completed in {time.time() - t0:.0f} s; preview: {json.dumps(run['preview'], indent=1)}")

    for f in (f"{rid}.json", f"{rid}.examples.jsonl"):
        (src / "training").mkdir(parents=True, exist_ok=True)
        shutil.copy(prep / "training" / f, src / "training" / f)
    unused = [m["decision_id"] for m in allowed if m["decision_id"] not in picked]
    (src / "prepared_demo.json").write_text(json.dumps({
        "patient_id": PATIENT, "prepared_run_id": rid, "note": NOTE, "rule": RULE, "tick_decision_ids": picked,
        "example_ids": example_ids, "other_matching_decisions": unused, "preview": run["preview"],
        "training_seconds": round(time.time() - t0)}, indent=1))
    print(f"prepared: {src / 'training' / (rid + '.json')}, runs/ditto/{rid}/, {src / 'prepared_demo.json'}")


if __name__ == "__main__":
    main()
