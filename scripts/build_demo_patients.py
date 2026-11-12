"""Precompute the six patients the hosted console offers, so the page needs no live compute.

  uv run python scripts/build_demo_patients.py            (about 15-25 minutes on a laptop CPU)

For each patient it calls the console server's own endpoint functions (neurocue/demo/server.py), so
every number is what the live server would have produced for that patient and seed:

  - 100 picture-naming responses in order (Dell model + word realizer + PSST framing)
  - the true lesion weights
  - two four-session courses on identical copies (cueing hierarchy, MaskablePPO trained on retention)
  - a DITTO result for every option the "teach it" form offers (situation x word rule x cue),
    each taught from the original policy with six demonstrations

The posterior over lesion weights is cheap, so the page computes it itself from the fitter's grid
table (written once as table.json). Output: neurocue/demo/static/patients/ (index.json, table.json,
one JSON per patient, and the pictures the responses use).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocue.demo import server as S  # noqa: E402
from patientmodel.dell_sp import NAMING_CATEGORIES  # noqa: E402

OUT = ROOT / "neurocue" / "demo" / "static" / "patients"
N_NAMING = 100
CUES = ["phonemic_initial", "semantic", "whole_word", ""]
RULES = ["due", "repeat"]
CTX_ORDER = ["nonword", "semantic", "formal", "mixed", "unrelated", "correct"]

# Six patients spread over severity (WAB AQ) and lesion type. Names and faces are invented.
PATIENTS = [
    dict(id="marisol", name="Marisol", age=23, aq=62, lesion="balanced", seed=1,
         blurb="Moderate, both connections weakened. Many nonwords, some meaning errors. A graduate student and distance runner: aphasia can follow a stroke at any age.",
         avatar=dict(skin="#c48a62", hair="#2a1a12", hairStyle="long", top="#1c6560", eyes="#4a2c18", earrings=True, slim=True)),
    dict(id="harold", name="Harold", age=74, aq=34, lesion="phonological", seed=2,
         blurb="Severe, phonological. Finds the word, loses its sounds: about half his answers are nonwords. A former machinist.",
         avatar=dict(skin="#e8bfa0", hair="#cfcac4", hairStyle="bald", top="#5b6b7a", eyes="#5d7a8c", glasses=True, beard="#d8d3cc")),
    dict(id="aisha", name="Aisha", age=45, aq=86, lesion="semantic", seed=3,
         blurb="Mild, semantic. Names most pictures; when she errs, a word near in meaning or sound wins. Back at work part-time as an accountant.",
         avatar=dict(skin="#7a4a2e", hair="#1a1210", hairStyle="wrap", wrap="#a8475a", top="#2f4f8a", eyes="#2a170c", earrings=True)),
    dict(id="kenji", name="Kenji", age=67, aq=48, lesion="semantic", seed=4,
         blurb="Severe-moderate, semantic weight the weaker one. A spread of every error type. Keeps a vegetable garden.",
         avatar=dict(skin="#e0b48c", hair="#3a3a3a", hairStyle="short", top="#6e5a3c", eyes="#2a1d14")),
    dict(id="ruth", name="Ruth", age=79, aq=72, lesion="phonological", seed=5,
         blurb="Moderate-mild, phonological. Nearly every error is a nonword. Sang in a church choir for forty years.",
         avatar=dict(skin="#f0cdb4", hair="#e6e2dc", hairStyle="bun", top="#8a4f7d", eyes="#4f6f5a", glasses=True)),
    dict(id="darnell", name="Darnell", age=38, aq=28, lesion="balanced", seed=6,
         blurb="Severe, the lowest AQ of the six; both connections weakened. Over half his answers are nonwords; about one in four is right. A paramedic and father of two.",
         avatar=dict(skin="#5c3a26", hair="#151010", hairStyle="curly", top="#b5552f", eyes="#24150c", beard="#1d1512")),
]


def rel_image(item_id: str) -> str:
    return f"patients/img/{item_id}.png"


def build_patient(spec: dict, used_images: set[str]) -> dict:
    t0 = time.time()
    p = S.new_patient(S.NewPatient(aq=spec["aq"], lesion=spec["lesion"], seed=spec["seed"]))
    pid = p["id"]
    responses = []
    for _ in range(N_NAMING):
        r = S.name_picture(pid)["response"]
        used_images.add(r["item_id"]); r["image"] = rel_image(r["item_id"])
        responses.append(r)
    truth = S.truth(pid)
    courses = S.course(pid, S.CourseReq(policies=["spaced_hierarchy", "ppo_retention"], teach="retention"))
    trials = courses["ppo_retention"]["trials"]
    ctxs = [c for c in CTX_ORDER if any(t["prev"] == c for t in trials)] + ["any_error"]
    teach = {}
    d = S.PATIENTS[pid]
    for ctx in ctxs:
        for rule in RULES:
            for cue in CUES:
                d.demos = []; d.tuned = None
                key = f"{ctx}|{rule}|{cue}"
                try:
                    teach[key] = S.override(pid, S.OverrideReq(trial=-1, context=ctx, cue=cue or None, rule=rule,
                                                              n_demos=6, include_course=False))
                    teach[key].pop("course", None)
                except HTTPException as e:
                    teach[key] = {"error": str(e.detail)}
    print(f"  {spec['name']}: {N_NAMING} responses, {len(teach)} teach results, {time.time() - t0:.0f}s", flush=True)
    meta = {k: spec[k] for k in ("id", "name", "age", "aq", "lesion", "seed", "blurb", "avatar")}
    meta["expected_accuracy"] = p["expected_accuracy"]
    meta["observed_accuracy"] = float(np.mean([r["correct"] for r in responses]))
    return {"patient": meta, "responses": responses, "truth": truth, "courses": courses, "teach_contexts": ctxs, "teach": teach}


def main() -> None:
    t0 = time.time()
    S.Shared.load()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "img").mkdir(exist_ok=True)
    t = S.Shared.env.table
    (OUT / "table.json").write_text(json.dumps({
        "categories": list(NAMING_CATEGORIES), "s_values": t.s_values.round(5).tolist(), "p_values": t.p_values.round(5).tolist(),
        "logp": np.log(np.clip(t.probs, 1e-4, None)).round(4).tolist()}))
    used: set[str] = set(); index = []
    for spec in PATIENTS:
        data = build_patient(spec, used)
        (OUT / f"{spec['id']}.json").write_text(json.dumps(data, separators=(",", ":")))
        index.append(data["patient"])
    pics = {it.item_id: it.image for it in S.Shared.pictures}
    for item_id in sorted(used):
        shutil.copyfile(pics[item_id], OUT / "img" / f"{item_id}.png")
    (OUT / "index.json").write_text(json.dumps({"patients": index, "n_naming": N_NAMING,
                                                 "built": time.strftime("%Y-%m-%d")}, indent=1))
    print(f"wrote {len(index)} patients and {len(used)} pictures to {OUT.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
