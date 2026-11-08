"""Can a few clinician examples steer one patient's adapter? (per-patient DITTO, contract section 8)

Situation: 2 failed tries on a picture, the meaning (semantic) hint already shown. The clinician
wants {"op":"retry","hint":"first_sound"}. Starting from the shared adapter, a per-patient adapter is
trained (training/ditto.py with init_adapter) on only 3, 5 or 8 such examples, for 2 seeds, and
compared with the shared adapter on:
  (a) its own training examples;
  (b) 20 new situations of the same kind, on pictures not used in any training example, with
      different earlier attempts;
  (c) the 55 general held-out test situations (v3 test split): how often its choice equals the
      shared adapter's.
All options follow the contract (core.practice.contract_options: a hint already shown is never
offered again). Greedy decoding, as served.

    uv run --extra llm python -m scripts.patient_steering_experiment                    # greedy generation
    uv run --extra llm python -m scripts.patient_steering_experiment --decoding score   # option scoring

Trained runs are reused, so the second command only re-measures the same adapters.

With anchors (training/patient_adapter.py), from the shared adapter v4_score-s0:
    uv run --extra llm python -m scripts.patient_steering_experiment --decoding score \
        --shared models/ditto_llm/v4_score-s0 --anchors 60 --tag anchored
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import yaml

from core.config import load_clinical, resolve
from core.practice import DecisionContext, contract_options, format_action, load_items, load_practice, messages, parse_action
from training.build_demonstrations import Spec, build_context

SHARED = "models/ditto_llm/20261107-214025"
TARGET = format_action(parse_action('{"op":"retry","hint":"first_sound"}'))
SIZES, SEEDS = (3, 5, 8), (0, 1)
TRAIN_WORDS = ["cat", "banana", "octopus", "chair", "house", "giraffe", "fish", "hammer"]
OUT = "results/patient_steering"


def semantic_failed(word: str, k: int, seed: int) -> dict:
    """First try on `word` a wrong word (so the meaning hint was shown), second try with the meaning
    hint also incorrect (any error type), 1-4 earlier attempts on other pictures, before 20 min."""
    rng = np.random.default_rng([seed, k])
    err = ["sound", "meaning", "no_response", "unclassified"][int(rng.integers(4))]
    spec = Spec(f"P{k:03d}", "train", word, (err, 12.0 if err == "no_response" else round(float(rng.uniform(4, 10)), 1)),
                [("none", "meaning", round(float(rng.uniform(4, 9)), 1))],
                minutes=float(round(rng.uniform(2, 19), 0)), n_earlier=int(rng.integers(1, 5)))
    ctx = build_context(spec, load_items(), rng)
    assert ctx.current.hints_shown == ["semantic"] and ctx.current.attempts == 2
    return ctx


def row(ctx: DecisionContext, eid: str) -> dict:
    c = load_clinical()
    opts = [format_action(o) for o in contract_options(ctx, c.approved_cues, c.max_tier_step, 3)]
    assert TARGET in opts
    return {"example_id": eid, "split": "train", "family_id": eid, "context": ctx.model_dump(), "options": opts,
            "preferred": TARGET, "accepted": [TARGET], "rule": "patient: semantic failed -> first_sound",
            "note": "2 failed tries, meaning hint already shown: give the first sound", "scope": "patient",
            "provenance": {"source": "generated clinician example (experiment)", "review_status": "pending human review"}}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoding", choices=["generate", "score"], default="generate")
    ap.add_argument("--shared", default=SHARED)
    ap.add_argument("--anchors", type=int, default=0, help="number of anchor situations (0 = none)")
    ap.add_argument("--tag", default="", help="output subfolder and run-id prefix")
    args = ap.parse_args()
    decoding, shared_path = args.decoding, args.shared
    out = resolve(OUT) / args.tag if args.tag else resolve(OUT)
    out.mkdir(parents=True, exist_ok=True)
    words = [i.preferred_name for i in load_items().items]
    eval_words = [w for w in words if w not in TRAIN_WORDS]
    train_rows = [row(semantic_failed(w, k, 100), f"pt-train-{k}") for k, w in enumerate(TRAIN_WORDS)]
    new_rows = [row(semantic_failed(eval_words[k % len(eval_words)], k, 200), f"pt-new-{k}") for k in range(20)]
    c = load_clinical()
    general = []
    for r in (json.loads(line) for line in resolve("data/demonstrations/demonstrator_a_v3.jsonl").read_text().splitlines()):
        if r["split"] == "test":
            ctx = DecisionContext(**r["context"])
            general.append({"example_id": r["example_id"], "context": r["context"],
                            "options": [format_action(o) for o in contract_options(ctx, c.approved_cues, c.max_tier_step, 3)]})
    (out / "situations.json").write_text(json.dumps({"train": train_rows, "new": new_rows}, indent=1))

    from agents.llm_runtime import LLMRuntime
    from training.patient_adapter import anchor_rows, mix
    pcfg = load_practice()
    rt = LLMRuntime(pcfg.llm.model_copy(update={"decoding": decoding}))
    rt.ensure_adapter("shared", resolve(shared_path))
    rule = {"when": {"hints_tried_include": ["semantic"], "failed_attempts_on_picture": {"min": 2}},
            "then": {"op": "retry", "hint": "first_sound"}}
    anchors = []
    if args.anchors:
        train_ctx = [DecisionContext(**r["context"]) for r in
                     (json.loads(line) for line in resolve("data/demonstrations/demonstrator_a_v4.jsonl").read_text().splitlines())
                     if r["split"] == "train"]
        anchors = anchor_rows(rt, "shared", train_ctx, [rule], pcfg.history_in_prompt, limit=args.anchors,
                              fmt=pcfg.llm.prompt_format)
        print(f"{len(anchors)} anchors from the shared adapter on training-split situations", flush=True)
    # Free the model before training: a second copy in the trainer process would swap on a 24 GB laptop.
    import gc
    import torch
    rt.model, rt.adapters = None, {}
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    base_cfg = yaml.safe_load(resolve("configs/ditto_llm.yaml").read_text())
    if args.anchors:      # option-scoring sampling and a fixed SFT length (no early stop on a tiny batch)
        p = yaml.safe_load(resolve(base_cfg["practice_config"]).read_text())
        p["llm"]["decoding"] = decoding
        (out / "practice.yaml").write_text(yaml.safe_dump(p, sort_keys=False))
        base_cfg["practice_config"] = str((out / "practice.yaml").relative_to(resolve(".")))
        base_cfg["sft"] = {**base_cfg["sft"], "max_steps": 40, "stop_loss": 0.0}
        # the gentle DITTO setting of the shared adapter (round 2, g2)
        base_cfg["ditto"].update({"learning_rate": 2.0e-5, "beta": 0.2, "rpo_alpha": 0.5, "steps_per_round": 30})
    prefix = f"patient-{args.tag}" if args.tag else "patient-steer"
    runs = {}
    for n in SIZES:
        demo = out / f"examples_{n}.jsonl"
        demo.write_text("".join(json.dumps(r) + "\n" for r in mix(train_rows[:n], anchors)))
        for seed in SEEDS:
            rid = f"{prefix}-n{n}-s{seed}"
            cfg = {**base_cfg, "demonstrations": str(demo.relative_to(resolve("."))), "init_adapter": shared_path,
                   "seed": seed}
            cpath = out / f"config_{rid}.yaml"
            cpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
            man = resolve("runs/ditto") / rid / "manifest.json"
            if not (man.exists() and json.loads(man.read_text())["status"] == "completed"):
                t0 = time.time()
                subprocess.run([sys.executable, "-m", "training.ditto", "--config", str(cpath), "--run-id", rid], check=True)
                print(f"trained {rid} in {time.time() - t0:.0f} s", flush=True)
            runs[rid] = {"n": n, "seed": seed, "manifest": json.loads(man.read_text())}

    rt.ensure_adapter("shared", resolve(shared_path))           # reloads the base model
    for rid, info in runs.items():
        rt.ensure_adapter(rid.replace("-", "_"), resolve("runs/ditto") / rid / info["manifest"]["final_adapter"])

    def choose(adapter: str, r: dict) -> str:
        ctx = DecisionContext(**r["context"])
        msgs = messages(ctx, [parse_action(o) for o in r["options"]], pcfg.history_in_prompt, fmt=pcfg.llm.prompt_format)
        if decoding == "score":
            lps, _ = rt.score(msgs, r["options"], adapter)
            return r["options"][int(np.argmax(lps))]
        text, _ = rt.generate(msgs, adapter)
        a = parse_action(text)
        return a.key() if a is not None and a.key() in r["options"] else f"INVALID:{text[:40]}"

    shared = {r["example_id"]: choose("shared", r) for r in train_rows + new_rows + general}
    results = []
    for rid, info in runs.items():
        name, n = rid.replace("-", "_"), info["n"]
        mine = {r["example_id"]: choose(name, r) for r in train_rows[:n] + new_rows + general}
        results.append({
            "run": rid, "examples": n, "seed": info["seed"], "train_seconds": info["manifest"]["seconds"],
            "sft_steps": info["manifest"]["sft"]["steps"],
            "a_train_new": sum(mine[r["example_id"]] == TARGET for r in train_rows[:n]),
            "a_train_shared": sum(shared[r["example_id"]] == TARGET for r in train_rows[:n]),
            "b_new_new": sum(mine[r["example_id"]] == TARGET for r in new_rows),
            "b_new_shared": sum(shared[r["example_id"]] == TARGET for r in new_rows),
            "c_same": sum(mine[r["example_id"]] == shared[r["example_id"]] for r in general),
            "c_total": len(general),
            "invalid": sum(v.startswith("INVALID") for v in mine.values()),
            "b_choices": dict(sorted({v: list(mine[r["example_id"]] for r in new_rows).count(v)
                                      for v in set(mine[r["example_id"]] for r in new_rows)}.items())),
        })
        print(results[-1], flush=True)
    shared_new = {v: [shared[r["example_id"]] for r in new_rows].count(v) for v in set(shared[r["example_id"]] for r in new_rows)}
    suffix = "" if decoding == "generate" else "_score"
    (out / f"results{suffix}.json").write_text(json.dumps({"target": TARGET, "shared_adapter": shared_path, "anchors": len(anchors), "results": results,
                                                  "shared_choices_on_new": shared_new,
                                                  "shared_invalid": sum(v.startswith("INVALID") for v in shared.values())},
                                                 indent=1))
    lines = [f"# Per-patient steering ({decoding}): 2 failed tries, meaning hint shown -> first-sound hint", "",
             f"Shared adapter `{shared_path}`, {len(anchors)} anchors; per-patient adapters trained from it (training/ditto.py, init_adapter) on "
             "only the clinician examples. Options follow the contract (a hint already shown is never offered "
             f"again). Decoding: {decoding}. Generated by `scripts/patient_steering_experiment.py`.", "",
             "| examples | seed | (a) chooses first sound on its training examples: new / shared | "
             "(b) on 20 new situations: new / shared | (c) general test situations unchanged | invalid | train s |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['examples']} | {r['seed']} | {r['a_train_new']}/{r['examples']} / {r['a_train_shared']}/{r['examples']} | "
              f"{r['b_new_new']}/20 / {r['b_new_shared']}/20 | {r['c_same']}/{r['c_total']} | {r['invalid']} | "
              f"{r['train_seconds']:.0f} |" for r in results]
    lines += ["", f"Shared adapter's choices on the 20 new situations: {shared_new}"]
    (out / f"README{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
