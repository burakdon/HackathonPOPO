"""Per-patient DITTO (API contract section 8): a patient's adapter trained from the shared adapter on that
patient's clinician examples only, plus anchors.

    uv run --extra llm python -m training.patient_ditto --run-id demo-ana-r1 --examples <rows.jsonl>

Called by service.training.SubprocessTrainer with the same run id and examples file it used to pass to
training.ditto; writes the same runs/ditto/<run_id>/manifest.json, so progress, preview and activation are
unchanged.

What it trains (configs/patient_ditto.yaml):
  start     the shared adapter (init_adapter, models/ditto_llm/sft200-g2-s0), which is also the starting point
            of the SFT stage and hence of the DPO reference;
  rows      the patient's examples (repeated so they are about a third of the rows) plus anchors: the shared
            adapter's own choice, computed with the loaded model by option scoring, on training-split situations
            of other kinds than the patient's examples, with the app's option rule (no hint twice, hints only
            while tries are left). Without anchors a few examples made the adapter apply the clinician's choice
            everywhere (results/patient_steering/README_score.md); live DITTO on the PPO policy uses the same
            idea (service/live_ditto.py);
  training  brief SFT, then gentle DITTO rounds (low learning rate, higher beta, RPO term), option scoring for
            sampling, prompt format v2;
  selection the stage (SFT or a round) with the highest agreement on the training rows (the patient's examples
            and the anchors), ties to the earlier stage. A patient run has no held-out split, so this only checks
            that the update took without breaking the anchored behaviour; the preview measures the rest.
"""
from __future__ import annotations

import argparse
import json
import math
import random

from core.config import load_clinical, resolve
from core.practice import DecisionContext, contract_options, format_action, messages, parse_action

CONFIG = "configs/patient_ditto.yaml"


def failed_on_picture(ctx: DecisionContext) -> int:
    return sum(1 for a in ctx.history if a.exercise_id == ctx.current.exercise_id and not a.correct)


def app_options(ctx: DecisionContext, max_attempts: int = 3) -> list[str]:
    """The live app's options (service.engine.decision_options): contract options, hints only while tries are left."""
    c = load_clinical()
    opts = contract_options(ctx, c.approved_cues, c.max_tier_step, max_attempts)
    retries = [o for o in opts if o.operation == "retry_current"]
    return [format_action(o) for o in (retries or opts)]


def same_kind(ctx: DecisionContext, example: DecisionContext) -> bool:
    """Whether `ctx` is a situation of the kind a clinician example is about, which anchors must not pin to the
    shared adapter's choice: same outcome of the latest attempt, at least the example's hints already shown (no
    hints if the example had none), and at least as many failed tries on the picture."""
    shown, ex_shown = set(ctx.current.hints_shown), set(example.current.hints_shown)
    return (ctx.latest.correct == example.latest.correct and ex_shown <= shown and bool(shown) == bool(ex_shown)
            and failed_on_picture(ctx) >= failed_on_picture(example))


def mix(patient_rows: list[dict], anchors: list[dict], share: float) -> tuple[list[dict], int]:
    """Patient rows repeated (distinct ids) to make up about `share` of all rows; returns (rows, repeats)."""
    if not anchors:
        return list(patient_rows), 1
    reps = max(1, math.ceil(share * len(anchors) / ((1 - share) * len(patient_rows))))
    return [{**r, "example_id": f"{r['example_id']}-r{j}"} for j in range(reps) for r in patient_rows] + anchors, reps


def make_prepare(cfg: dict):
    a = cfg["anchors"]

    def prepare(model, enc, fmt, rows, history_in_prompt):
        import torch

        from training.ditto import answer_logps, load_examples
        examples = [DecisionContext(**r["context"]) for r in rows]
        pool = [DecisionContext(**r["context"]) for r in load_examples(a["source"], "train")]
        random.Random(cfg["seed"]).shuffle(pool)
        anchors, skipped = [], 0
        model.eval()
        with torch.inference_mode():
            for ctx in pool:
                if len(anchors) >= a["n"]:
                    break
                if any(same_kind(ctx, e) for e in examples):
                    skipped += 1
                    continue
                opts = app_options(ctx)
                prompt = enc.prompt_ids(messages(ctx, [parse_action(o) for o in opts], history_in_prompt, fmt=fmt))
                lp = answer_logps(model, enc, [(prompt, enc.answer_ids(o)) for o in opts])
                choice = opts[int(lp.argmax())]
                k = len(anchors)
                anchors.append({"example_id": f"anchor-{k:03d}", "split": "train", "family_id": f"anchor-{k:03d}",
                                "context": ctx.model_dump(), "options": opts, "preferred": choice, "accepted": [choice],
                                "rule": "anchor", "note": "shared adapter's own choice", "scope": "patient",
                                "provenance": {"source": "anchor: shared adapter choice", "pool": a["source"]}})
        out, reps = mix(rows, anchors, a["patient_share"])
        print(f"{len(rows)} patient examples (x{reps}) + {len(anchors)} anchors "
              f"({skipped} pool situations skipped as the same kind as an example)", flush=True)
        return out, {"patient_examples": [r["example_id"] for r in rows], "repeats": reps, "anchors": len(anchors),
                     "skipped_same_kind": skipped, "anchor_source": a["source"], "init_adapter": cfg["init_adapter"]}

    return prepare


def main() -> None:
    import yaml

    from training.ditto import run
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--examples", required=True, help="the patient's examples (demonstration-format JSONL)")
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text())
    cfg["demonstrations"] = str(resolve(args.examples))
    tmp = resolve("runs/ditto") / f"{args.run_id}.config.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    run(str(tmp), args.run_id, smoke=False, extra_demos=None, prepare_rows=make_prepare(cfg))


if __name__ == "__main__":
    main()
