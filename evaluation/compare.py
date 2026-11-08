"""Compare decision models on reserved situations the training never saw.

    uv run --extra llm python -m evaluation.compare --run <run_id>               # test split
    uv run --extra llm python -m evaluation.compare --run <run_id> --split dev

Every condition gets the same situations, the same rendered context and the same valid options:
  base       the unadapted model with the normal task instruction
  prompted   the same model with the 20 training demonstrations in its prompt (never test ones)
  sft        the run's supervised-only adapter
  ditto@k    the run's DITTO adapter after round k (the last round is the served adapter)
  bandit     the Beta-Bernoulli bandit baseline, updated on the situation's earlier attempts

Reported with counts and denominators: valid structured output, constraint violations (parsed but
not a valid option) before fallback, fallback rate, agreement with the demonstrator's accepted
actions on valid outputs, end-to-end agreement after fallback, and decision latency. Ten test
situations are a small exploratory check, not a statistical or clinical validation.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone

from agents.llm_policy import BanditDecisionPolicy
from agents.registry import ModelRegistry
from core.config import load_beta_bandit, load_clinical, resolve
from core.practice import DecisionContext, load_practice, parse_action, to_state
from core.schema import Outcome
from training.ditto import examples_hash, load_examples


def bandit_for(ctx: DecisionContext, bcfg, clinical, seed: int) -> BanditDecisionPolicy:
    """A fresh bandit that has observed this situation's earlier attempts (fair: same history)."""
    b = BanditDecisionPolicy(bcfg, clinical, seed=seed)
    for k in range(len(ctx.history)):
        part = DecisionContext(current=ctx.current.model_copy(update={"exercise_id": ctx.history[k].exercise_id}),
                               history=ctx.history[:k + 1], session_minutes=ctx.session_minutes)
        a = ctx.history[k]
        b.observe(to_state(part, None, k), a.tier, a.hint,
                  Outcome(correct=a.correct, error_type="correct" if a.correct else a.error,
                          response_time=a.response_time_s))
    return b


def evaluate(run_id: str, split: str, rounds: bool, conditions: list[str] | None = None) -> dict:
    pcfg, clinical, bcfg = load_practice(), load_clinical(), load_beta_bandit()
    reg = ModelRegistry(pcfg)
    manifest = reg.run(run_id)
    if set(manifest["demonstrations"]["example_ids"]) & {r["example_id"] for r in load_examples(
            manifest["demonstrations"]["path"], split)}:
        raise RuntimeError(f"{split} situations were used for training in run {run_id}")
    rows = load_examples(manifest["demonstrations"]["path"], split)
    conds = conditions or (["base", "prompted", f"sft:{run_id}"]
                           + ([f"ditto:{run_id}@round{r['round']}" for r in manifest["rounds"][:-1]] if rounds else [])
                           + [f"ditto:{run_id}", "bandit"])
    results = {}
    for cond in conds:
        per = []
        for i, r in enumerate(rows):
            ctx = DecisionContext(**r["context"])
            opts = [parse_action(o) for o in r["options"]]
            fb = bandit_for(ctx, bcfg, clinical, seed=1000 + i)
            d = fb.decide(ctx, opts) if cond == "bandit" else reg.policy(cond).decide(ctx, opts, fb)
            per.append({"example_id": r["example_id"], "rule": r["rule"], "accepted": r["accepted"],
                        "raw_output": d.raw_output, "parsed": d.parsed, "valid": d.valid,
                        "final_action": d.action.key(), "source": d.source, "fallback_reason": d.fallback_reason,
                        "agree_valid": bool(d.valid) and d.action.key() in r["accepted"],
                        "agree_end_to_end": d.action.key() in r["accepted"],
                        "latency_ms": round(d.latency_ms, 1)})
            print(f"{cond:40s} {r['example_id']}: {d.action.key():36s} "
                  f"{'OK ' if per[-1]['agree_end_to_end'] else 'MISS'} {d.fallback_reason or ''}")
        n = len(per)
        lm = cond != "bandit"
        lat = [p["latency_ms"] for p in per]
        results[cond] = {
            "n": n,
            "valid_output": sum(bool(p["valid"]) for p in per) if lm else None,
            "constraint_violations": sum(bool(p["parsed"]) and not p["valid"] for p in per) if lm else None,
            "unparseable": sum(p["parsed"] is False for p in per) if lm else None,
            "fallbacks": sum(p["fallback_reason"] is not None for p in per),
            "agree_on_valid": sum(p["agree_valid"] for p in per) if lm else None,
            "agree_end_to_end": sum(p["agree_end_to_end"] for p in per),
            "latency_ms_median": round(statistics.median(lat), 1),
            "latency_ms_max": round(max(lat), 1),
            "per_example": per,
        }
    return {"run_id": run_id, "split": split, "created": datetime.now(timezone.utc).isoformat(),
            "dataset_hash": examples_hash(rows), "example_ids": [r["example_id"] for r in rows],
            "base_model": pcfg.llm.base_model, "base_revision": pcfg.llm.revision,
            "trained_on": manifest["demonstrations"]["example_ids"], "results": results}


def to_markdown(rep: dict) -> str:
    lines = [f"# Decision-model comparison: run `{rep['run_id']}`, {rep['split']} split",
             "",
             f"{len(rep['example_ids'])} reserved situations (dataset hash `{rep['dataset_hash']}`), never used in "
             f"training or in prompts. Base model `{rep['base_model']}` @ `{(rep['base_revision'] or '')[:7]}`. "
             "Generated by `evaluation/compare.py`; numbers are counts out of the number of situations. "
             "Mock preferences of a fictional demonstrator; a small exploratory check, not a statistical or "
             "clinical validation.",
             "",
             "| model | valid output | constraint violations | unparseable | fallbacks | agrees (valid outputs) | agrees end to end | median latency |",
             "|---|---|---|---|---|---|---|---|"]
    for cond, r in rep["results"].items():
        n = r["n"]
        f = (lambda v: "n/a" if v is None else f"{v}/{n}")
        av = "n/a" if r["agree_on_valid"] is None else f"{r['agree_on_valid']}/{r['valid_output']}"
        lines.append(f"| {cond} | {f(r['valid_output'])} | {f(r['constraint_violations'])} | {f(r['unparseable'])} | "
                     f"{f(r['fallbacks'])} | {av} | {r['agree_end_to_end']}/{n} | {r['latency_ms_median']:.0f} ms |")
    lines += ["", "Per situation (final action; ✓ = among the demonstrator's accepted actions, "
              "† = fallback to the bandit):", ""]
    conds = list(rep["results"])
    lines.append("| situation | rule | accepted | " + " | ".join(conds) + " |")
    lines.append("|---|---|---|" + "---|" * len(conds))
    for i, eid in enumerate(rep["example_ids"]):
        any_r = rep["results"][conds[0]]["per_example"][i]
        cells = []
        for c in conds:
            p = rep["results"][c]["per_example"][i]
            a = parse_action(p["final_action"])
            short = f"retry {a.cue}" if a.operation == "retry_current" else f"next t{a.tier}"
            cells.append(f"{short} {'✓' if p['agree_end_to_end'] else '✗'}{'†' if p['fallback_reason'] else ''}")
        acc = ", ".join((lambda a: f"retry {a.cue}" if a.operation == "retry_current" else f"next t{a.tier}")(
            parse_action(x)) for x in any_r["accepted"])
        lines.append(f"| {eid} | {any_r['rule']} | {acc} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--rounds", action="store_true", help="also evaluate intermediate DITTO rounds")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--markdown", default=None, help="also write a Markdown summary here (e.g. results/...)")
    args = ap.parse_args()
    rep = evaluate(args.run, args.split, args.rounds, args.conditions)
    pcfg = load_practice()
    out = resolve(pcfg.eval_dir) / f"{args.run}-{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    md = to_markdown(rep)
    if args.markdown:
        resolve(args.markdown).write_text(md)
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
