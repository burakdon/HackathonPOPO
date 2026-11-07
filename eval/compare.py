"""Full evaluation: every policy on the same held-out patients, bootstrap CIs and paired differences.

  uv run python -m eval.compare --config configs/eval.yaml

For each cohort, the same `n_patients` held-out TEST patients are played by every policy with the
same random outcome stream (common random numbers). Learned policies with several training seeds
are run once per seed and their metrics averaged per patient (per-seed means are written too).
Outputs (results/):
  comparison.csv           cohort, policy, metric, mean, 95% CI (percentile bootstrap over patients)
  comparison_seeds.csv     per-seed means for multi-seed policies
  paired_differences.csv   A - B on the same patients, bootstrap 95% CI, share of bootstrap draws > 0
  comparison.md            readable tables
  comparison.png           bars with CIs: true control error, learning gain, probe lift, quit rate
  rolling_accuracy.png     rolling accuracy vs trial, the 75-90% band shaded
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from agents.baselines import FixedPolicy, StaircasePolicy
from agents.logistic_bandit import LogisticThompsonBandit
from agents.oracle import OraclePolicy
from core.config import load_clinical, load_logistic_bandit, load_yaml, resolve
from eval.metrics import rolling_accuracy, session_metrics
from sim.config import load_sim
from sim.items import load_items
from sim.population import make_patients
from sim.session import run_session


def bootstrap_ci(values: np.ndarray, B: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean and percentile 95% CI of the mean, over non-NaN values."""
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    means = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def make_runners(cfg: dict, clinical) -> dict[str, list]:
    """policy key -> list of factories (one per training seed); factory(spec_index, patient) -> Policy."""
    lcfg = load_logistic_bandit()
    o = cfg["oracle"]
    runners: dict[str, list] = {}
    for p in cfg["policies"]:
        t = p["type"]
        if t == "fixed":
            runners[p["key"]] = [lambda k, pat, tier=p["tier"]: FixedPolicy(tier=tier)]
        elif t == "staircase":
            runners[p["key"]] = [lambda k, pat: StaircasePolicy()]
        elif t == "logistic_bandit":
            runners[p["key"]] = [lambda k, pat: LogisticThompsonBandit(lcfg, clinical, seed=cfg["seed"] * 100_000 + k)]
        elif t == "ppo":
            from agents.ppo import PPOPolicy
            from envs.aphasia_env import load_env_config
            env_cfg = load_env_config()
            pols = [PPOPolicy(m, clinical, env_cfg) for m in p["models"]]
            runners[p["key"]] = [lambda k, pat, pol=pol: pol for pol in pols]
        elif t == "oracle":
            runners[p["key"]] = [lambda k, pat: OraclePolicy(pat, clinical, o["tolerance"], o["cue_cost"],
                                                             o["tier_bonus"], o["cue_only_after_error"])]
        elif t == "gemini":
            from agents.llm_prompted import GeminiPolicy
            runners[p["key"]] = [lambda k, pat: GeminiPolicy(clinical, seed=k)]
        else:
            raise ValueError(f"unknown policy type {t}")
    return runners


def evaluate(cfg: dict) -> dict:
    clinical, sim_cfg = load_clinical(), load_sim()
    sim_cfg = sim_cfg.model_copy(update={"dropout": sim_cfg.dropout.model_copy(update={"enabled": cfg["dropout"]})})
    if cfg.get("learning_peak") is not None:   # ablation: simulator whose learning peaks elsewhere
        sim_cfg = sim_cfg.model_copy(update={"learning": sim_cfg.learning.model_copy(update={"peak": cfg["learning_peak"]})})
    bank = load_items(sim_cfg.items)
    runners = make_runners(cfg, clinical)
    n_trials, W = sim_cfg.session.n_trials, clinical.window
    per_patient, per_seed, curves = {}, {}, {}
    for cohort in cfg["cohorts"]:
        specs = make_patients(sim_cfg, cohort, "test", cfg["n_patients"], root_seed=cfg["seed"])
        for key, factories in runners.items():
            t0 = time.time()
            rows, seed_rows, rolls = [], [[] for _ in factories], []
            for spec in specs:
                runs = []
                for s, make in enumerate(factories):
                    patient, selector = spec.build(bank, sim_cfg, clinical)
                    res = run_session(make(spec.index, patient), patient, selector, sim_cfg, clinical)
                    m = session_metrics(res, clinical)
                    runs.append(m)
                    seed_rows[s].append(m)
                    r = np.full(n_trials - W + 1, np.nan)
                    ra = rolling_accuracy([t["outcome"] for t in res.trace], W)
                    r[: len(ra)] = ra
                    rolls.append(r)
                rows.append({k: float(np.nanmean([x[k] for x in runs])) for k in cfg["metrics"]})
            per_patient[(cohort, key)] = rows
            per_seed[(cohort, key)] = seed_rows
            curves[(cohort, key)] = np.array(rolls)
            print(f"  {cohort:15} {key:18} {len(specs)} patients x {len(factories)} seed(s) in {time.time() - t0:.1f}s",
                  flush=True)
    return {"per_patient": per_patient, "per_seed": per_seed, "curves": curves, "window": W, "n_trials": n_trials}


def summarise(cfg: dict, ev: dict, out: Path) -> None:
    rng = np.random.default_rng(cfg["seed"])
    B, labels = cfg["bootstrap_samples"], {p["key"]: p["label"] for p in cfg["policies"]}
    rows = []
    for (cohort, key), pts in ev["per_patient"].items():
        for metric in cfg["metrics"]:
            v = np.array([p[metric] for p in pts])
            mean, lo, hi = bootstrap_ci(v, B, rng)
            rows.append({"cohort": cohort, "policy": key, "label": labels[key], "metric": metric,
                         "mean": mean, "ci_low": lo, "ci_high": hi, "n": int((~np.isnan(v)).sum())})
    oracle = {(r["cohort"], r["metric"]): r["mean"] for r in rows if r["policy"] == "oracle"}
    for r in rows:
        r["pct_of_oracle"] = (100 * r["mean"] / oracle[(r["cohort"], r["metric"])]
                              if r["metric"] in cfg["pct_of_oracle"] and (r["cohort"], r["metric"]) in oracle else "")
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    seed_rows = []
    for (cohort, key), seeds in ev["per_seed"].items():
        if len(seeds) > 1:
            for s, srows in enumerate(seeds):
                for metric in cfg["metrics"]:
                    seed_rows.append({"cohort": cohort, "policy": key, "seed": s, "metric": metric,
                                      "mean": float(np.nanmean([x[metric] for x in srows]))})
    with open(out / "comparison_seeds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cohort", "policy", "seed", "metric", "mean"])
        w.writeheader()
        w.writerows(seed_rows)

    paired = []
    for a, b in cfg["paired"]:
        for cohort in cfg["cohorts"]:
            for metric in cfg["metrics"]:
                va = np.array([p[metric] for p in ev["per_patient"][(cohort, a)]])
                vb = np.array([p[metric] for p in ev["per_patient"][(cohort, b)]])
                d = va - vb
                d = d[~np.isnan(d)]
                if not len(d):
                    continue
                means = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
                paired.append({"cohort": cohort, "a": a, "b": b, "metric": metric, "mean_diff": float(d.mean()),
                               "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5)),
                               "p_draws_positive": float(np.mean(means > 0)), "n": len(d)})
    with open(out / "paired_differences.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0]))
        w.writeheader()
        w.writerows(paired)

    write_markdown(cfg, rows, paired, labels, out / "comparison.md")


def write_markdown(cfg, rows, paired, labels, path: Path) -> None:
    show = [("time_in_band", "in band", 2, 1), ("control_error", "\\|p-0.85\\|", 3, 1), ("learning_gain", "gain x100", 2, 100),
            ("probe_recall", "probe recall", 2, 1), ("probe_lift", "probe lift x100", 2, 100), ("quit", "quit", 2, 1),
            ("frustration_events", "frustration", 2, 1)]
    get = {(r["cohort"], r["policy"], r["metric"]): r for r in rows}
    lines = ["<!-- Generated by: uv run python -m eval.compare --config configs/eval.yaml. Do not edit by hand. -->",
             f"{cfg['n_patients']} held-out test patients per cohort; dropout {'on' if cfg['dropout'] else 'off'}; "
             f"mean [95% bootstrap CI]; learned policies averaged over their training seeds per patient.", ""]
    for cohort in cfg["cohorts"]:
        lines += [f"**{cohort}**", "", "| policy | " + " | ".join(h for _, h, _, _ in show) + " |",
                  "|" + "---|" * (len(show) + 1)]
        for p in cfg["policies"]:
            cells = []
            for m, _, nd, sc in show:
                r = get[(cohort, p["key"], m)]
                cell = f"{sc * r['mean']:.{nd}f} [{sc * r['ci_low']:.{nd}f}, {sc * r['ci_high']:.{nd}f}]"
                if r["pct_of_oracle"] != "":
                    cell += f" ({r['pct_of_oracle']:.0f}%)"
                cells.append(cell)
            lines.append(f"| {p['label']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["Percentages: share of the privileged oracle's value.", "",
              "**Paired differences (A - B, same patients), mean [95% bootstrap CI]**", "",
              "| cohort | A - B | \\|p-0.85\\| | gain x100 | probe lift x100 | quit |", "|---|---|---|---|---|---|"]
    pget = {(d["cohort"], d["a"], d["b"], d["metric"]): d for d in paired}
    for a, b in cfg["paired"]:
        for cohort in cfg["cohorts"]:
            cells = []
            for m, sc, nd in (("control_error", 1, 3), ("learning_gain", 100, 2), ("probe_lift", 100, 2), ("quit", 1, 2)):
                d = pget.get((cohort, a, b, m))
                cells.append("" if d is None else f"{sc * d['mean_diff']:+.{nd}f} [{sc * d['ci_low']:+.{nd}f}, {sc * d['ci_high']:+.{nd}f}]")
            lines.append(f"| {cohort} | {labels[a]} - {labels[b]} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--n-patients", type=int, default=None, help="override for quick runs")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.n_patients:
        cfg["n_patients"] = args.n_patients
    out = resolve(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ev = evaluate(cfg)
    summarise(cfg, ev, out)
    from eval.plots import plot_comparison_bars, plot_rolling_accuracy
    plot_comparison_bars(cfg, out / "comparison.csv", out / "comparison.png")
    plot_rolling_accuracy(cfg, ev, out / "rolling_accuracy.png")
    print(f"done in {time.time() - t0:.0f}s -> {out}/comparison.csv, comparison.md, paired_differences.csv, "
          f"comparison.png, rolling_accuracy.png")


if __name__ == "__main__":
    main()
