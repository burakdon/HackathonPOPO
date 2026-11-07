"""PRELIMINARY comparison: baselines, bandits and PPO on the three preset cohorts (held-out test split).

Every policy faces the same patients with the same random outcome stream (common random numbers),
so differences between rows are due to the policy, not to luck. Normal-approximation 95% CIs over
patients; PPO metrics are first averaged over its training seeds for each patient. The full
evaluation (oracle, bootstrap CIs, 200+ patients) is eval/compare.py.

"learn eff." is a diagnostic for the learning-gain column: the mean, over trials, of the expected
item learning per trial implied by the simulator's learning rule at the true p (1.0 = every trial
at the learning peak with no cue).

Usage:  uv run python -m scripts.sanity_table --sessions 50 --dropout on --out results/sanity_dropout_on.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agents.baselines import FixedPolicy, StaircasePolicy
from agents.beta_bandit import BetaThompsonBandit
from agents.logistic_bandit import LogisticThompsonBandit
from core.config import load_beta_bandit, load_clinical, load_logistic_bandit, resolve
from eval.metrics import mean_ci, session_metrics
from sim.config import load_sim
from sim.items import load_items
from sim.population import make_patients
from sim.session import run_session

COLUMNS = [("time_in_band", "in band", 2), ("accuracy", "acc", 2), ("mean_tier", "mean tier", 2),
           ("frac_tier1", "% tier1", 2), ("learning_gain", "gain x100", 2), ("learn_eff", "learn eff.", 2),
           ("control_error", "|p-.85|", 3), ("frustration_events", "frustr.", 2), ("quit", "quit", 2),
           ("n_trials", "trials", 1)]


def learning_efficiency(trace: list[dict], cfg, clinical) -> float:
    a, b = cfg.learning.a, cfg.learning.b
    peak = a / (a + b)
    norm = peak ** a * (1 - peak) ** b
    max_s = max(clinical.cue_strength.values())
    vals = [(t["p_true"] ** a * (1 - t["p_true"]) ** b / norm)
            * (1 - cfg.learning.cue_discount * clinical.cue_strength[t["cue"]] / max_s) for t in trace]
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", choices=["on", "off"], default="on")
    ap.add_argument("--ppo-models", nargs="*", default=["models/ppo_seed0.zip", "models/ppo_seed1.zip",
                                                          "models/ppo_seed2.zip"])
    ap.add_argument("--out", default=None, help="also write the table to this markdown file")
    args = ap.parse_args()

    cfg, clinical = load_sim(), load_clinical()
    cfg = cfg.model_copy(update={"dropout": cfg.dropout.model_copy(update={"enabled": args.dropout == "on"})})
    bank = load_items(cfg.items)
    beta_cfg, logit_cfg = load_beta_bandit(), load_logistic_bandit()
    makers: dict[str, list] = {
        "fixed (tier 2)": [lambda k: FixedPolicy(tier=2)],
        "staircase": [lambda k: StaircasePolicy()],
        "Beta bandit": [lambda k: BetaThompsonBandit(beta_cfg, clinical, seed=args.seed * 100_000 + k)],
        "contextual bandit": [lambda k: LogisticThompsonBandit(logit_cfg, clinical, seed=args.seed * 100_000 + k)],
    }
    ppo_paths = [p for p in args.ppo_models if resolve(p).exists()]
    if ppo_paths:
        from agents.ppo import PPOPolicy
        from envs.aphasia_env import load_env_config
        env_cfg = load_env_config()
        ppo_policies = [PPOPolicy(p, clinical, env_cfg) for p in ppo_paths]
        makers[f"PPO ({len(ppo_paths)} seeds)"] = [lambda k, pol=pol: pol for pol in ppo_policies]

    lines = [f"PRELIMINARY: {args.sessions} held-out patients per cohort, {cfg.session.n_trials} trials, start tier "
             f"{cfg.session.start_tier}, dropout {args.dropout}; mean ± 95% CI over patients (normal approx.)", "",
             "| cohort | policy | " + " | ".join(h for _, h, _ in COLUMNS) + " |",
             "|" + "---|" * (len(COLUMNS) + 2)]
    print("\n".join(lines), flush=True)
    for cohort in cfg.presets:
        specs = make_patients(cfg, cohort, "test", args.sessions, root_seed=args.seed)
        for name, variants in makers.items():
            per_patient = []
            for spec in specs:
                runs = []
                for make in variants:   # several seeds: average per patient
                    patient, selector = spec.build(bank, cfg, clinical)
                    res = run_session(make(spec.index), patient, selector, cfg, clinical)
                    m = session_metrics(res, clinical)
                    m["learn_eff"] = learning_efficiency(res.trace, cfg, clinical)
                    runs.append(m)
                per_patient.append({k: float(np.mean([r[k] for r in runs])) for k in runs[0]})
            cells = []
            for key, _, nd in COLUMNS:
                vals = np.array([r[key] for r in per_patient]) * (100 if key == "learning_gain" else 1)
                m, h = mean_ci(vals)
                cells.append(f"{m:.{nd}f} ± {h:.{nd}f}")
            line = f"| {cohort} | {name} | " + " | ".join(cells) + " |"
            lines.append(line)
            print(line, flush=True)
    if args.out:
        out = resolve(args.out)
        out.write_text(f"<!-- Generated by: uv run python -m scripts.sanity_table --sessions {args.sessions} "
                       f"--dropout {args.dropout} (seed {args.seed}). Do not edit by hand. -->\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
