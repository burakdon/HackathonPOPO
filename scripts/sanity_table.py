"""PRELIMINARY sanity table: baselines vs bandits on the three preset cohorts (held-out test split).

Every policy faces the same patients with the same random outcome stream (common random numbers),
so differences between rows are due to the policy, not to luck. Normal-approximation 95% CIs.
The full evaluation (oracle, PPO, bootstrap CIs, more sessions) is eval/compare.py in Step 3.

Usage:  uv run python -m scripts.sanity_table --sessions 50
"""
from __future__ import annotations

import argparse

import numpy as np

from agents.baselines import FixedPolicy, StaircasePolicy
from agents.beta_bandit import BetaThompsonBandit
from agents.logistic_bandit import LogisticThompsonBandit
from core.config import load_beta_bandit, load_clinical, load_logistic_bandit
from eval.metrics import mean_ci, session_metrics
from sim.config import load_sim
from sim.items import load_items
from sim.population import make_patients
from sim.session import run_session

COLUMNS = [("time_in_band", "in band", 2), ("accuracy", "acc", 2), ("mean_tier", "mean tier", 2),
           ("tier_progression", "tier +", 2), ("frac_tier1", "% tier1", 2),
           ("learning_gain", "gain x100", 2), ("control_error", "|p-.85|", 3),
           ("frustration_events", "frustr.", 2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, clinical = load_sim(), load_clinical()
    bank = load_items(cfg.items)
    beta_cfg, logit_cfg = load_beta_bandit(), load_logistic_bandit()
    makers = {
        "fixed(t2)": lambda k: FixedPolicy(tier=2),
        "staircase": lambda k: StaircasePolicy(),
        "beta_bandit": lambda k: BetaThompsonBandit(beta_cfg, clinical, seed=args.seed * 100_000 + k),
        "logistic_bandit(hand prior)": lambda k: LogisticThompsonBandit(
            logit_cfg.model_copy(update={"prior_file": None}), clinical, seed=args.seed * 100_000 + k),
        "logistic_bandit(fitted prior)": lambda k: LogisticThompsonBandit(
            logit_cfg.model_copy(update={"prior_file": "configs/logistic_bandit_prior.yaml"}), clinical,
            seed=args.seed * 100_000 + k),
    }
    print(f"PRELIMINARY: {args.sessions} held-out patients per cohort, {cfg.session.n_trials} trials, "
          f"start tier {cfg.session.start_tier}; mean ± 95% CI (normal approx.)\n")
    header = f"| cohort | policy | " + " | ".join(h for _, h, _ in COLUMNS) + " |"
    print(header + "\n|" + "---|" * (len(COLUMNS) + 2))
    for cohort in cfg.presets:
        specs = make_patients(cfg, cohort, "test", args.sessions, root_seed=args.seed)
        for name, make in makers.items():
            rows = []
            for spec in specs:
                patient, selector = spec.build(bank, cfg, clinical)
                rows.append(session_metrics(run_session(make(spec.index), patient, selector, cfg, clinical), clinical))
            cells = []
            for key, _, nd in COLUMNS:
                vals = np.array([r[key] for r in rows]) * (100 if key == "learning_gain" else 1)
                m, h = mean_ci(vals)
                cells.append(f"{m:.{nd}f} ± {h:.{nd}f}")
            print(f"| {cohort} | {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
