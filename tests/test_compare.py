"""Evaluation harness: bootstrap CIs, paired differences, oracle, outputs."""
import csv

import numpy as np

from core.config import load_yaml
from eval.compare import bootstrap_ci, evaluate, summarise


def test_bootstrap_ci_brackets_the_mean_and_ignores_nan():
    rng = np.random.default_rng(0)
    v = np.r_[rng.normal(1.0, 1.0, 400), [np.nan] * 5]
    m, lo, hi = bootstrap_ci(v, 1000, rng)
    assert lo < m < hi and abs(m - np.nanmean(v)) < 1e-12 and 0.1 < hi - lo < 0.3


def test_harness_end_to_end_small(tmp_path):
    cfg = load_yaml("configs/eval.yaml")
    cfg.update(n_patients=6, bootstrap_samples=200, cohorts=["severe"])
    ev = evaluate(cfg)
    summarise(cfg, ev, tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "comparison.csv")))
    assert len(rows) == len(cfg["policies"]) * len(cfg["metrics"])
    for r in rows:
        if r["mean"] != "nan":
            assert float(r["ci_low"]) <= float(r["mean"]) <= float(r["ci_high"])
    oracle_gain = next(r for r in rows if r["policy"] == "oracle" and r["metric"] == "learning_gain")
    assert abs(float(oracle_gain["pct_of_oracle"]) - 100.0) < 1e-9
    paired = list(csv.DictReader(open(tmp_path / "paired_differences.csv")))
    assert {(p["a"], p["b"]) for p in paired} == {tuple(x) for x in cfg["paired"]}
    get = lambda key, m: np.array([p[m] for p in ev["per_patient"][("severe", key)]])
    d = next(p for p in paired if p["a"] == "ppo_sim" and p["b"] == "fixed" and p["metric"] == "control_error")
    assert abs(float(d["mean_diff"]) - np.mean(get("ppo_sim", "control_error") - get("fixed", "control_error"))) < 1e-12
    assert (tmp_path / "comparison.md").read_text().count("|") > 50


def test_oracle_controls_true_difficulty_better_than_fixed():
    from agents.baselines import FixedPolicy
    from agents.oracle import OraclePolicy
    from core.config import load_clinical
    from eval.metrics import session_metrics
    from sim.config import load_sim
    from sim.items import load_items
    from sim.population import make_patients
    from sim.session import run_session
    cfg, cl = load_sim(), load_clinical()
    bank = load_items(cfg.items)
    err = {"oracle": [], "fixed": []}
    for spec in make_patients(cfg, "severe", "test", 10, 0):
        p, s = spec.build(bank, cfg, cl)
        err["oracle"].append(session_metrics(run_session(OraclePolicy(p, cl, 0.06, 0.03, 0.01, True), p, s, cfg, cl), cl)["control_error"])
        p, s = spec.build(bank, cfg, cl)
        err["fixed"].append(session_metrics(run_session(FixedPolicy(2), p, s, cfg, cl), cl)["control_error"])
    assert np.mean(err["oracle"]) < np.mean(err["fixed"]) - 0.1
