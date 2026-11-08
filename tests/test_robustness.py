"""Scorer noise and out-of-distribution cohorts."""
import numpy as np

from agents.baselines import StaircasePolicy
from core.config import load_clinical
from sim.config import load_sim
from sim.items import load_items
from sim.population import make_patients
from sim.session import run_session


def _run(noise, seed=0, k=0):
    cfg, cl = load_sim(), load_clinical()
    bank = load_items(cfg.items)
    spec = make_patients(cfg, "moderate_broca", "test", 1, 0, start=k)[0]
    p, s = spec.build(bank, cfg, cl)
    rng = np.random.default_rng(seed) if noise else None
    return run_session(StaircasePolicy(), p, s, cfg, cl, obs_noise=noise, noise_rng=rng).trace


def test_noise_flips_observed_outcomes_at_the_requested_rate_and_metrics_stay_true():
    flips = n = 0
    for k in range(40):
        tr = _run(0.10, seed=k, k=k)
        flips += sum(t["observed_outcome"] != t["outcome"] for t in tr)
        n += len(tr)
        assert all((t["error_type"] == "correct") == t["outcome"] for t in tr)   # true outcome kept in trace
    assert 0.07 < flips / n < 0.13


def test_zero_noise_is_identical_to_no_noise():
    assert _run(0.0) == _run(0.0)
    assert all(t["observed_outcome"] == t["outcome"] for t in _run(0.0))


def test_ood_cohorts_lie_outside_the_training_ranges():
    cfg = load_sim()
    moved = {"ood_very_severe": ["ability"], "ood_fast_fatigue": ["fatigue_rate", "fatigue_recovery"],
             "ood_semantic_responder": ["phon_resp", "sem_resp"]}
    for cohort, params in moved.items():
        specs = make_patients(cfg, cohort, "test", 100, 0)
        for name in params:
            v = np.array([getattr(s.params, name) for s in specs])
            d = cfg.population[name]
            assert np.all((v < d.low) | (v > d.high)), (cohort, name)
