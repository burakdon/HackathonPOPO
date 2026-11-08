"""The SP model reproduces Dell's published behaviour, and the fitter recovers parameters."""
from __future__ import annotations

import numpy as np
import pytest

from patientmodel.dell_sp import CATEGORIES, NAMING_CATEGORIES, DellSPConfig, DellSPModel, load_dell_sp_config
from patientmodel.fit import build_grid, counts_from_proportions, fit_counts

# Published SP-model proportions at Dell's fitted (s, p). Dell et al. (2004) Table 2 "SP-ind" rows,
# renormalised over the six naming categories; T1 is Dell et al. (2013) Table 1.
PUBLISHED = {
    "AK2": ((0.064, 0.074), [0.84, 0.01, 0.00, 0.01, 0.00, 0.00]),
    "JR1": ((0.025, 0.044), [0.46, 0.02, 0.01, 0.01, 0.00, 0.00]),
    "S16": ((0.021, 0.032), [0.66, 0.06, 0.03, 0.02, 0.02, 0.01]),
    "S21": ((0.024, 0.021), [0.61, 0.05, 0.04, 0.01, 0.01, 0.09]),
    "GB": ((0.017, 0.021), [0.40, 0.07, 0.06, 0.02, 0.04, 0.10]),
    "VP": ((0.008, 0.028), [0.28, 0.11, 0.14, 0.02, 0.14, 0.04]),
    "AS1": ((0.005, 0.018), [0.15, 0.09, 0.14, 0.02, 0.15, 0.19]),
    # (Dell 2013 Table 1 is AS1 from Dell 2004 re-reported; kept once as AS1.)
}


@pytest.fixture(scope="session")
def model() -> DellSPModel:
    cfg, _ = load_dell_sp_config()
    return DellSPModel(cfg)


def probs(model: DellSPModel, s: float, p: float, n: int = 6000, seed: int = 0) -> np.ndarray:
    r = model.simulate(s, p, n, seed=seed)
    return np.array([r[c] for c in NAMING_CATEGORIES])


def test_network_layout_matches_walker_2016(model: DellSPModel) -> None:
    # 6 words x 10 features minus 3 shared = 57 semantic units on a typical trial (Walker 2016, p. 37)
    assert model.plain.n_features == 57
    assert model.mixed.n_features == 54
    assert model.plain.onset_ids.size == 6 and model.plain.vowel_ids.size == 2 and model.plain.coda_ids.size == 2


def test_normal_network_names_almost_everything(model: DellSPModel) -> None:
    pr = dict(zip(NAMING_CATEGORIES, probs(model, 0.04, 0.04, n=10000)))
    assert pr["correct"] >= 0.95
    assert pr["formal"] <= 0.01          # "pure formal errors do not occur" for healthy parameters (Walker 2016, p. 38)
    assert pr["semantic"] >= pr["unrelated"]


def test_random_pattern_when_weights_are_zero(model: DellSPModel) -> None:
    pr = dict(zip(NAMING_CATEGORIES, probs(model, 0.0, 0.0, n=10000)))
    assert 0.70 <= pr["nonword"] <= 0.80   # 18 of the 24 CVC strings are not words
    assert pr["mixed"] <= 0.015            # chance opportunity for mixed errors is about .008 (Dell et al. 2004, fn 1)


def test_lesion_directions(model: DellSPModel) -> None:
    low_s = dict(zip(NAMING_CATEGORIES, probs(model, 0.005, 0.04)))
    low_p = dict(zip(NAMING_CATEGORIES, probs(model, 0.04, 0.005)))
    word_errors = lambda d: d["semantic"] + d["formal"] + d["mixed"] + d["unrelated"]
    assert word_errors(low_s) > low_s["nonword"]
    assert low_p["nonword"] > word_errors(low_p)
    assert probs(model, 0.01, 0.04)[0] < probs(model, 0.02, 0.04)[0] < probs(model, 0.04, 0.04)[0]
    assert probs(model, 0.04, 0.01)[0] < probs(model, 0.04, 0.02)[0] < probs(model, 0.04, 0.04)[0]


@pytest.mark.parametrize("name", list(PUBLISHED))
def test_reproduces_published_fits(model: DellSPModel, name: str) -> None:
    (s, p), published = PUBLISHED[name]
    published = np.array(published) / np.sum(published)
    rmsd = np.sqrt(np.mean((probs(model, s, p, n=8000, seed=3) - published) ** 2))
    # Observed .002-.011 across seeds; the alternative noise placement (scaled by the previous
    # activation) gives .08-.13, so .015 separates the two readings with margin.
    assert rmsd < 0.015, f"{name}: rmsd {rmsd:.3f} against Dell's model proportions"


def test_seed_determinism(model: DellSPModel) -> None:
    assert model.simulate(0.02, 0.03, 2000, seed=11) == model.simulate(0.02, 0.03, 2000, seed=11)


def test_omission_threshold_produces_omissions() -> None:
    m = DellSPModel(DellSPConfig(omission_threshold=0.05))
    r = m.simulate(0.01, 0.03, 4000, seed=0)
    assert r["omission"] > 0.05
    assert abs(sum(r.values()) - 1.0) < 1e-9
    assert DellSPModel(DellSPConfig()).simulate(0.01, 0.03, 2000, seed=0)["omission"] == 0.0


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        DellSPConfig(decay=1.5)
    with pytest.raises(ValueError):
        DellSPConfig(shared_features=6)


# ------------------------------------------------------------------------------------ fitting

@pytest.fixture(scope="session")
def coarse_grid(model: DellSPModel):
    return build_grid(model, grid_step=0.005, trials_per_cell=1500, seed=1)


def test_fit_recovers_known_parameters(model: DellSPModel, coarse_grid) -> None:
    truth = (0.020, 0.030)
    cat = model.simulate_trials(*truth, 3000, np.random.default_rng(7))
    counts = dict(zip(NAMING_CATEGORIES, np.bincount(cat, minlength=7)[:6].tolist()))
    fit = fit_counts(counts, coarse_grid)
    assert abs(fit.s - truth[0]) <= 0.005 and abs(fit.p - truth[1]) <= 0.005
    assert fit.s_interval[0] <= fit.s <= fit.s_interval[1]
    assert fit.p_interval[0] <= fit.p <= fit.p_interval[1]


def test_fit_dell_2013_example_patient(coarse_grid) -> None:
    # Dell et al. (2013) Table 1: fitted s = .005, p = .018, rmsd = .045 on a 175-item PNT.
    counts = counts_from_proportions(dict(zip(NAMING_CATEGORIES, [.20, .07, .15, .02, .29, .26])), 175)
    assert sum(counts.values()) == 175
    fit = fit_counts(counts, coarse_grid)
    assert fit.s <= 0.010
    assert 0.010 <= fit.p <= 0.030
    assert fit.rmsd < 0.07
    assert not fit.at_boundary
