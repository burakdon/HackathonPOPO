"""Population sampler, item bank, realizer and patient runtime."""
from __future__ import annotations

import numpy as np
import pytest

from patientmodel.dell_sp import DellSPModel, load_dell_sp_config
from patientmodel.fit import build_grid
from patientmodel.items import load_item_bank
from patientmodel.lexicon import formally_related, phoneme_edit_distance
from patientmodel.population import PopulationConfig, PopulationSampler, expected_accuracy


@pytest.fixture(scope="module")
def model() -> DellSPModel:
    cfg, _ = load_dell_sp_config()
    return DellSPModel(cfg)


@pytest.fixture(scope="module")
def coarse_grid(model: DellSPModel):
    return build_grid(model, grid_step=0.005, trials_per_cell=1500, seed=1)


def test_severity_curve_matches_psst_fit() -> None:
    cfg = PopulationConfig()
    # PSST logistic fit: 9% correct at AQ 30, 72% at AQ 90
    assert abs(expected_accuracy(30, cfg) - 0.09) < 0.02
    assert abs(expected_accuracy(90, cfg) - 0.72) < 0.02


def test_population_tracks_severity(coarse_grid) -> None:
    ps = PopulationSampler(coarse_grid)
    mild = ps.from_profile("mild", 150, seed=0)
    severe = ps.from_profile("severe", 150, seed=0)
    acc = lambda pats: np.mean([p.expected_naming["correct"] for p in pats])
    assert acc(mild) > acc(severe) + 0.3
    assert all(p.s > 0 and p.p > 0 for p in mild + severe)
    assert all(p.aq >= 76 for p in mild) and all(p.aq <= 51 for p in severe)


def test_population_is_reproducible(coarse_grid) -> None:
    ps = PopulationSampler(coarse_grid)
    a, b = ps.sample(20, seed=4), ps.sample(20, seed=4)
    assert [(x.s, x.p, x.aq) for x in a] == [(x.s, x.p, x.aq) for x in b]


def test_ps_ratio_prior_is_respected(coarse_grid) -> None:
    ps = PopulationSampler(coarse_grid)
    pats = ps.sample(400, seed=2)
    log_ratio = np.log([p.p / p.s for p in pats])
    assert 0.2 < log_ratio.mean() < 0.9      # Dell's fitted patients: about 0.5


def test_item_bank_loads_with_norms_and_pictures() -> None:
    items = load_item_bank()
    assert len(items) >= 500
    assert sum(i.image is not None for i in items) >= 200
    words = {i.word for i in items}
    assert {"house", "cat", "volcano"} <= words
    d = {i.word: i.difficulty for i in items}
    assert d["house"] < d["volcano"]          # frequent, short, high agreement vs rare, long


def test_phonological_helpers() -> None:
    assert phoneme_edit_distance(("K", "AE", "T"), ("K", "AE", "T")) == 0
    assert phoneme_edit_distance(("K", "AE", "T"), ("M", "AE", "T")) == 1
    assert formally_related(("K", "AE", "T"), ("K", "AW"))          # shared onset
    assert formally_related(("K", "AE", "T"), ("B", "AE", "T"))     # shared vowel and coda
    assert not formally_related(("K", "AE", "T"), ("D", "AO", "G"))


@pytest.mark.slow
def test_patient_end_to_end_with_lexicon(model: DellSPModel) -> None:
    from patientmodel.lexicon import build_lexicon
    from patientmodel.patient import Patient
    from patientmodel.realize import Realizer
    items = load_item_bank()
    lex = build_lexicon(extra_words=[i.word for i in items])
    rz = Realizer(lex)
    bank = [i for i in items if i.word in lex.index][:120]
    mild = Patient("mild", 0.025, 0.044, model, rz, rng=np.random.default_rng(0))
    severe = Patient("severe", 0.005, 0.018, model, rz, rng=np.random.default_rng(0))
    rm, rs = mild.naming_test(bank), severe.naming_test(bank)
    assert np.mean([r.correct for r in rm]) > np.mean([r.correct for r in rs]) + 0.3
    for r in rm + rs:
        assert r.category in {"correct", "semantic", "formal", "mixed", "unrelated", "nonword", "omission"}
        assert r.latency_s > 0
        if r.category == "correct":
            assert r.word == r.item.word
        if r.category == "nonword":
            assert not lex.is_word(r.phonemes) and r.word != r.item.word
        if r.category in {"semantic", "formal", "mixed", "unrelated"}:
            assert r.word in lex.index and r.word != r.item.word
    # item effects: easy items are named more often than hard ones by the mild patient
    d = np.array([r.item.difficulty for r in rm]); c = np.array([r.correct for r in rm])
    assert c[d < np.median(d)].mean() >= c[d >= np.median(d)].mean()
