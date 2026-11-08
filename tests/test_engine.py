"""Population sampler, item bank, lexicon, realizer, fitter statistics and patient runtime."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from patientmodel.dell_sp import NAMING_CATEGORIES, DellSPConfig, DellSPModel, load_dell_sp_config
from patientmodel.fit import GridTable, build_grid, counts_from_proportions, fit_counts, grid_cache_path
from patientmodel.items import load_item_bank
from patientmodel.lexicon import VOWELS, formally_related, is_phonotactically_legal, phoneme_edit_distance
from patientmodel.patient import ResponseDurationModel
from patientmodel.population import PopulationConfig, PopulationSampler, expected_accuracy


@pytest.fixture(scope="session")
def model() -> DellSPModel:
    cfg, _ = load_dell_sp_config()
    return DellSPModel(cfg)


@pytest.fixture(scope="session")
def coarse_grid(model: DellSPModel) -> GridTable:
    return build_grid(model, grid_step=0.005, trials_per_cell=1500, seed=1)


# ------------------------------------------------------------------------------ population

def test_severity_curve_is_monotone_and_in_range() -> None:
    cfg = PopulationConfig()
    aq = np.array([20.0, 50.0, 80.0, 95.0])
    acc = expected_accuracy(aq, cfg)
    assert np.all(np.diff(acc) > 0) and 0.1 < acc[0] < 0.4 and 0.5 < acc[-1] < 0.9


def test_population_tracks_severity(coarse_grid) -> None:
    ps = PopulationSampler(coarse_grid)
    mild = ps.from_profile("mild", 150, seed=0)
    severe = ps.from_profile("severe", 150, seed=0)
    acc = lambda pats: np.mean([p.expected_naming["correct"] for p in pats])
    assert acc(mild) > acc(severe) + 0.2
    assert all(p.s > 0 and p.p > 0 for p in mild + severe)


def test_population_is_reproducible_and_profiles_differ(coarse_grid) -> None:
    ps = PopulationSampler(coarse_grid)
    a, b = ps.sample(20, seed=4), ps.sample(20, seed=4)
    assert [(x.s, x.p, x.aq) for x in a] == [(x.s, x.p, x.aq) for x in b]
    m1, m2 = ps.from_profile("mild", 5, seed=1), ps.from_profile("mild", 5, seed=2)
    assert [x.aq for x in m1] != [x.aq for x in m2]


def test_ps_ratio_prior_shifts_the_cohort(coarse_grid) -> None:
    with_prior = PopulationSampler(coarse_grid, PopulationConfig(log_ps_ratio_mean=0.5, log_ps_ratio_sd=0.6))
    flat = PopulationSampler(coarse_grid, PopulationConfig(log_ps_ratio_mean=0.0, log_ps_ratio_sd=50.0))
    lr = lambda ps: np.mean([np.log(p.p / p.s) for p in ps.sample(400, seed=2)])
    assert lr(with_prior) > lr(flat) + 0.15
    assert 0.2 < lr(with_prior) < 0.9


# ------------------------------------------------------------------------------ items and lexicon helpers

def test_item_bank_loads_with_norms_and_pictures() -> None:
    items = load_item_bank()
    assert len(items) >= 500
    assert sum(i.image is not None for i in items) >= 200
    d = {i.word: i.difficulty for i in items}
    assert d["house"] < d["volcano"]
    assert 20 <= sum(not i.pronounceable for i in items) <= 60      # compound names such as "wateringcan"
    assert all(not i.pronounceable for i in items if i.word == "wateringcan")


def test_phonological_helpers() -> None:
    assert phoneme_edit_distance(("K", "AE", "T"), ("K", "AE", "T")) == 0
    assert phoneme_edit_distance(("K", "AE", "T"), ("M", "AE", "T")) == 1
    assert formally_related(("K", "AE", "T"), ("K", "AW"))
    assert formally_related(("K", "AE", "T"), ("B", "AE", "T"))
    assert not formally_related(("K", "AE", "T"), ("D", "AO", "G"))


def test_phonotactic_legality() -> None:
    assert is_phonotactically_legal(("K", "AE", "T"))
    assert is_phonotactically_legal(("S", "T", "R", "IY", "T"))
    assert not is_phonotactically_legal(("HH", "AA", "HH"))        # HH must precede a vowel
    assert not is_phonotactically_legal(("ZH", "M", "EY", "Z"))    # ZH never initial, ZH+M not an onset
    assert not is_phonotactically_legal(("K", "K", "AE", "T"))     # identical adjacent phonemes
    assert not is_phonotactically_legal(("B", "R", "N"))           # no vowel
    assert not is_phonotactically_legal(("NG", "OY", "G"))
    corkscrew = ("K", "AO", "R", "K", "S", "K", "R", "UW")
    assert not is_phonotactically_legal(corkscrew)                  # five-consonant run, strict
    assert is_phonotactically_legal(("G", "AO", "R", "K", "S", "K", "R", "UW"), reference=corkscrew)


# ------------------------------------------------------------------------------ fitter statistics

def test_grid_cache_key_covers_every_config_field() -> None:
    base = DellSPConfig()
    base_key = grid_cache_path(base, 0.002, 4000, 0)
    for f in dataclasses.fields(DellSPConfig):
        v = getattr(base, f.name)
        new = (v + 0.011) if isinstance(v, float) else (v + 1) if isinstance(v, int) else 0.03
        changed = dataclasses.replace(base, **{f.name: new})
        assert grid_cache_path(changed, 0.002, 4000, 0) != base_key, f.name
    assert grid_cache_path(base, 0.005, 4000, 0) != base_key
    assert grid_cache_path(base, 0.002, 1000, 0) != base_key
    assert grid_cache_path(base, 0.002, 4000, 7) != base_key


def test_counts_use_effective_naming_n() -> None:
    props = dict(zip(NAMING_CATEGORIES, [.17, .03, .00, .01, .00, .00]))   # IOC, 79% non-naming
    assert sum(counts_from_proportions(props, 175, non_naming_share=0.79).values()) == 37
    assert sum(counts_from_proportions(props, 175).values()) == 175


def test_fit_interval_widens_with_less_evidence(coarse_grid) -> None:
    props = dict(zip(NAMING_CATEGORIES, [.46, .02, .01, .01, .00, .00]))
    wide = fit_counts(counts_from_proportions(props, 175, non_naming_share=0.5), coarse_grid)
    narrow = fit_counts(counts_from_proportions(props, 175), coarse_grid)
    width = lambda f: (f.s_interval[1] - f.s_interval[0]) + (f.p_interval[1] - f.p_interval[0])
    assert wide.n == 88 and narrow.n == 175
    assert width(wide) >= width(narrow)


def test_fit_flags_boundary_cells(coarse_grid) -> None:
    # The random pattern (no lexical influence) is the s = p = 0 corner of the grid.
    random_pattern = dict(zip(NAMING_CATEGORIES, [.04, .04, .08, .01, .08, .75]))
    fit = fit_counts(counts_from_proportions(random_pattern, 2000), coarse_grid)
    assert fit.at_boundary and fit.s == 0.0 and fit.p == 0.0
    interior = fit_counts(counts_from_proportions(dict(zip(NAMING_CATEGORIES, [.40, .07, .06, .02, .04, .10])), 175), coarse_grid)
    assert not interior.at_boundary


def test_build_grid_handles_all_omission_cells(model: DellSPModel) -> None:
    m = DellSPModel(dataclasses.replace(model.cfg, omission_threshold=1e6))
    table = build_grid(m, grid_step=0.03, trials_per_cell=50, seed=0)
    assert np.isfinite(table.probs).all()
    assert np.allclose(table.probs.sum(axis=-1), 1.0)


def test_fit_recovers_known_parameters(model: DellSPModel, coarse_grid) -> None:
    truth = (0.020, 0.030)
    cat = model.simulate_trials(*truth, 3000, np.random.default_rng(7))
    counts = dict(zip(NAMING_CATEGORIES, np.bincount(cat, minlength=7)[:6].tolist()))
    fit = fit_counts(counts, coarse_grid)
    assert abs(fit.s - truth[0]) <= 0.005 and abs(fit.p - truth[1]) <= 0.005
    assert fit.s_interval[0] <= fit.s <= fit.s_interval[1]
    assert fit.p_interval[0] <= fit.p <= fit.p_interval[1]
    assert not fit.at_boundary


def test_fit_dell_2013_example_patient(coarse_grid) -> None:
    counts = counts_from_proportions(dict(zip(NAMING_CATEGORIES, [.20, .07, .15, .02, .29, .26])), 175)
    fit = fit_counts(counts, coarse_grid)
    assert fit.s <= 0.010 and 0.010 <= fit.p <= 0.030 and fit.rmsd < 0.07


# ------------------------------------------------------------------------------ patient runtime

def test_duration_model_is_a_duration_not_latency() -> None:
    m = ResponseDurationModel()
    rng = np.random.default_rng(0)
    ok = np.median([m.sample("correct", rng) for _ in range(4000)])
    err = np.median([m.sample("semantic", rng) for _ in range(4000)])
    assert abs(ok - 1.38) < 0.12 and abs(err - 2.15) < 0.2
    assert m.sample("omission", rng) > np.percentile([m.sample("nonword", rng) for _ in range(4000)], 90)


@pytest.mark.slow
def test_patient_end_to_end_with_lexicon(model: DellSPModel) -> None:
    from patientmodel.lexicon import build_lexicon, wordnet
    from patientmodel.patient import Patient
    from patientmodel.realize import Realizer
    items = load_item_bank()
    lex = build_lexicon(extra_words=[i.word for i in items if i.pronounceable])
    wn = wordnet()
    rz = Realizer(lex)
    bank = [i for i in items if i.pronounceable and i.word in lex.index][:150]

    # distinct patients without explicit seeds get distinct streams
    a, b = Patient("a", 0.016, 0.027, model, rz), Patient("b", 0.016, 0.027, model, rz)
    assert [r.category for r in a.naming_test(bank[:30])] != [r.category for r in b.naming_test(bank[:30])]
    # diagnostics do not perturb the response stream
    c1 = Patient("c", 0.016, 0.027, model, rz)
    c2 = Patient("c", 0.016, 0.027, model, rz)
    c2.category_probabilities(bank[0], n=200)
    assert [r.category for r in c1.naming_test(bank[:20])] == [r.category for r in c2.naming_test(bank[:20])]

    mild = Patient("mild", 0.025, 0.044, model, rz, rng=np.random.default_rng(0))
    severe = Patient("severe", 0.005, 0.018, model, rz, rng=np.random.default_rng(0))
    rm, rs = mild.naming_test(bank), severe.naming_test(bank)
    assert np.mean([r.correct for r in rm]) > np.mean([r.correct for r in rs]) + 0.3
    for r in rm + rs:
        assert r.category in {"correct", "semantic", "formal", "mixed", "unrelated", "nonword", "omission"}
        assert r.duration_s > 0
        assert r.fallback_from is None or isinstance(r.fallback_from, str)
        assert (r.s_eff, r.p_eff) == (mild if r in rm else severe).effective_weights(r.item)
        if r.category == "correct":
            assert r.word == r.item.word
        if r.category == "nonword":
            assert r.phonemes not in lex.all_prons and is_phonotactically_legal(r.phonemes, reference=lex.pron(r.item.word))
        if r.category in {"semantic", "formal", "mixed", "unrelated"}:
            assert r.word in lex.index and r.word != r.item.word
            assert len(wn.synsets(r.word, pos=wn.NOUN)) > 0 or r.word in {i.word for i in items}
    # item effects: effective weights shrink with difficulty, and switching beta off removes the effect
    hard, easy = max(bank, key=lambda i: i.difficulty), min(bank, key=lambda i: i.difficulty)
    assert mild.effective_weights(hard)[0] < mild.effective_weights(easy)[0]
    flat = Patient("flat", 0.025, 0.044, model, rz, item_beta=0.0)
    assert flat.effective_weights(hard) == flat.effective_weights(easy) == (0.025, 0.044)


@pytest.mark.slow
def test_nonwords_are_legal_nonwords_across_the_bank(model: DellSPModel) -> None:
    from patientmodel.lexicon import build_lexicon
    from patientmodel.realize import Realizer
    items = load_item_bank()
    lex = build_lexicon(extra_words=[i.word for i in items if i.pronounceable])
    rz = Realizer(lex)
    rng = np.random.default_rng(0)
    n = 0
    for it in [i for i in items if i.pronounceable and i.word in lex.index]:
        for _ in range(3):
            ph = rz.nonword(it.word, rng); n += 1
            assert ph not in lex.all_prons, (it.word, ph)
            assert is_phonotactically_legal(ph, reference=lex.pron(it.word)), (it.word, ph)
            assert ph != lex.pron(it.word)
    assert n > 1000


@pytest.mark.slow
def test_treatable_patient_learns_and_forgets(model: DellSPModel) -> None:
    from patientmodel.lexicon import build_lexicon
    from patientmodel.patient import Patient
    from patientmodel.realize import Realizer
    from patientmodel.treatment import TreatablePatient
    items = load_item_bank()
    lex = build_lexicon(extra_words=[i.word for i in items if i.pronounceable])
    rz = Realizer(lex)
    bank = [i for i in items if i.pronounceable and i.word in lex.index]
    rng = np.random.default_rng(5)
    test = list(rng.choice(bank, 40, replace=False)); practised, untreated = test[:20], test[20:]
    pt = TreatablePatient(Patient("severe", 0.005, 0.018, model, rz, rng=np.random.default_rng(2)))
    # baseline retrievability equals the snapshot's P(correct) for the item
    for it in test[:3]:
        assert abs(pt.retrievability(it, 0.0) - pt.baseline_probability(it)) < 1e-6
    base = np.mean([pt.retrievability(i, 0.0) for i in practised])
    for d in (0, 3, 7, 10):
        pt.session(practised, d, mode="attempt", presentations=3, lag=24)
    r17 = np.mean([pt.retrievability(i, 17) for i in practised])
    r60 = np.mean([pt.retrievability(i, 60) for i in practised])
    assert r17 > base + 0.2                       # practice helps
    assert r60 < r17                              # and fades
    assert r60 > base                             # but not below the untreated floor
    assert abs(np.mean([pt.retrievability(i, 17) for i in untreated]) - np.mean([pt.baseline_probability(i) for i in untreated])) < 1e-6
    # probes leave no trace; attempts do
    before = pt.retrievability(untreated[0], 20)
    pt.probe(untreated[0], 20); assert pt.retrievability(untreated[0], 20.01) == pytest.approx(before, abs=1e-9)
    pt.attempt(untreated[0], 20.02); assert pt.retrievability(untreated[0], 20.03) > before
    # responses on failure are realized errors, never the target
    resp = [pt.probe(i, 17) for i in practised]
    assert all((r.correct and r.word == r.item.word) or (not r.correct and r.word != r.item.word) for r in resp)
