"""Simulator validity: the four required behaviours plus item bank, error model, fatigue, learning."""
import csv

import numpy as np
import pytest

from core.config import load_clinical
from sim.config import ItemsConfig, load_sim
from sim.items import ItemSelector, load_items
from sim.patient import PatientParams, SimPatient
from sim.population import make_patients


@pytest.fixture(scope="module")
def env():
    cfg, clinical = load_sim(), load_clinical()
    return cfg, clinical, load_items(cfg.items)


def preset_patients(env, preset, n=100, split="test", seed=0):
    cfg, clinical, bank = env
    return [s.build(bank, cfg, clinical)[0] for s in make_patients(cfg, preset, split, n, seed)]


def mean_p(patients, tier, cue="none"):
    return float(np.mean([p.p_correct_tier(tier, cue) for p in patients]))


# ---- the four tests required by the project spec ----
@pytest.mark.parametrize("tier", [1, 2, 3])
def test_mild_outperforms_severe_at_equal_tier(env, tier):
    assert mean_p(preset_patients(env, "mild_anomic"), tier) > mean_p(preset_patients(env, "severe"), tier) + 0.2


def test_stronger_cues_raise_p_correct(env):
    cfg, clinical, _ = env
    for p in preset_patients(env, "moderate_broca", 20):
        for tier in (1, 2, 3):
            by_cue = {c: p.p_correct_tier(tier, c) for c in clinical.cue_strength}
            assert all(by_cue[c] >= by_cue["none"] for c in by_cue)
            assert by_cue["first_sound"] > by_cue["first_letter"] > by_cue["none"]  # phonemic hierarchy


def test_fatigue_lowers_p_correct_over_a_session(env):
    """Same item, same cue, no learning: p after 40 trials is lower than at the start."""
    cfg, clinical, bank = env
    cfg_nolearn = cfg.model_copy(update={"learning": cfg.learning.model_copy(update={"generalisation": 0.0})})
    for spec in make_patients(cfg, "moderate_broca", "test", 20, 0):
        params = PatientParams(**{**spec.params.to_dict(), "learn_rate": 0.0})
        p = SimPatient(params, bank, cfg_nolearn, clinical, np.random.default_rng(0))
        item = int(bank.indices_in_tier(2)[0])
        before = p.p_correct(item, "none")
        for _ in range(40):
            p.respond(item, "none")
        assert p.fatigue > 0 and p.p_correct(item, "none") < before


def test_same_seed_gives_identical_results(env):
    from agents.beta_bandit import BetaThompsonBandit
    from core.config import load_beta_bandit
    from sim.session import run_session
    cfg, clinical, bank = env

    def play(seed):
        spec = make_patients(cfg, "population", "train", 1, seed)[0]
        patient, selector = spec.build(bank, cfg, clinical)
        r = run_session(BetaThompsonBandit(load_beta_bandit(), clinical, seed), patient, selector, cfg, clinical)
        return [(t["word"], t["tier"], t["cue"], t["outcome"], t["error_type"], t["response_time"]) for t in r.trace]

    assert play(3) == play(3)
    assert play(3) != play(4)


# ---- population and split ----
def test_train_and_test_patients_are_disjoint(env):
    cfg, _, _ = env
    train = {tuple(s.params.to_dict().values()) for s in make_patients(cfg, "population", "train", 300, 0)}
    test = {tuple(s.params.to_dict().values()) for s in make_patients(cfg, "population", "test", 300, 0)}
    assert len(train) == 300 and len(test) == 300 and not train & test


def test_population_is_diverse_and_within_bounds(env):
    cfg, _, _ = env
    specs = make_patients(cfg, "population", "train", 500, 0)
    for name, d in cfg.population.items():
        v = np.array([getattr(s.params, name) for s in specs])
        assert v.min() >= d.low and v.max() <= d.high
        assert v.std() > 0.25 * d.sd   # genuinely randomised, not collapsed onto the mean


# ---- error model ----
@pytest.mark.parametrize("preset,dominant", [
    ("moderate_broca", "sound"), ("severe", "no_response"), ("mild_anomic", "no_response"),
])
def test_error_mix_matches_aphasia_type(env, preset, dominant):
    counts = {"sound": 0, "meaning": 0, "no_response": 0}
    for p in preset_patients(env, preset, 30):
        item = int(np.argmax(p.bank.difficulty))   # hardest item, so even mild patients fail often
        for _ in range(100):
            r = p.respond(item, "none")
            if not r.correct:
                counts[r.error_type] += 1
    assert max(counts, key=counts.get) == dominant
    assert sum(counts.values()) > 300


def test_broca_makes_more_sound_errors_than_anomic(env):
    def sound_share(preset):
        n_sound = n_err = 0
        for p in preset_patients(env, preset, 30):
            item = int(np.argmax(p.bank.difficulty))
            for _ in range(100):
                r = p.respond(item, "none")
                n_err += not r.correct
                n_sound += r.error_type == "sound"
        return n_sound / n_err
    assert sound_share("moderate_broca") > 2 * sound_share("mild_anomic")


# ---- fatigue recovery ----
def test_fatigue_recovers_during_success_runs(env):
    cfg, clinical, bank = env
    params = PatientParams(ability=6.0, phon_resp=1, sem_resp=1, phon_impair=.3, sem_impair=.3, omission_bias=.4,
                           fatigue_rate=0.0, fatigue_recovery=0.1, learn_rate=0.0)
    p = SimPatient(params, bank, cfg, clinical, np.random.default_rng(0))
    p.fatigue = 1.0
    for _ in range(10):
        p.respond(int(bank.indices_in_tier(1)[0]), "none")
    assert p.fatigue == pytest.approx(0.9 ** 10, rel=1e-6)


# ---- learning ----
def test_practice_raises_latent_retrieval(env):
    cfg, clinical, bank = env
    p = preset_patients(env, "moderate_broca", 1)[0]
    item = int(bank.indices_in_tier(2)[0])
    before_item, before_mean = p.retrieval_probabilities()[item], p.mean_retrieval()
    for _ in range(30):
        p.respond(item, "none")
    assert p.retrieval_probabilities()[item] > before_item and p.mean_retrieval() > before_mean


@pytest.mark.parametrize("peak", [0.5, 0.7, 0.85])
def test_expected_learning_peaks_at_configured_value(env, peak):
    cfg, clinical, bank = env
    c = cfg.model_copy(update={"learning": cfg.learning.model_copy(update={"peak": peak})})
    p = SimPatient(make_patients(c, "population", "train", 1, 0)[0].params, bank, c, clinical, np.random.default_rng(0))
    grid = np.linspace(0.01, 0.99, 981)
    g = p.expected_gain(grid)
    assert grid[np.argmax(g)] == pytest.approx(peak, abs=0.002)
    assert g.max() == pytest.approx(1.0)


def test_weaker_cues_teach_more(env):
    cfg, clinical, bank = env
    def gain(cue):
        p = preset_patients(env, "mild_anomic", 1)[0]
        item = int(bank.indices_in_tier(3)[0])
        # Force a success at the same p by comparing the deterministic expected update directly.
        prob = p.p_correct(item, "none")
        factor = 1 - cfg.learning.cue_discount * clinical.cue_strength[cue] / max(clinical.cue_strength.values())
        return p.params.learn_rate * p.expected_gain(prob) / prob * factor
    assert gain("none") > gain("first_letter") > gain("first_sound")


# ---- item bank ----
def test_placeholder_bank_difficulty_rises_with_tier(env):
    _, _, bank = env
    means = [bank.difficulty[bank.tier == t].mean() for t in (1, 2, 3)]
    assert means[0] < means[1] < means[2]
    assert len(bank) == 60 and all((bank.tier == t).sum() == 20 for t in (1, 2, 3))


def test_item_csv_schema_is_validated(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("word,frequency,tier\ncat,4.8,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_items(ItemsConfig(path=str(bad), beta_frequency=1, beta_length=1))
    one_tier = tmp_path / "one_tier.csv"
    with open(one_tier, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "image_file", "frequency", "length", "tier"])
        w.writerow(["cat", "cat.png", 4.8, 3, 1])
    with pytest.raises(ValueError, match="every tier"):
        load_items(ItemsConfig(path=str(one_tier), beta_frequency=1, beta_length=1))


def test_item_selector_shows_unseen_items_first(env):
    _, _, bank = env
    sel = ItemSelector(bank, np.random.default_rng(0))
    first_pass = [sel.next(2) for _ in range(20)]
    assert sorted(first_pass) == sorted(bank.indices_in_tier(2).tolist())   # no repeats before all seen
    assert sel.next(2) == first_pass[0]                                     # then least recently seen
