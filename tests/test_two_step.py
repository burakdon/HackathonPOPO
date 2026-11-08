"""Two-step patient model: closed form, error types from the failing step, cue x error interaction."""
import numpy as np
import pytest

from core.config import load_clinical
from sim.config import load_sim
from sim.items import load_items
from sim.population import make_patients
from sim.two_step import TwoStepParams, TwoStepPatient

CFG = "configs/sim_two_step.yaml"


@pytest.fixture(scope="module")
def setup():
    cfg = load_sim(CFG)
    return cfg, load_clinical(), load_items(cfg.items)


def patient(setup, seed=0, **kw):
    cfg, clinical, bank = setup
    base = dict(sem_strength=2.0, phon_strength=2.0, sem_resp=1.0, phon_resp=1.0, omission_bias=0.4, tot_rate=0.3,
                fatigue_rate=0.0, fatigue_recovery=0.1, learn_rate=0.0)
    return TwoStepPatient(TwoStepParams(**{**base, **kw}), bank, cfg, clinical, np.random.default_rng(seed))


def test_default_simulator_is_unchanged():
    assert load_sim().model == "logistic"


def test_population_builds_two_step_patients_deterministically(setup):
    cfg, clinical, bank = setup
    a = make_patients(cfg, "moderate_broca", "test", 3, 0)
    b = make_patients(cfg, "moderate_broca", "test", 3, 0)
    assert [s.params for s in a] == [s.params for s in b]
    p, _ = a[0].build(bank, cfg, clinical)
    assert isinstance(p, TwoStepPatient)


def test_error_distribution_is_a_distribution_and_matches_sampling(setup):
    p = patient(setup, sem_strength=0.5, phon_strength=0.8)
    d = p.error_distribution(0, "none")
    assert abs(sum(d.values()) - 1) < 1e-9
    n, counts = 20000, {k: 0 for k in d}
    for _ in range(n):              # no learning, no fatigue: every trial has the same distribution
        counts[p.respond(0, "none").error_type] += 1
    for k in d:
        assert abs(counts[k] / n - d[k]) < 0.02, (k, counts[k] / n, d[k])


def test_weak_step_decides_error_type_and_which_cue_helps(setup):
    phon_weak = patient(setup, sem_strength=3.0, phon_strength=0.0)
    sem_weak = patient(setup, sem_strength=0.0, phon_strength=3.0)
    i = int(setup[2].indices_in_tier(2)[0])
    dp, ds = phon_weak.error_distribution(i, "none"), sem_weak.error_distribution(i, "none")
    assert dp["sound"] > dp["meaning"] and ds["meaning"] > ds["sound"]
    gain = lambda pt, c: pt.p_correct(i, c) - pt.p_correct(i, "none")
    assert gain(phon_weak, "first_sound") > gain(phon_weak, "semantic")
    assert gain(sem_weak, "semantic") > gain(sem_weak, "first_sound")


def test_common_random_numbers_four_draws_per_trial(setup):
    a, b = patient(setup, seed=5, phon_strength=-2.0), patient(setup, seed=5, phon_strength=4.0)
    a.respond(0, "none")
    b.respond(0, "none")
    assert a.rng.random() == b.rng.random()      # same stream position whatever the outcome


def test_learning_raises_retrieval_and_initial_is_frozen(setup):
    p = patient(setup, learn_rate=0.3)
    start = p.mean_retrieval()
    for _ in range(200):
        p.respond(3, "none")
    assert p.mean_retrieval() > start
    assert np.isclose(p.initial_retrieval.mean(), start)


def test_environment_runs_a_full_episode_on_two_step_patients(setup):
    from envs.aphasia_env import AphasiaEnv
    env = AphasiaEnv(cohort="population", split="val", sim_cfg=setup[0])
    env.reset(options={"patient_index": 0})
    done, n = False, 0
    while not done:
        a = int(np.flatnonzero(env.action_masks())[0])
        _, _, done, _, info = env.step(a)
        n += 1
    assert "learning_gain" in info and n > 0
