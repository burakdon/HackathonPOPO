"""Item-level learning model: forgetting, spacing, retrieval-practice effects, and the fit harness."""
from __future__ import annotations

import numpy as np

from patientmodel.learning import DAY, SECOND, ItemMemory, LearningModel, LearningParams
from patientmodel.learning_fit import CONDITIONS, SubjectParams, load_observed, simulate_subject
from patientmodel.treatment import fitted_learning_params


def test_baseline_inverts_probability() -> None:
    p = LearningParams()
    for p0 in (0.02, 0.3, 0.8):
        beta = LearningModel.beta_for_probability(p0, p)
        assert abs(LearningModel(p).probability(ItemMemory(beta=beta), 0.0) - p0) < 1e-9


def test_practice_raises_then_forgetting_lowers() -> None:
    m = LearningModel(LearningParams())
    mem = ItemMemory(beta=LearningModel.beta_for_probability(0.05, m.p))
    p0 = m.probability(mem, 0.0)
    m.study(mem, 0.0)
    p_soon = m.probability(mem, 10 * SECOND)
    p_day = m.probability(mem, 1 * DAY)
    p_month = m.probability(mem, 30 * DAY)
    assert p_soon > p_day > p_month > p0


def test_spacing_effect_emerges_from_decay() -> None:
    """Two study events: the longer gap between them gives the more durable trace at a delayed test."""
    m = LearningModel(LearningParams())
    def retention(gap_seconds: float) -> float:
        mem = ItemMemory(beta=LearningModel.beta_for_probability(0.05, m.p))
        m.study(mem, 0.0); m.study(mem, gap_seconds * SECOND)
        return m.probability(mem, 7 * DAY)
    assert retention(24 * 12) > retention(1 * 12)


def test_successful_retrieval_beats_study_and_failure() -> None:
    p = LearningParams(w_success=1.0, w_study=0.6, w_fail_feedback=0.6)
    m = LearningModel(p)
    def after(weight: float) -> float:
        mem = ItemMemory(beta=LearningModel.beta_for_probability(0.3, p))
        m.record(mem, 0.0, weight)
        return m.probability(mem, 7 * DAY)
    assert after(p.w_success) > after(p.w_study) == after(p.w_fail_feedback)


def test_tests_do_not_add_traces() -> None:
    m = LearningModel(LearningParams())
    mem = ItemMemory(beta=0.0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        m.test(mem, 1.0, rng)
    assert mem.times == []


def test_middleton_data_loads_with_expected_shape() -> None:
    obs = load_observed()
    assert len(obs) == 4 * (4 * 2 + 2 * 3)       # 4 subjects x (4 conditions x 2 tests + 2 RP conditions x 3 days)
    for (subj, cond, measure), (k, n) in obs.items():
        assert 0 <= k <= n and n == 48


def test_protocol_simulation_reproduces_effect_directions() -> None:
    """With the parameters fitted to Middleton 2019 (results/learning_fit.json; defaults if absent)
    the simulated protocol shows the four published effects."""
    params = fitted_learning_params()
    pred = simulate_subject(params, SubjectParams(beta=LearningModel.beta_for_probability(0.03, params), learn_scale=1.0), n_items=600, seed=1)
    rp_sp, rp_ma = pred[(("retrievalpractice", "Spaced"), "retention")], pred[(("retrievalpractice", "Massed"), "retention")]
    rep_sp, rep_ma = pred[(("repetition", "Spaced"), "retention")], pred[(("repetition", "Massed"), "retention")]
    assert rp_sp > rp_ma and rep_sp > rep_ma            # spacing effect
    assert rp_sp > rep_sp and rp_ma > rep_ma            # retrieval-practice effect
    assert pred[(("retrievalpractice", "Spaced"), "followup")] < rp_sp   # forgetting between 7 and 28 days
    assert pred[(("retrievalpractice", "Spaced"), "day1")] < pred[(("retrievalpractice", "Spaced"), "day3")]
