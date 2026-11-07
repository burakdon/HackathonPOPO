"""Contextual logistic bandit, shared selection rule, and baselines."""
import numpy as np
import pytest

from agents.base import Policy
from agents.baselines import FixedPolicy, StaircasePolicy
from agents.logistic_bandit import N_FEATURES, LogisticThompsonBandit
from agents.selection import select_action
from core.config import load_clinical, load_logistic_bandit
from core.masks import ACTIONS, action_index, action_mask, is_allowed
from core.schema import Action, Outcome, State
from tests.conftest import trial


@pytest.fixture
def lcfg():
    return load_logistic_bandit()


def st(tier=1, last=None, n=0):
    return State(last_trials=last or [], current_tier=tier, trial_count=n, session_minutes=0.0)


def outcome(correct, err="sound"):
    return Outcome(correct=correct, error_type="correct" if correct else err, response_time=2.0)


def test_policies_implement_protocol(lcfg, clinical):
    for p in (LogisticThompsonBandit(lcfg, clinical, 0), FixedPolicy(), StaircasePolicy()):
        assert isinstance(p, Policy)


def test_feature_matrix_shape_and_interactions(lcfg, clinical):
    b = LogisticThompsonBandit(lcfg, clinical, 0)
    X = b.features(st(2, [trial(False) | {"error_type": "sound"}]))
    assert X.shape == (15, N_FEATURES)
    fs = X[action_index(2, "first_sound")]
    sem = X[action_index(2, "semantic")]
    none = X[action_index(2, "none")]
    from agents.logistic_bandit import FEATURES
    assert fs[FEATURES.index("last_sound_x_phonemic")] == 1 and sem[FEATURES.index("last_sound_x_semantic")] == 1
    assert none[FEATURES.index("last_sound_x_phonemic")] == 0


def test_update_moves_prediction_in_right_direction(lcfg, clinical):
    b = LogisticThompsonBandit(lcfg, clinical, 0)
    i = action_index(2, "none")
    p0 = b.predict(st(2))[i]
    for _ in range(10):
        b.update(st(2), Action(tier=2, cue="none"), outcome(True))
    p1 = b.predict(st(2))[i]
    for _ in range(30):
        b.update(st(2), Action(tier=2, cue="none"), outcome(False))
    assert p1 > p0 > b.predict(st(2))[i]


def test_recovers_true_probability_on_stationary_patient(lcfg, clinical):
    """500 trials on a fixed arm with p=0.7 (empty context): posterior mean within 0.05."""
    cfg = lcfg.model_copy(update={"process_noise": 0.0})
    b = LogisticThompsonBandit(cfg, clinical, 0)
    rng = np.random.default_rng(1)
    for _ in range(500):
        b.update(st(2), Action(tier=2, cue="first_letter"), outcome(bool(rng.random() < 0.7)))
    assert abs(b.predict(st(2))[action_index(2, "first_letter")] - 0.7) < 0.05


def test_never_chooses_masked_action(lcfg, clinical):
    b = LogisticThompsonBandit(lcfg, clinical, 0)
    rng = np.random.default_rng(2)
    for k in range(300):
        tier = int(rng.integers(1, 4))
        s = st(tier, [trial(bool(rng.random() < 0.5))], k)
        mask = action_mask(tier, ["none", "semantic"], 1)
        a = b.act(s, mask)
        assert is_allowed(a, mask)
        b.update(s, a, outcome(bool(rng.random() < 0.7)))


def test_deterministic_given_seed(lcfg, clinical):
    def run(seed):
        b, out = LogisticThompsonBandit(lcfg, clinical, seed), []
        s = st(1)
        for k in range(30):
            a = b.act(s, action_mask(s.current_tier, clinical.approved_cues, 1))
            c = k % 4 != 0
            b.update(s, a, outcome(c))
            s = st(a.tier, [trial(c, a.tier, a.cue)], k + 1)
            out.append((a.tier, a.cue))
        return out
    assert run(5) == run(5) and run(5) != run(6)


def test_progression_rule_prefers_harder_tier_then_weaker_cue():
    p = np.full(15, 0.5)
    allowed = np.ones(15, dtype=bool)
    strengths = np.array([{"none": 0, "first_letter": 1, "sentence_frame": 2, "semantic": 2, "first_sound": 3}[c]
                          for _, c in ACTIONS], dtype=float)
    p[action_index(1, "none")] = 0.85          # easy tier, exactly on target
    p[action_index(2, "first_sound")] = 0.82   # harder tier, strong cue, within tolerance
    p[action_index(2, "first_letter")] = 0.80  # harder tier, weaker cue, within tolerance
    i, progressive = select_action(p, allowed, strengths, 0.85, 0.06, 0.03, 0.01)
    assert progressive and ACTIONS[i] == (2, "first_letter")
    # Nothing within tolerance -> closest to target.
    p2 = np.full(15, 0.3); p2[action_index(1, "none")] = 0.7
    i2, prog2 = select_action(p2, allowed, strengths, 0.85, 0.06, 0.03, 0.01)
    assert not prog2 and ACTIONS[i2] == (1, "none")


def test_staircase_two_up_one_down(clinical):
    s = StaircasePolicy()
    mask = action_mask(2, clinical.approved_cues, 1)
    assert s.act(st(2, [trial(True, 2), trial(True, 2)]), mask).tier == 3
    assert s.act(st(2, [trial(True, 1), trial(True, 2)]), mask).tier == 2   # only one correct at this tier
    assert s.act(st(2, [trial(True, 2), trial(False, 2)]), mask).tier == 1


def test_fixed_policy_steps_toward_target_tier_within_mask(clinical):
    f = FixedPolicy(tier=3)
    assert f.act(st(1), action_mask(1, clinical.approved_cues, 1)).tier == 2
    assert f.act(st(2), action_mask(2, clinical.approved_cues, 1)).tier == 3
