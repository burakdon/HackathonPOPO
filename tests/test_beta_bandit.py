"""Ported Beta-Thompson bandit: constraint compliance, the current_tier bug fix, determinism."""
import numpy as np

from agents.base import Policy
from agents.beta_bandit import BetaThompsonBandit
from core.masks import action_index, action_mask, is_allowed
from core.schema import Action, Outcome, State
from tests.conftest import trial


def make(bandit_cfg, clinical, seed=0):
    return BetaThompsonBandit(bandit_cfg, clinical, seed=seed)


def state(tier=2, last=None, n=0):
    return State(last_trials=last or [], current_tier=tier, trial_count=n, session_minutes=0.0)


def test_implements_policy_protocol(bandit_cfg, clinical):
    assert isinstance(make(bandit_cfg, clinical), Policy)


def test_never_chooses_masked_action(bandit_cfg, clinical):
    b = make(bandit_cfg, clinical)
    rng = np.random.default_rng(1)
    for i in range(500):
        tier = int(rng.integers(1, 4))
        last = [trial(outcome=bool(rng.random() < 0.5))]
        mask = action_mask(tier, ["none", "semantic"], clinical.max_tier_step)
        s = state(tier, last, i)
        a = b.act(s, mask)
        assert is_allowed(a, mask)
        correct = bool(rng.random() < 0.7)
        b.update(s, a, Outcome(correct=correct, error_type="correct" if correct else "meaning",
                               response_time=1.0))


def test_reads_current_tier_from_state_after_override(bandit_cfg, clinical):
    """Scaffold bug: it planned from its own last choice. After a clinician moves the patient to
    tier 3, the agent must never jump to tier 1."""
    b = make(bandit_cfg, clinical)
    b.act(state(1), action_mask(1, clinical.approved_cues, 1))  # agent at tier 1
    for _ in range(200):
        a = b.act(state(3), action_mask(3, clinical.approved_cues, 1))
        assert a.tier in (2, 3)


def test_cue_only_after_error(bandit_cfg, clinical):
    b = make(bandit_cfg, clinical)
    mask = action_mask(2, clinical.approved_cues, 1)
    cues_after_success = {b.act(state(2, [trial(True)]), mask).cue for _ in range(200)}
    cues_at_start = {b.act(state(2), mask).cue for _ in range(200)}
    assert cues_after_success == {"none"} and cues_at_start == {"none"}
    cues_after_error = {b.act(state(2, [trial(False)]), mask).cue for _ in range(500)}
    assert len(cues_after_error) > 1


def test_update_moves_posterior_in_right_direction(bandit_cfg, clinical):
    b = make(bandit_cfg, clinical)
    i = action_index(2, "none")
    before = b.posterior_mean()[i]
    for _ in range(5):
        b.update(state(), Action(tier=2, cue="none"), Outcome(correct=True, error_type="correct", response_time=1.0))
    up = b.posterior_mean()[i]
    for _ in range(10):
        b.update(state(), Action(tier=2, cue="none"), Outcome(correct=False, error_type="sound", response_time=1.0))
    down = b.posterior_mean()[i]
    assert up > before > down


def test_discount_decays_unpulled_arms_toward_prior(bandit_cfg, clinical):
    b = make(bandit_cfg, clinical)
    j = action_index(3, "none")
    for _ in range(10):
        b.update(state(), Action(tier=3, cue="none"), Outcome(correct=True, error_type="correct", response_time=1.0))
    excess = b.alpha[j] - b.prior_alpha[j]
    for _ in range(20):
        b.update(state(), Action(tier=1, cue="none"), Outcome(correct=True, error_type="correct", response_time=1.0))
    assert np.isclose(b.alpha[j] - b.prior_alpha[j], excess * bandit_cfg.discount ** 20)


def test_same_seed_same_actions(bandit_cfg, clinical):
    def run(seed):
        b = make(bandit_cfg, clinical, seed)
        s, out = state(2), []
        for k in range(40):
            a = b.act(s, action_mask(s.current_tier, clinical.approved_cues, 1))
            correct = k % 3 != 0
            b.update(s, a, Outcome(correct=correct, error_type="correct" if correct else "sound", response_time=1.0))
            s = state(a.tier, [trial(correct, a.tier, a.cue)], k + 1)
            out.append((a.tier, a.cue))
        return out
    assert run(7) == run(7)
    assert run(7) != run(8)
