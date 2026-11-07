"""Gymnasium env: API compliance, masks, consistency with the session runner, no reward hacking."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from agents.baselines import StaircasePolicy
from core.masks import ACTION_CUES, ACTION_TIERS, action_index
from core.schema import Outcome
from envs.aphasia_env import AphasiaEnv
from sim.population import make_patients
from sim.session import run_session


@pytest.fixture(scope="module")
def env():
    return AphasiaEnv()


def test_gymnasium_api_compliance():
    check_env(AphasiaEnv(strict_masks=False), skip_render_check=True)


def test_reset_is_reproducible_with_seed(env):
    o1, i1 = env.reset(seed=11)
    o2, i2 = env.reset(seed=11)
    assert i1 == i2 and np.array_equal(o1, o2)


def test_masks_respected_over_random_episodes(env):
    rng = np.random.default_rng(0)
    for ep in range(20):
        env.reset(seed=ep)
        done = False
        while not done:
            mask = env.action_masks()
            assert all(abs(t - env.tier) <= 1 for t in ACTION_TIERS[mask])
            last_correct = env.history[-1]["outcome"] if env.history else True
            if last_correct:   # a cue may only be offered right after an error
                assert all(ACTION_CUES[i] == "none" for i in np.flatnonzero(mask))
            a = int(rng.choice(np.flatnonzero(mask)))
            _, _, done, _, info = env.step(a)
        assert 1 <= len(info["result"].trace) <= 40
    with pytest.raises(ValueError):
        env.reset(seed=0)
        env.step(action_index(3, "none"))   # tier 1 -> 3 is masked


def test_env_reproduces_session_runner_exactly():
    """Same patient, same policy: the env and run_session produce identical traces."""
    env = AphasiaEnv(cohort="moderate_broca", split="test")
    spec = make_patients(env.sim_cfg, "moderate_broca", "test", 1, 0, start=5)[0]
    patient, selector = spec.build(env.bank, env.sim_cfg, env.clinical)
    ref = run_session(StaircasePolicy(), patient, selector, env.sim_cfg, env.clinical).trace

    env.reset(options={"patient_index": 5})
    pol, done = StaircasePolicy(), False
    while not done:
        state = env._state()
        a = pol.act(state, env.action_masks())
        _, _, done, _, info = env.step(action_index(a.tier, a.cue))
        last = env.history[-1]
        pol.update(state, a, Outcome(correct=last["outcome"], error_type=last["error_type"],
                                     response_time=last["response_time"]))
    assert info["result"].trace == ref


def _mean_return(policy_fn, n=40):
    env = AphasiaEnv(cohort="mild_anomic", split="val")
    returns = []
    for k in range(n):
        env.reset(options={"patient_index": k})
        total, done, t = 0.0, False, 0
        while not done:
            _, r, done, _, _ = env.step(policy_fn(env, t))
            total += r
            t += 1
        returns.append(total)
    return float(np.mean(returns))


def test_tier_oscillation_does_not_beat_holding_a_tier():
    """Env-level reward-hacking check: alternating tiers 1<->2 every trial must not out-score simply
    holding tier 2 on mild patients."""
    def hold(env, t):
        return action_index(min(env.tier + 1, 2), "none")
    def oscillate(env, t):
        return action_index(2 if env.tier == 1 else 1, "none")
    r_hold, r_osc = _mean_return(hold), _mean_return(oscillate)
    print(f"mean return: hold tier 2 = {r_hold:.2f}, oscillate 1<->2 = {r_osc:.2f}")
    assert r_osc <= r_hold


def _episode_terms(variant, n=5):
    from core.reward import TERMS, load_reward
    env = AphasiaEnv(cohort="mild_anomic", split="val", reward_cfg=load_reward(variant=variant))
    totals = dict.fromkeys(TERMS, 0.0)
    for k in range(n):
        env.reset(options={"patient_index": k})
        done = False
        while not done:
            a = int(np.flatnonzero(env.action_masks())[0])
            _, _, done, _, info = env.step(a)
            for key, v in info["reward_terms"].items():
                totals[key] += v
    return totals


def test_privileged_terms_only_in_privileged_variant():
    priv, proxy = _episode_terms("privileged"), _episode_terms("full")
    assert priv["control"] < 0 and priv["learning"] > 0 and priv["probe_p"] > 0 and priv["band"] == 0
    assert proxy["control"] == proxy["learning"] == proxy["probe_p"] == 0 and proxy["band"] > 0
