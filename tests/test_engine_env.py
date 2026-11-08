"""The engine therapy environment: shapes, masks, reward telescoping, baselines, reproducibility."""
import numpy as np
import pytest

from neurocue.baselines import BASELINES
from neurocue.env import ACTIONS, N_ACTIONS, EngineEnvConfig, EngineTherapyEnv, action_index


@pytest.fixture(scope="module")
def env():
    cfg = EngineEnvConfig(n_items=6, schedule_days=(0.0, 3.0), trials_per_session=10, test_days=(10.0, 31.0), probe_trials=300)
    return EngineTherapyEnv(cfg, split="test", root_seed=1)


def test_reset_and_spaces(env):
    obs, info = env.reset(options={"patient_index": 0})
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    assert len(env.items) == 6 and all(0.0 <= env.tp.baseline_probability(it) <= 1.0 for it in env.items)
    assert env.action_space.n == N_ACTIONS == 16


def test_repeat_masked_on_first_trial_then_allowed(env):
    env.reset(options={"patient_index": 0})
    m = env.action_masks()
    assert not m[action_index("repeat", None)] and m[action_index("due", None)]
    env.step(action_index("due", None))
    assert env.action_masks()[action_index("repeat", "whole_word")]
    with pytest.raises(ValueError):
        env.reset(options={"patient_index": 0}); env.step(action_index("repeat", None))


def test_privileged_reward_telescopes_to_gain(env):
    _, info = env.reset(options={"patient_index": 2})
    total, done = 0.0, False
    while not done:
        a = action_index("due", None) if env.trial_in_session % 2 == 0 else action_index("weakest", "phonemic_initial")
        _, r, done, _, info = env.step(a)
        total += r
    assert abs(total - 100.0 * info["gain"][10.0]) < 1e-6
    assert info["retention"][10.0] >= info["retention"][31.0]          # forgetting between the tests


def test_same_patient_same_stream(env):
    def play(k):
        env.reset(options={"patient_index": k}); out = []
        for _ in range(8):
            _, r, _, _, info = env.step(action_index("due", None)); out.append((info["response"]["correct"], round(r, 6)))
        return out, (env.tp.snapshot.s, env.tp.snapshot.p)
    a, pa = play(3); b, pb = play(3); c, pc = play(4)
    assert a == b and pa == pb and pa != pc


def test_baselines_run_and_spacing_beats_massing(env):
    gains = {}
    for name in ("spaced_plain", "massed_plain", "spaced_hierarchy", "errorless", "weakest_hierarchy", "random"):
        pol = BASELINES[name](); rng = np.random.default_rng(0); gs = []
        for k in range(6):
            _, info = env.reset(options={"patient_index": k}); done = False
            while not done:
                _, _, done, _, info = env.step(pol.act(env.observable(), env.action_masks(), rng))
            gs.append(info["gain"][10.0])
        gains[name] = float(np.mean(gs))
    assert all(np.isfinite(v) for v in gains.values())
    assert gains["spaced_plain"] > gains["massed_plain"], gains
