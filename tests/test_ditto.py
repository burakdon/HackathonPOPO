"""DITTO-style tuning of the PPO policy: demonstrations gain probability, the reference is frozen,
and accepted-choice anchors keep the policy unchanged elsewhere."""
import numpy as np
import pytest

from core.config import resolve

MODEL = "models/ppo_seed0.zip"
pytestmark = pytest.mark.skipif(not resolve(MODEL).exists(), reason="trained model not present")


@pytest.fixture(scope="module")
def setup():
    from sb3_contrib import MaskablePPO
    from envs.aphasia_env import AphasiaEnv
    from train.ditto_ppo import collect_states
    ref = MaskablePPO.load(str(resolve(MODEL)), device="cpu")
    states = collect_states(ref, AphasiaEnv(cohort="population", split="val"), range(300, 306))
    return ref, states


def _demos(ref, states, n=6):
    from agents.ditto import Demo, action_probs
    demos, rest = [], []
    for s in states:
        p = action_probs(ref, s["obs"][None], s["mask"][None])[0]
        allowed = np.flatnonzero(s["mask"])
        if len(demos) < n and len(allowed) > 1:
            demos.append(Demo(s["obs"], s["mask"], int(allowed[np.argsort(p[allowed])[0]])))  # least likely
        else:
            rest.append(s)
    return demos, rest


def test_demonstrations_gain_probability_and_reference_is_frozen(setup):
    from agents.ditto import action_probs, ditto
    ref, states = setup
    demos, _ = _demos(ref, states)
    obs, masks = np.stack([d.obs for d in demos]), np.stack([d.mask for d in demos])
    acts = [d.action for d in demos]
    before = action_probs(ref, obs, masks)[np.arange(len(demos)), acts]
    tuned, log = ditto(ref, demos, rounds=2, samples_per_state=8, beta=0.5, lr=1e-3, steps_per_round=30, seed=0)
    after = action_probs(tuned, obs, masks)[np.arange(len(demos)), acts]
    assert after.mean() > before.mean() + 0.3
    assert np.allclose(action_probs(ref, obs, masks)[np.arange(len(demos)), acts], before)   # ref untouched
    assert len(log) == 2


def test_anchors_keep_accepted_states_unchanged(setup):
    from agents.ditto import action_probs, ditto
    ref, states = setup
    demos, rest = _demos(ref, states)
    anchors = [(s["obs"], s["mask"]) for s in rest]
    obs, masks = np.stack([o for o, _ in anchors]), np.stack([m for _, m in anchors])
    tv = lambda m: 0.5 * np.abs(action_probs(m, obs, masks) - action_probs(ref, obs, masks)).sum(1).mean()
    plain, _ = ditto(ref, demos, 2, 8, 0.5, 1e-3, 30, 0)
    anchored, _ = ditto(ref, demos, 2, 8, 0.5, 1e-3, 30, 0, anchors=anchors, anchor_weight=5.0)
    assert tv(anchored) < tv(plain) and tv(anchored) < 0.05
