"""Trained PPO policy: loads, respects masks through a full session, and is deterministic."""
import pytest

from core.config import load_clinical, resolve
from sim.config import load_sim
from sim.items import load_items
from sim.population import make_patients
from sim.session import run_session

MODEL = "models/ppo_seed0.zip"
pytestmark = pytest.mark.skipif(not resolve(MODEL).exists(), reason="trained model not present")


def test_ppo_plays_masked_sessions_deterministically():
    from agents.ppo import PPOPolicy
    from envs.aphasia_env import load_env_config
    cfg, clinical = load_sim(), load_clinical()
    bank = load_items(cfg.items)
    pol = PPOPolicy(MODEL, clinical, load_env_config())
    def play(k):
        spec = make_patients(cfg, "moderate_broca", "test", 1, 0, start=k)[0]
        patient, selector = spec.build(bank, cfg, clinical)
        return [(t["tier"], t["cue"]) for t in run_session(pol, patient, selector, cfg, clinical).trace]
    a = play(3)   # run_session raises if a masked action is returned
    assert a == play(3) and len(a) >= 1
