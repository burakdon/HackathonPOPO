"""POST /learn_from_overrides: one DITTO-style round on a session's overrides, hot-swapping only
that session's policy, within a few seconds on CPU."""
import pytest
from fastapi.testclient import TestClient

from core.config import resolve
from service.app import create_app

from core.config import load_service

LIVE_MODEL = load_service().ppo_model
pytestmark = pytest.mark.skipif(not resolve(LIVE_MODEL).exists(), reason="model missing")


def realistic_sound_error_states(n: int) -> list[dict]:
    """Decision states the live policy actually visits right after a sound error, from validation
    patients outside the service's anchor pool (patients 0-29)."""
    from sb3_contrib import MaskablePPO
    from envs.aphasia_env import AphasiaEnv
    model = MaskablePPO.load(str(resolve(LIVE_MODEL)), device="cpu")
    env, out, k = AphasiaEnv(cohort="moderate_broca", split="val"), [], 100
    while len(out) < n:
        obs, _ = env.reset(options={"patient_index": k})
        done = False
        while not done and len(out) < n:
            state = env._state()
            if state.last_trials and state.last_trials[-1].error_type == "sound":
                out.append(state.model_dump())
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _, done, _, _ = env.step(int(a))
        k += 1
    return out


def test_learn_from_overrides_adopts_preference_and_hot_swaps_only_that_session(tmp_path):
    c = TestClient(create_app(overrides_path=tmp_path / "o.jsonl", agent="ppo"))
    assert c.post("/learn_from_overrides", json={"session_id": "demo"}).status_code == 400   # nothing logged yet
    states = realistic_sound_error_states(12)
    for st in states[:10]:
        st = {**st, "session_id": "demo"}
        agent_action = c.post("/act", json=st).json()
        c.post("/override", json={"state": st, "agent_action": agent_action,     # clinician keeps the tier
                                  "clinician_action": {"tier": agent_action["tier"], "cue": "first_sound"}})
    r = c.post("/learn_from_overrides", json={"session_id": "demo"})
    assert r.status_code == 200
    res = r.json()
    print(res)
    assert res["n_overrides"] == 10 and res["seconds"] < 10
    change = next(x for x in res["by_state_type"] if x["state_type"] == "after a sound error")
    assert change["clinician_cue"] == "first_sound" and change["p_after"] > change["p_before"] + 0.3
    assert res["other_states_unchanged"] > 0.95
    held_out = [{**st, "session_id": "demo"} for st in states[10:]]
    assert all(c.post("/act", json=st).json()["cue"] == "first_sound" for st in held_out)   # swapped session
    fresh = TestClient(create_app(overrides_path=tmp_path / "o2.jsonl", agent="ppo"))
    for st in held_out:                                                                   # other sessions untouched
        other = {**st, "session_id": "other"}
        assert c.post("/act", json=other).json() == fresh.post("/act", json=other).json()


def test_learn_requires_ppo_agent(tmp_path):
    c = TestClient(create_app(overrides_path=tmp_path / "o.jsonl", agent="logistic_bandit"))
    assert c.post("/learn_from_overrides", json={"session_id": "x"}).status_code == 409
