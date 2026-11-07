"""Agent service: endpoint contracts, session isolation, override storage, determinism."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from tests.conftest import EXAMPLE_STATE, trial


@pytest.fixture
def overrides(tmp_path):
    return tmp_path / "overrides.jsonl"


@pytest.fixture(params=["ppo", "logistic_bandit", "beta_bandit"])
def client(overrides, request):
    """Every service test runs against both live agents."""
    return TestClient(create_app(overrides_path=overrides, agent=request.param))


def posterior(agent) -> np.ndarray:
    """Agent-agnostic snapshot of the learned state, for isolation checks."""
    return agent.mu.copy() if hasattr(agent, "mu") else agent.posterior_mean().copy()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "agent": client.app.state.agent, "active_sessions": 0}


def test_act_returns_valid_action_for_contract_example(client):
    r = client.post("/act", json=EXAMPLE_STATE)
    assert r.status_code == 200
    a = r.json()
    assert set(a) == {"tier", "cue", "rationale"}
    assert a["tier"] in (1, 2, 3) and abs(a["tier"] - EXAMPLE_STATE["current_tier"]) <= 1


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_act_respects_tier_step_mask(client, tier):
    for k in range(50):
        r = client.post("/act", json={**EXAMPLE_STATE, "current_tier": tier, "session_id": f"s{k}"})
        assert abs(r.json()["tier"] - tier) <= 1


def test_invalid_bodies_rejected_with_422(client):
    assert client.post("/act", json={**EXAMPLE_STATE, "current_tier": 5}).status_code == 422
    assert client.post("/act", json={"current_tier": 2}).status_code == 422
    bad_outcome = {"state": EXAMPLE_STATE, "action": {"tier": 2, "cue": "none"},
                   "outcome": {"correct": True, "error_type": "sound", "response_time": 1.0}}
    assert client.post("/update", json=bad_outcome).status_code == 422


def test_update_changes_only_its_own_session(client):
    if client.app.state.agent == "ppo":
        pytest.skip("the PPO agent keeps no learned per-session state")
    store = client.app.state.store
    a_state = {**EXAMPLE_STATE, "session_id": "A"}
    b_state = {**EXAMPLE_STATE, "session_id": "B"}
    client.post("/act", json=a_state)
    client.post("/act", json=b_state)
    b_before = posterior(store.get("B"))
    for _ in range(10):
        r = client.post("/update", json={
            "state": a_state, "action": {"tier": 2, "cue": "none"},
            "outcome": {"correct": True, "error_type": "correct", "response_time": 1.2}})
        assert r.status_code == 200 and r.json()["session_id"] == "A"
    assert not np.allclose(posterior(store.get("A")), b_before)
    assert np.allclose(posterior(store.get("B")), b_before)
    assert client.get("/health").json()["active_sessions"] == 2


def test_missing_session_id_uses_default_session(client):
    client.post("/act", json=EXAMPLE_STATE)
    assert "default" in client.app.state.store._agents


def test_override_appends_jsonl(client, overrides):
    body = {"state": {**EXAMPLE_STATE, "session_id": "A"},
            "agent_action": {"tier": 2, "cue": "semantic"},
            "clinician_action": {"tier": 2, "cue": "first_sound"},
            "note": "phonemic responder"}
    for n in (1, 2):
        r = client.post("/override", json=body)
        assert r.json() == {"status": "stored", "session_id": "A", "n_overrides": n}
    rows = [json.loads(line) for line in overrides.read_text().splitlines()]
    assert rows[1]["clinician_action"]["cue"] == "first_sound"
    assert rows[1]["state"]["last_trials"][0]["word"] == "cat"


@pytest.mark.parametrize("agent", ["ppo", "logistic_bandit", "beta_bandit"])
def test_full_session_replays_identically(overrides, agent):
    """Same session_id + same requests -> same actions, across two independent app instances."""
    def play():
        c = TestClient(create_app(overrides_path=overrides, agent=agent))
        tier, hist, actions = 2, [], []
        for k in range(40):
            st = {"session_id": "replay", "last_trials": hist[-10:], "current_tier": tier,
                  "trial_count": k, "session_minutes": k * 0.15}
            a = c.post("/act", json=st).json()
            correct = (k * 7) % 5 != 0
            c.post("/update", json={"state": st, "action": a, "outcome": {
                "correct": correct, "error_type": "correct" if correct else "meaning", "response_time": 2.0}})
            hist.append(trial(correct, a["tier"], a["cue"], word=f"w{k}") |
                        {"error_type": "correct" if correct else "meaning"})
            tier = a["tier"]
            actions.append((a["tier"], a["cue"]))
        return actions
    assert play() == play()


@pytest.mark.parametrize("origin,allowed", [
    ("http://localhost:5173", True),
    ("http://192.168.1.23:5173", True),     # another laptop on the same Wi-Fi
    ("http://10.0.0.7:5173", True),
    ("http://172.20.4.1:5173", True),
    ("http://evil.example.com", False),
    ("http://192.168.1.23:8080", False),    # not the Vite dev port
    ("http://172.32.0.1:5173", False),      # outside the 172.16/12 private block
])
def test_cors_preflight(client, origin, allowed):
    r = client.options("/act", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                        "Access-Control-Request-Headers": "content-type"})
    assert (r.headers.get("access-control-allow-origin") == origin) == allowed
