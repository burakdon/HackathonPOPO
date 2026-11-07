"""Paper-variant DITTO, the SFT baseline, override masking, and the live DITTO loop in the service."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.config import resolve
from tests.conftest import EXAMPLE_STATE

MODEL = "models/ppo_seed0.zip"
pytestmark = pytest.mark.skipif(not resolve(MODEL).exists(), reason="trained model not present")
MIX = {"online": 0.7, "replay": 0.2, "intermodel": 0.1}


@pytest.fixture(scope="module")
def setup():
    from sb3_contrib import MaskablePPO
    from envs.aphasia_env import AphasiaEnv
    from train.ditto_ppo import collect_states
    ref = MaskablePPO.load(str(resolve(MODEL)), device="cpu")
    return ref, collect_states(ref, AphasiaEnv(cohort="population", split="val"), range(300, 306))


def _demos(ref, states, n=6):
    from agents.ditto import Demo, action_probs
    demos = []
    for s in states:
        p = action_probs(ref, s["obs"][None], s["mask"][None])[0]
        allowed = np.flatnonzero(s["mask"])
        if len(demos) < n and len(allowed) > 1:
            demos.append(Demo(s["obs"], s["mask"], int(allowed[np.argsort(p[allowed])[0]])))
    return demos


def _demo_prob(model, demos):
    from agents.ditto import action_probs
    obs, masks = np.stack([d.obs for d in demos]), np.stack([d.mask for d in demos])
    return action_probs(model, obs, masks)[np.arange(len(demos)), [d.action for d in demos]]


def test_paper_variant_learns_uses_all_pair_types_and_leaves_input_untouched(setup):
    from agents.ditto import ditto
    ref, states = setup
    demos = _demos(ref, states)
    before = _demo_prob(ref, demos)
    tuned, log = ditto(ref, demos, rounds=3, samples_per_state=8, beta=0.5, lr=1e-3, steps_per_round=30, seed=0,
                       sft_steps=10, mixture=MIX)
    assert _demo_prob(tuned, demos).mean() > before.mean() + 0.3
    assert np.allclose(_demo_prob(ref, demos), before)                  # the input model is never modified
    assert log[0]["pairs_replay"] == 0 and log[0]["pairs_intermodel"] == 0     # nothing earlier in round 0
    later = [r for r in log[1:] if "batch" in r]
    assert later and all(r["batch"].get("replay", 0) > 0 for r in later)


def test_paper_variant_is_deterministic_given_seed(setup):
    from agents.ditto import ditto
    ref, states = setup
    demos = _demos(ref, states)
    kw = dict(rounds=2, samples_per_state=8, beta=0.5, lr=1e-3, steps_per_round=10, seed=3, sft_steps=5, mixture=MIX)
    a, _ = ditto(ref, demos, **kw)
    b, _ = ditto(ref, demos, **kw)
    assert np.allclose(_demo_prob(a, demos), _demo_prob(b, demos))


def test_bad_mixture_and_empty_demos_are_rejected(setup):
    from agents.ditto import ditto
    ref, states = setup
    with pytest.raises(ValueError):
        ditto(ref, _demos(ref, states), 1, 4, 0.5, 1e-3, 1, 0, mixture={"online": 1.0, "typo": 0.1})
    with pytest.raises(ValueError):
        ditto(ref, [], 1, 4, 0.5, 1e-3, 1, 0)


def test_behaviour_cloning_baseline_learns(setup):
    from agents.ditto import behaviour_cloning
    ref, states = setup
    demos = _demos(ref, states)
    assert _demo_prob(behaviour_cloning(ref, demos, 60, 1e-3, 0), demos).mean() > _demo_prob(ref, demos).mean() + 0.3


def test_override_after_a_correct_answer_with_a_cue_is_skipped():
    """The policy acts under cue-only-after-error; such a demonstration is impossible for it."""
    from core.config import load_clinical
    from envs.aphasia_env import load_env_config
    from train.ditto_ppo import demos_from_records
    ok = {"state": EXAMPLE_STATE, "clinician_action": {"tier": 2, "cue": "semantic"}}       # after a sound error
    after_correct = {"state": {**EXAMPLE_STATE, "last_trials": [{**EXAMPLE_STATE["last_trials"][0], "outcome": True,
                                                                 "error_type": "correct"}]},
                     "clinician_action": {"tier": 2, "cue": "semantic"}}
    demos, skipped = demos_from_records([ok, after_correct], load_clinical(), load_env_config())
    assert len(demos) == 1 and skipped == [after_correct]


# ---- service ----
@pytest.fixture
def ppo_client(tmp_path):
    from service.app import create_app
    return TestClient(create_app(agent="ppo", overrides_path=tmp_path / "o.jsonl", accepted_path=tmp_path / "a.jsonl"))


def _state(k, trials=None):
    return {**EXAMPLE_STATE, "session_id": "s1", "trial_count": k,
            "last_trials": trials if trials is not None else EXAMPLE_STATE["last_trials"]}


def test_only_reviewed_non_overridden_choices_are_logged_as_accepted(ppo_client, tmp_path):
    c = ppo_client
    outcome = {"correct": True, "error_type": "correct", "response_time": 2.0}
    a0 = c.post("/act", json=_state(0)).json()
    assert c.post("/update", json={"state": _state(0), "action": a0, "outcome": outcome}).json()["accepted_logged"] is False
    a1 = c.post("/act", json=_state(1)).json()
    assert c.post("/update", json={"state": _state(1), "action": a1, "outcome": outcome,
                                   "reviewed": True}).json()["accepted_logged"] is True
    a2 = c.post("/act", json=_state(2)).json()
    other = {"tier": a2["tier"], "cue": "semantic" if a2["cue"] != "semantic" else "first_letter"}
    c.post("/override", json={"state": _state(2), "agent_action": a2, "clinician_action": other})
    assert c.post("/update", json={"state": _state(2), "action": other, "outcome": outcome,
                                   "reviewed": True}).json()["accepted_logged"] is False
    assert c.get("/ditto/status").json()["n_accepted"] == 1


def test_ditto_train_adopts_the_override_and_reset_restores_the_base(ppo_client):
    c = ppo_client
    assert c.post("/ditto/train").status_code == 400                       # no overrides yet
    a = c.post("/act", json=_state(0)).json()
    target = {"tier": a["tier"], "cue": "semantic" if a["cue"] != "semantic" else "first_letter"}
    c.post("/override", json={"state": _state(0), "agent_action": a, "clinician_action": target})
    rep = c.post("/ditto/train").json()
    assert rep["n_overrides_used"] == 1 and rep["agreement_after"] == 1.0
    assert c.get("/health").json()["agent"] == "ppo+ditto"
    now = c.post("/act", json=_state(0)).json()
    assert (now["tier"], now["cue"]) == (target["tier"], target["cue"])
    c.post("/ditto/reset")
    back = c.post("/act", json=_state(0)).json()
    assert (back["tier"], back["cue"]) == (a["tier"], a["cue"]) and c.get("/health").json()["agent"] == "ppo"


def test_ditto_endpoints_need_the_ppo_agent(tmp_path):
    from service.app import create_app
    c = TestClient(create_app(agent="logistic_bandit", overrides_path=tmp_path / "o.jsonl"))
    assert c.post("/ditto/train").status_code == 409
    assert c.get("/ditto/status").json()["live_ditto_available"] is False


def test_live_demo_script_end_to_end(tmp_path):
    from scripts.ditto_live_demo import run_demo
    from service.app import create_app
    c = TestClient(create_app(agent="ppo", overrides_path=tmp_path / "o.jsonl", accepted_path=tmp_path / "a.jsonl"))
    res = run_demo(c, n_overrides=3, n_eval=4)
    assert res["taught"]["overrides"] == 3 and res["train"]["agreement_after"] == 1.0
    assert res["after"]["follow_rate"] > res["before"]["follow_rate"]
