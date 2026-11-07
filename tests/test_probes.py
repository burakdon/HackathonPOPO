"""Review probes: schedule, deterministic item rule, agent not consulted, metric, and the service."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from agents.baselines import FixedPolicy
from core.config import load_clinical
from eval.metrics import session_metrics
from service.app import create_app
from sim.config import ProbeConfig, load_sim
from sim.items import ItemSelector, load_items
from sim.population import make_patients
from sim.session import run_session


class CountingPolicy(FixedPolicy):
    def __init__(self):
        super().__init__(tier=2)
        self.calls = 0
        self.states = []

    def act(self, state, mask):
        self.calls += 1
        self.states.append(state)
        return super().act(state, mask)


@pytest.fixture(scope="module")
def env():
    cfg, clinical = load_sim(), load_clinical()
    return cfg, clinical, load_items(cfg.items)


def test_probe_schedule_and_agent_not_consulted(env):
    cfg, clinical, bank = env
    spec = make_patients(cfg, "mild_anomic", "test", 1, 0)[0]
    patient, selector = spec.build(bank, cfg, clinical)
    pol = CountingPolicy()
    trace = run_session(pol, patient, selector, cfg, clinical).trace
    probe_idx = [t["trial"] for t in trace if t["probe"]]
    assert probe_idx == [4, 9, 14, 19, 24, 29, 34, 39]            # every 5th trial
    assert pol.calls == len(trace) - len(probe_idx) == 32
    for t in trace:
        if t["probe"]:
            k = t["trial"]
            earlier = [u for u in trace[:k] if u["word"] == t["word"]]
            assert t["cue"] == "none" and t["tier"] == int(bank.tier[bank.words.index(t["word"])])
            assert any(not u["probe"] for u in earlier)               # practised before
            assert all(k - u["trial"] >= cfg.review_probes.min_gap for u in earlier)


def test_probes_do_not_change_current_tier(env):
    cfg, clinical, bank = env
    spec = make_patients(cfg, "severe", "test", 1, 0)[0]   # tier-2 policy, probes may be other tiers
    patient, selector = spec.build(bank, cfg, clinical)
    pol = CountingPolicy()
    run_session(pol, patient, selector, cfg, clinical)
    for s in pol.states:
        agent_trials = [t for t in s.last_trials if not t.probe]
        if agent_trials:
            assert s.current_tier == agent_trials[-1].tier


def test_probe_item_rule_is_deterministic():
    """Fewest probes first, then the item practised earliest."""
    cfg = load_sim()
    bank = load_items(cfg.items)
    sel = ItemSelector(bank, np.random.default_rng(0), ProbeConfig(enabled=True, every=5, min_gap=1))
    practiced = [sel.next(1) for _ in range(4)]                   # trials 0-3
    assert sel.probe_due()
    assert sel.next_probe() == practiced[0]                       # trial 4: earliest practised
    for _ in range(4):
        sel.next(1)                                               # trials 5-8
    assert sel.next_probe() == practiced[1]                       # trial 9: fewest probes, then earliest


def test_no_probes_when_disabled(env):
    cfg, clinical, bank = env
    c = cfg.model_copy(update={"review_probes": cfg.review_probes.model_copy(update={"enabled": False})})
    spec = make_patients(c, "mild_anomic", "test", 1, 0)[0]
    patient, selector = spec.build(bank, c, clinical)
    trace = run_session(FixedPolicy(tier=2), patient, selector, c, clinical).trace
    assert not any(t["probe"] for t in trace) and len(trace) == 40


def test_probe_recall_metric(env):
    cfg, clinical, bank = env
    spec = make_patients(cfg, "moderate_broca", "test", 1, 0)[0]
    patient, selector = spec.build(bank, cfg, clinical)
    r = run_session(FixedPolicy(tier=2), patient, selector, cfg, clinical)
    m = session_metrics(r, clinical)
    probes = [t for t in r.trace if t["probe"]]
    assert m["n_probes"] == len(probes) and m["probe_recall"] == pytest.approx(np.mean([t["outcome"] for t in probes]))
    assert 0.0 <= m["probe_p"] <= 1.0


def test_service_next_trial_serves_probes_and_skips_agent_update(tmp_path):
    c = TestClient(create_app(overrides_path=tmp_path / "o.jsonl"))
    hist, tier, probes = [], 1, []
    for k in range(10):
        state = {"session_id": "p1", "last_trials": hist[-10:], "current_tier": tier, "trial_count": k,
                 "session_minutes": k * 0.1}
        nt = c.post("/next_trial", json=state).json()
        assert set(nt) == {"word", "image_file", "tier", "cue", "probe", "rationale"}
        correct = k % 3 != 0
        up = c.post("/update", json={"state": state, "action": {"tier": nt["tier"], "cue": nt["cue"]},
                                     "outcome": {"correct": correct, "error_type": "correct" if correct else "meaning",
                                                 "response_time": 2.0}, "probe": nt["probe"]}).json()
        assert up["agent_updated"] == (not nt["probe"])
        hist.append({"word": nt["word"], "tier": nt["tier"], "cue": nt["cue"], "outcome": correct,
                     "error_type": "correct" if correct else "meaning", "response_time": 2.0, "probe": nt["probe"]})
        if nt["probe"]:
            probes.append(k)
            assert nt["cue"] == "none" and nt["word"] in [h["word"] for h in hist[:-1]]
        else:
            tier = nt["tier"]
    assert probes == [4, 9]


def test_probe_lift_reflects_learning_not_item_easiness(env):
    """Lift is 0 for a patient who cannot learn, whatever the item difficulty; positive with learning."""
    from sim.patient import PatientParams
    cfg, clinical, bank = env
    def mean_lift(learn_rate, tier):
        spec = make_patients(cfg, "mild_anomic", "test", 1, 0)[0]
        params = PatientParams(**{**spec.params.to_dict(), "learn_rate": learn_rate})
        spec.params = params
        patient, selector = spec.build(bank, cfg, clinical)
        trace = run_session(FixedPolicy(tier=tier), patient, selector, cfg, clinical).trace
        return np.mean([t["probe_lift"] for t in trace if t["probe"]])
    assert mean_lift(0.0, 1) == pytest.approx(0.0, abs=1e-9) and mean_lift(0.0, 2) == pytest.approx(0.0, abs=1e-9)
    assert mean_lift(0.3, 2) > 0.0
