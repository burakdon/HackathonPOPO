"""Dropout: no quitting without a failure streak; quitting rises with sustained failure."""
import numpy as np

from agents.baselines import FixedPolicy
from core.config import load_clinical
from sim.config import load_sim
from sim.items import load_items
from sim.patient import PatientParams, SimPatient
from sim.population import make_patients
from sim.session import run_session


def test_dropout_is_enabled_with_a_conservative_hazard():
    d = load_sim().dropout
    assert d.enabled and d.base == 0.0 and 0 < d.streak_gain <= 0.1


def test_no_quitting_without_three_failures_in_a_row():
    cfg, clinical = load_sim(), load_clinical()
    bank = load_items(cfg.items)
    params = PatientParams(ability=8.0, phon_resp=1, sem_resp=1, phon_impair=.3, sem_impair=.3, omission_bias=.4,
                           fatigue_rate=0.0, fatigue_recovery=0.1, learn_rate=0.0)
    p = SimPatient(params, bank, cfg, clinical, np.random.default_rng(0))
    assert not any(p.respond(int(bank.indices_in_tier(1)[0]), "none").quit for _ in range(500))


def test_sustained_failure_leads_to_quitting_and_only_when_enabled():
    cfg, clinical = load_sim(), load_clinical()
    bank = load_items(cfg.items)
    def quit_rate(enabled):
        c = cfg.model_copy(update={"dropout": cfg.dropout.model_copy(update={"enabled": enabled})})
        quits = []
        for s in make_patients(c, "severe", "test", 40, 0):
            patient, selector = s.build(bank, c, clinical)
            r = run_session(FixedPolicy(tier=3), patient, selector, c, clinical)
            quits.append(r.quit and len(r.trace) < c.session.n_trials)
        return float(np.mean(quits))
    assert quit_rate(True) > 0.5 and quit_rate(False) == 0.0
