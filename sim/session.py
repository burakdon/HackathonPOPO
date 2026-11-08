"""Play one session of any Policy against one simulated patient.

This is the same loop the live service runs (build State -> act -> show trial -> update), with the
simulated patient standing in for the person. The policy only ever sees the State JSON; hidden
quantities (true p, retrieval probabilities) are recorded in the trace for evaluation only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agents.base import Policy
from core.config import ClinicalConfig
from core.masks import action_mask, is_allowed
from core.schema import MAX_LAST_TRIALS, Outcome, State
from sim.config import SimConfig
from sim.items import ItemSelector
from sim.patient import SimPatient


def play_probe(patient: SimPatient, selector: ItemSelector, k: int) -> tuple[dict, dict, bool]:
    """One review-probe trial: the selector picks the item, no cue, at the item's own tier.
    Returns (observable record for the State JSON, full trace record, whether the patient quit)."""
    item = selector.next_probe()
    tier = int(patient.bank.tier[item])
    p_tier = patient.p_correct_tier(tier, "none")
    # Probe lift (sim-only): improvement of this item's uncued, unfatigued recall since the session
    # began, measured just before the probe. Unlike raw probe recall it cannot be raised by
    # practising only easy items.
    lift = float(patient.retrieval_probabilities()[item] - patient.initial_retrieval[item])
    r = patient.respond(item, "none")
    observable = {"word": patient.bank.words[item], "tier": tier, "cue": "none", "outcome": r.correct,
                  "error_type": r.error_type, "response_time": r.response_time, "probe": True}
    return observable, {**observable, "trial": k, "p_true": r.p_correct, "p_tier": p_tier,
                        "fatigue": patient.fatigue, "probe_lift": lift}, r.quit


@dataclass
class SessionResult:
    trace: list[dict] = field(default_factory=list)   # one dict per trial (observable + hidden fields)
    start_retrieval: float = 0.0                        # mean uncued p over the item bank, before
    end_retrieval: float = 0.0                          # ... and after the session
    start_tier: int = 1
    quit: bool = False

    @property
    def learning_gain(self) -> float:
        """True objective (simulation only): increase in mean retrieval probability over the bank."""
        return self.end_retrieval - self.start_retrieval


def _observe(correct: bool, error_type: str, obs_noise: float, rng: np.random.Generator | None) -> tuple[bool, str]:
    """What an imperfect scorer reports. With probability obs_noise the outcome is flipped: a correct
    answer is reported as a sound error, an error as correct. One uniform draw per trial, always."""
    if not obs_noise:
        return correct, error_type
    if rng.random() < obs_noise:
        return (False, "sound") if correct else (True, "correct")
    return correct, error_type


def run_session(policy: Policy, patient: SimPatient, selector: ItemSelector, cfg: SimConfig,
                clinical: ClinicalConfig, n_trials: int | None = None,
                session_id: str = "sim", obs_noise: float = 0.0,
                noise_rng: np.random.Generator | None = None) -> SessionResult:
    """`obs_noise` simulates scorer errors (evaluation only): the agent sees a flipped outcome with this
    probability, while the patient's true response, learning and all metrics use the true outcome."""
    n_trials = n_trials or cfg.session.n_trials
    tier = cfg.session.start_tier
    history: list[dict] = []
    minutes = 0.0
    result = SessionResult(start_retrieval=patient.mean_retrieval(), start_tier=tier)

    for k in range(n_trials):
        if selector.probe_due():
            # Review probe: chosen by the app, not the agent; no policy call, current tier unchanged.
            observable, record, quit_ = play_probe(patient, selector, k)
            seen, seen_err = _observe(observable["outcome"], observable["error_type"], obs_noise, noise_rng)
            observable = {**observable, "outcome": seen, "error_type": seen_err}
            record["observed_outcome"] = seen
            history.append(observable)
            result.trace.append(record)
            minutes += (observable["response_time"] + cfg.response_time.overhead_s) / 60.0
            if quit_:
                result.quit = True
                break
            continue
        state = State(session_id=session_id, last_trials=history[-MAX_LAST_TRIALS:],
                      current_tier=tier, trial_count=k, session_minutes=round(minutes, 2))
        mask = action_mask(tier, clinical.approved_cues, clinical.max_tier_step)
        action = policy.act(state, mask)
        if not is_allowed(action, mask):
            raise RuntimeError(f"{policy.name} returned masked action {action} at tier {tier}")

        item = selector.next(action.tier)
        p_tier = patient.p_correct_tier(action.tier, action.cue)
        r = patient.respond(item, action.cue)
        seen, seen_err = _observe(r.correct, r.error_type, obs_noise, noise_rng)
        policy.update(state, action, Outcome(correct=seen, error_type=seen_err, response_time=r.response_time))

        observable = {"word": patient.bank.words[item], "tier": action.tier, "cue": action.cue,
                      "outcome": seen, "error_type": seen_err, "response_time": r.response_time,
                      "probe": False}
        history.append(observable)
        result.trace.append({**observable, "outcome": r.correct, "error_type": r.error_type, "observed_outcome": seen,
                             "trial": k, "p_true": r.p_correct, "p_tier": p_tier, "fatigue": patient.fatigue})
        tier = action.tier
        minutes += (r.response_time + cfg.response_time.overhead_s) / 60.0
        if r.quit:
            result.quit = True
            break

    result.end_retrieval = patient.mean_retrieval()
    return result
