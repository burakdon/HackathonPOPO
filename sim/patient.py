"""Simulated patient: the hidden state of the POMDP.

p(correct | item i, cue c) = sigmoid(ability + item_learning[i] - difficulty[i] + cue_gain[c] - fatigue)

Everything here is hidden from agents. Agents only see the State JSON built from trial outcomes;
`p_correct` is exposed for the oracle and for simulation-only metrics (true control error,
learning gain), never as an observation.

Common random numbers: every trial consumes exactly the same number of draws from the patient's
outcome stream (one uniform for correctness, one for error type, one normal for response time,
one uniform for dropout), whatever happens. So two policies facing the same patient see the same
"luck" trial by trial, which makes paired comparisons much tighter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from core.config import ClinicalConfig
from core.schema import CUES
from sim.config import SimConfig
from sim.items import ItemBank

ERROR_TYPES_ON_FAIL = ("sound", "meaning", "no_response")


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class PatientParams:
    """Hidden per-patient parameters (see configs/sim.yaml for meaning, ranges and sources)."""

    ability: float
    phon_resp: float
    sem_resp: float
    phon_impair: float
    sem_impair: float
    omission_bias: float
    fatigue_rate: float
    fatigue_recovery: float
    learn_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TrialResult:
    correct: bool
    error_type: str
    response_time: float
    p_correct: float     # true probability at the moment of the trial (sim-only)
    quit: bool           # patient dropped out after this trial (only if dropout is enabled)


class SimPatient:
    def __init__(self, params: PatientParams, bank: ItemBank, cfg: SimConfig,
                 clinical: ClinicalConfig, rng: np.random.Generator) -> None:
        self.params, self.bank, self.cfg, self.clinical, self.rng = params, bank, cfg, clinical, rng
        self.ability = params.ability
        self.item_learning = np.zeros(len(bank))
        self.fatigue = 0.0
        self.fail_streak = 0
        self.cue_gain = {c: cfg.cues.phonemic_weight[c] * params.phon_resp
                            + cfg.cues.semantic_weight[c] * params.sem_resp for c in CUES}
        self._max_strength = max(clinical.cue_strength.values())
        err = np.array([params.phon_impair, params.sem_impair, params.omission_bias])
        self._error_cdf = np.cumsum(err / err.sum())
        self._learn_norm = self._expected_gain_max()
        self.initial_retrieval = self.retrieval_probabilities().copy()   # per item, at session start

    # ---- hidden-state queries (oracle and metrics only) ----
    def logit(self, item: int, cue: str) -> float:
        return (self.ability + self.item_learning[item] - self.bank.difficulty[item]
                + self.cue_gain[cue] - self.fatigue)

    def p_correct(self, item: int, cue: str) -> float:
        return float(sigmoid(self.logit(item, cue)))

    def p_correct_tier(self, tier: int, cue: str) -> float:
        """Mean p(correct) over the tier's items (what an agent choosing a tier effectively faces)."""
        idx = self.bank.indices_in_tier(tier)
        x = self.ability + self.item_learning[idx] - self.bank.difficulty[idx] + self.cue_gain[cue] - self.fatigue
        return float(sigmoid(x).mean())

    def retrieval_probabilities(self) -> np.ndarray:
        """Uncued, unfatigued p(correct) for every item: the basis of the true learning objective."""
        return sigmoid(self.ability + self.item_learning - self.bank.difficulty)

    def mean_retrieval(self) -> float:
        return float(self.retrieval_probabilities().mean())

    # ---- learning rule ----
    def _expected_gain_max(self) -> float:
        a, b = self.cfg.learning.a, self.cfg.learning.b
        p = a / (a + b)
        return p ** a * (1 - p) ** b

    def expected_gain(self, p: float | np.ndarray) -> float | np.ndarray:
        """Expected item-learning per trial at success probability p, normalised to max 1 at the peak."""
        a, b = self.cfg.learning.a, self.cfg.learning.b
        return p ** a * (1 - p) ** b / self._learn_norm

    # ---- one trial ----
    def respond(self, item: int, cue: str) -> TrialResult:
        u_correct, u_error, z_rt, u_quit = self.rng.random(), self.rng.random(), self.rng.standard_normal(), self.rng.random()
        p = self.p_correct(item, cue)
        correct = bool(u_correct < p)

        if correct:
            # h(p) = expected_gain(p) / p, so that E[gain] = p * h(p) = expected_gain(p).
            cue_factor = 1.0 - self.cfg.learning.cue_discount * self.clinical.cue_strength[cue] / self._max_strength
            gain = self.params.learn_rate * self.expected_gain(p) / p * cue_factor
            self.item_learning[item] += gain
            self.ability += self.cfg.learning.generalisation * gain
            error_type = "correct"
        else:
            error_type = ERROR_TYPES_ON_FAIL[int(np.searchsorted(self._error_cdf, u_error, side="right"))]

        self.fail_streak = 0 if correct else self.fail_streak + 1
        f = self.cfg.fatigue
        self.fatigue = min(f.cap, self.fatigue * (1 - self.params.fatigue_recovery)
                           + self.params.fatigue_rate * (1 + f.streak_gain * self.fail_streak))

        rt_cfg = self.cfg.response_time
        log_rt = (np.log(rt_cfg.base_s) + rt_cfg.difficulty_gain * (1 - p)
                  + rt_cfg.fatigue_gain * self.fatigue + rt_cfg.sigma * z_rt)
        rt = float(np.exp(log_rt)) + (0.0 if correct else rt_cfg.fail_extra_s)

        d = self.cfg.dropout
        hazard = d.base + d.streak_gain * max(0, self.fail_streak - 2)
        quit_ = bool(d.enabled and u_quit < hazard)
        return TrialResult(correct, error_type, round(rt, 2), p, quit_)
