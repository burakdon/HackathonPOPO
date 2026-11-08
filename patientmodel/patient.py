"""A synthetic aphasia patient you can show pictures to.

Combines the SP model (which error category), the item bank (how hard this picture is for anyone),
the realizer (which word actually comes out), and a latency model measured on PSST. This is the
snapshot patient; practice-driven change over days is added by the learning layer (next phase).

Item effects (ASSUMPTION, to be calibrated): the patient's lexical weights are scaled per item by
exp(-item_beta * difficulty), so hard pictures (low name agreement, slow, rare, long) are named less
accurately by every patient, and more so by impaired ones. item_beta = 0 switches this off.

Latency (PSST, 3,291 utterances): log-normal located at the observed medians, correct 1.38 s,
incorrect 2.15 s, with log-SDs 0.82 and 1.05. The real log-durations are right-skewed, so the median
rather than the log-mean is matched. Little dependence on severity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from patientmodel.dell_sp import CATEGORIES, DellSPConfig, DellSPModel
from patientmodel.items import Item
from patientmodel.realize import Realization, Realizer


@dataclass(frozen=True)
class LatencyModel:
    correct_log_mean: float = 0.32      # ln(1.38 s), the PSST median for correct responses
    correct_log_sd: float = 0.82
    error_log_mean: float = 0.77        # ln(2.15 s), the PSST median for incorrect responses
    error_log_sd: float = 1.05
    omission_seconds: float = 8.0      # time until the trial times out with no response (ASSUMPTION)

    def sample(self, category: str, rng: np.random.Generator) -> float:
        if category == "omission":
            return self.omission_seconds
        if category == "correct":
            return float(np.exp(rng.normal(self.correct_log_mean, self.correct_log_sd)))
        return float(np.exp(rng.normal(self.error_log_mean, self.error_log_sd)))


@dataclass
class Response:
    item: Item
    category: str                     # one of CATEGORIES
    word: str | None
    phonemes: tuple[str, ...] | None
    latency_s: float
    correct: bool
    s_eff: float
    p_eff: float


@dataclass
class Patient:
    patient_id: str
    s: float
    p: float
    model: DellSPModel
    realizer: Realizer
    aq: float | None = None
    item_beta: float = 0.15
    latency: LatencyModel = field(default_factory=LatencyModel)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def effective_weights(self, item: Item) -> tuple[float, float]:
        scale = float(np.exp(-self.item_beta * item.difficulty))
        return self.s * scale, self.p * scale

    def category_probabilities(self, item: Item, n: int = 2000) -> dict[str, float]:
        s, p = self.effective_weights(item)
        return self.model.simulate(s, p, n, seed=int(self.rng.integers(2**31)))

    def name(self, item: Item) -> Response:
        """One naming attempt at this picture."""
        s, p = self.effective_weights(item)
        cat = CATEGORIES[int(self.model.simulate_trials(s, p, 1, self.rng)[0])]
        real: Realization = self.realizer.realize(item.word, cat, self.rng)
        return Response(item, real.category, real.word, real.phonemes, self.latency.sample(real.category, self.rng),
                        real.category == "correct", s, p)

    def naming_test(self, items: list[Item]) -> list[Response]:
        return [self.name(it) for it in items]
