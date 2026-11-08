"""A synthetic aphasia patient you can show pictures to.

Combines the SP model (which error category), the item bank (how hard this picture is for anyone),
the realizer (which word actually comes out), and a latency model measured on PSST. This is the
snapshot patient; practice-driven change over days is added by the learning layer (next phase).

Item effects (ASSUMPTION, to be calibrated): the patient's lexical weights are scaled per item by
exp(-item_beta * difficulty), so hard pictures (low name agreement, slow, rare, long) are named less
accurately by every patient, and more so by impaired ones. item_beta = 0 switches this off.

Response duration: PSST gives the *length of each recorded utterance* (3,291 clips), not the time
from picture onset to speech. We use it as a proxy for how long a response occupies: log-normal at
the observed medians, correct 1.38 s, incorrect 2.15 s, log-SDs 0.82 and 1.05. It is NOT a measure
of response latency and must not be reported as one; a latency model needs onset-aligned data.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from patientmodel.cues import Cue, CueConfig, load_cue_config
from patientmodel.dell_sp import CATEGORIES, DellSPConfig, DellSPModel
from patientmodel.items import Item
from patientmodel.realize import Realization, Realizer


@dataclass(frozen=True)
class ResponseDurationModel:
    """Utterance duration proxy (see module docstring); not latency."""
    correct_log_mean: float = 0.32      # ln(1.38 s), the PSST median clip length for correct responses
    correct_log_sd: float = 0.82
    error_log_mean: float = 0.77        # ln(2.15 s), the PSST median clip length for incorrect responses
    error_log_sd: float = 1.05
    omission_seconds: float = 15.0     # trial time-out with no response; ASSUMPTION, set above the error p90 (12.7 s)

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
    duration_s: float                 # utterance-duration proxy, not latency (see ResponseDurationModel)
    correct: bool
    fallback_from: str | None         # model category when the realizer had to substitute one
    s_eff: float
    p_eff: float
    cue: str | None = None            # the cue kind given on this attempt, if any


@dataclass
class Patient:
    patient_id: str
    s: float
    p: float
    model: DellSPModel
    realizer: Realizer
    aq: float | None = None
    item_beta: float = 0.15
    duration: ResponseDurationModel = field(default_factory=ResponseDurationModel)
    rng: np.random.Generator | None = None   # None: seeded from the patient_id, so distinct patients get distinct streams
    cue_cfg: CueConfig = field(default_factory=load_cue_config)

    def __post_init__(self) -> None:
        if self.rng is None:
            seed = int.from_bytes(hashlib.md5(self.patient_id.encode()).digest()[:4], "little")
            self.rng = np.random.default_rng(seed)
        self._probe_rng = np.random.default_rng(self.rng.integers(2**31))   # separate stream for diagnostics

    def effective_weights(self, item: Item) -> tuple[float, float]:
        scale = float(np.exp(-self.item_beta * item.difficulty))
        return self.s * scale, self.p * scale

    def category_probabilities(self, item: Item, n: int = 2000, cue: Cue | None = None) -> dict[str, float]:
        """Diagnostic: category distribution for this item, optionally under a cue. Uses its own
        stream, so calling it does not change the sequence of responses `name` will produce."""
        s, p = self.effective_weights(item)
        return self.model.simulate(s, p, n, seed=int(self._probe_rng.integers(2**31)),
                                   cue=cue.input(self.cue_cfg) if cue else None)

    def name(self, item: Item, cue: Cue | None = None) -> Response:
        """One naming attempt at this picture, unaided or with a cue."""
        s, p = self.effective_weights(item)
        ci = cue.input(self.cue_cfg) if cue else None
        cat = CATEGORIES[int(self.model.simulate_trials(s, p, 1, self.rng, ci)[0])]
        real: Realization = self.realizer.realize(item.word, cat, self.rng)
        return Response(item=item, category=real.category, word=real.word, phonemes=real.phonemes,
                        duration_s=self.duration.sample(real.category, self.rng), correct=real.category == "correct",
                        fallback_from=real.fallback_from, s_eff=s, p_eff=p, cue=cue.kind if cue else None)

    def naming_test(self, items: list[Item]) -> list[Response]:
        return [self.name(it) for it in items]
