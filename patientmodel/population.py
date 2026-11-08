"""Sample synthetic patients: lexical weights (s, p) grounded in severity and published fits.

Two data anchors:
1. Severity -> naming accuracy. On the PSST data (107 AphasiaBank participants) a logistic fit gives
   logit(accuracy) = -3.88 + 0.0535 * AQ (WAB-R Aphasia Quotient), with a residual SD of 0.14 in
   accuracy between participants at the same AQ. ASSUMPTION: PSST items (BNT short form, VNT) are
   harder than the Philadelphia Naming Test, so this curve likely understates PNT accuracy at a given
   AQ; the intercept is exposed in the config to recalibrate when PNT-by-AQ data are available.
2. The split between semantic and phonological impairment. Dell's fitted patients (Dell et al. 2004
   Table 2; 2013 Table 1) have p/s ratios whose log is roughly normal with median about 1.6 and
   geometric SD about exp(0.6), excluding one near-zero-s outlier. We use that as a lognormal prior on
   p/s and let the target accuracy pick the (s, p) cells consistent with it.

The sampler draws AQ, maps it to a target accuracy with heterogeneity, then samples a grid cell of
the SP model whose predicted accuracy matches, weighted by the p/s prior, and jitters within the cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from patientmodel.fit import GridTable


@dataclass(frozen=True)
class PopulationConfig:
    # Severity distribution (PSST participants): AQ mean 69.9, SD 17.3, clipped to [10, 98].
    aq_mean: float = 69.9
    aq_sd: float = 17.3
    aq_min: float = 10.0
    aq_max: float = 98.0
    # logit(accuracy) = acc_intercept + acc_slope * AQ, plus between-patient noise in logit space.
    acc_intercept: float = -3.88
    acc_slope: float = 0.0535
    acc_logit_sd: float = 0.65     # reproduces the 0.14 residual SD in accuracy around the curve
    # Lognormal prior on p/s from Dell's fitted patients.
    log_ps_ratio_mean: float = 0.5
    log_ps_ratio_sd: float = 0.6
    # How tightly a grid cell's predicted accuracy must match the target (SD in accuracy units).
    accuracy_match_sd: float = 0.02
    min_weight: float = 0.001      # exclude the degenerate s = 0 or p = 0 cells


@dataclass
class SyntheticPatient:
    patient_id: str
    s: float
    p: float
    aq: float
    target_accuracy: float
    expected_naming: dict[str, float] = field(default_factory=dict)


def expected_accuracy(aq: float | np.ndarray, cfg: PopulationConfig) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(cfg.acc_intercept + cfg.acc_slope * np.asarray(aq, dtype=float))))


class PopulationSampler:
    def __init__(self, table: GridTable, cfg: PopulationConfig | None = None) -> None:
        self.table, self.cfg = table, cfg or PopulationConfig()
        S, P = np.meshgrid(table.s_values, table.p_values, indexing="ij")
        ok = (S >= self.cfg.min_weight) & (P >= self.cfg.min_weight)
        self._cells = np.argwhere(ok)
        self._correct = table.probs[ok][:, 0]
        log_ratio = np.log(P[ok] / S[ok])
        self._prior = np.exp(-0.5 * ((log_ratio - self.cfg.log_ps_ratio_mean) / self.cfg.log_ps_ratio_sd) ** 2)
        self._step_s = float(np.diff(table.s_values).mean()) if len(table.s_values) > 1 else 0.0
        self._step_p = float(np.diff(table.p_values).mean()) if len(table.p_values) > 1 else 0.0

    def sample_aq(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.clip(rng.normal(self.cfg.aq_mean, self.cfg.aq_sd, n), self.cfg.aq_min, self.cfg.aq_max)

    def weights_for_accuracy(self, target: float, rng: np.random.Generator) -> tuple[float, float]:
        """Sample (s, p) whose predicted accuracy is near `target`, under the p/s prior."""
        like = np.exp(-0.5 * ((self._correct - target) / self.cfg.accuracy_match_sd) ** 2)
        w = like * self._prior
        if w.sum() <= 0:                                   # target outside the grid's range: nearest cells
            w = np.exp(-0.5 * ((self._correct - target) / (5 * self.cfg.accuracy_match_sd)) ** 2) * self._prior
        w = w / w.sum()
        i, j = self._cells[rng.choice(len(w), p=w)]
        s = float(self.table.s_values[i] + rng.uniform(-0.5, 0.5) * self._step_s)
        p = float(self.table.p_values[j] + rng.uniform(-0.5, 0.5) * self._step_p)
        return max(s, self.cfg.min_weight), max(p, self.cfg.min_weight)

    def sample(self, n: int, seed: int = 0, aq: np.ndarray | None = None, prefix: str = "syn") -> list[SyntheticPatient]:
        rng = np.random.default_rng(seed)
        aqs = self.sample_aq(n, rng) if aq is None else np.asarray(aq, dtype=float)
        logit = self.cfg.acc_intercept + self.cfg.acc_slope * aqs + rng.normal(0.0, self.cfg.acc_logit_sd, len(aqs))
        targets = 1.0 / (1.0 + np.exp(-logit))
        out = []
        for k, (a, t) in enumerate(zip(aqs, targets)):
            s, p = self.weights_for_accuracy(float(t), rng)
            pred = self.table.lookup(s, p)
            out.append(SyntheticPatient(f"{prefix}-{k:04d}", s, p, float(a), float(t),
                                        {c: float(v) for c, v in zip(("correct", "semantic", "formal", "mixed", "unrelated", "nonword"), pred)}))
        return out

    def from_profile(self, profile: str, n: int, seed: int = 0) -> list[SyntheticPatient]:
        """Convenience cohorts by severity band on the AQ scale (WAB-R conventions, approximate)."""
        bands = {"mild": (76.0, 93.9), "moderate": (51.0, 75.9), "severe": (26.0, 50.9), "very_severe": (10.0, 25.9)}
        lo, hi = bands[profile]
        rng = np.random.default_rng(seed)
        return self.sample(n, seed=seed, aq=rng.uniform(lo, hi, n), prefix=profile)
