"""Patient population: parameter sampling, named evaluation cohorts, and the train/test split.

Training uses the broad `population` distribution (domain randomisation): the agent never trains
on three fixed profiles, so it cannot memorise them. Evaluation uses the named `presets` as
cohorts and always draws from the `test` split.

Seeding: patient k of split s comes from SeedSequence([root_seed, split_id[s], k]). The parameter
draw, the patient's outcome stream and the item-selection stream are three children of that
sequence, so (root_seed, split, k) fully determines a patient and its session, and the train and
test splits are disjoint random streams by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import ClinicalConfig
from sim.config import ParamDist, SimConfig
from sim.items import ItemBank, ItemSelector
from sim.patient import PatientParams, SimPatient

_MAX_REJECTIONS = 1000


def sample_truncated_normal(d: ParamDist, rng: np.random.Generator) -> float:
    """Rejection sampling; falls back to clipping if the window is extremely narrow."""
    if d.sd == 0:
        return d.mean
    for _ in range(_MAX_REJECTIONS):
        x = rng.normal(d.mean, d.sd)
        if d.low <= x <= d.high:
            return float(x)
    return float(np.clip(rng.normal(d.mean, d.sd), d.low, d.high))


def sample_params(dists: dict[str, ParamDist], order: list[str], rng: np.random.Generator,
                  cls=PatientParams):
    # Fixed parameter order keeps draws reproducible regardless of YAML key order.
    return cls(**{name: sample_truncated_normal(dists[name], rng) for name in order})


def _model_classes(cfg: SimConfig):
    if cfg.model == "two_step":
        from sim.two_step import TwoStepParams, TwoStepPatient
        return TwoStepParams, TwoStepPatient
    return PatientParams, SimPatient


@dataclass
class PatientSpec:
    """Everything needed to rebuild one simulated patient and its session deterministically."""

    cohort: str          # "population" or a preset name
    split: str
    index: int
    params: PatientParams
    outcome_seed: np.random.SeedSequence
    selection_seed: np.random.SeedSequence

    def build(self, bank: ItemBank, cfg: SimConfig, clinical: ClinicalConfig) -> tuple[SimPatient, ItemSelector]:
        """Fresh patient + item selector. Call again to replay the identical patient for another policy."""
        patient = _model_classes(cfg)[1](self.params, bank, cfg, clinical, np.random.default_rng(self.outcome_seed))
        selector = ItemSelector(bank, np.random.default_rng(self.selection_seed), cfg.review_probes)
        return patient, selector


_THETAS: dict[str, np.ndarray] = {}


def _pnt_thetas(path: str) -> np.ndarray:
    if path not in _THETAS:
        import csv
        from core.config import resolve
        with open(resolve(path), newline="") as f:
            _THETAS[path] = np.array([float(r["theta_t"]) for r in csv.DictReader(f)])
    return _THETAS[path]


def ground_in_pnt(params, cfg: SimConfig, rng: np.random.Generator):
    """Replace s and p with values implied by a real patient's naming ability (two-step model).

    theta: a real Philadelphia Naming Test ability score (T-scale) plus small jitter. q: that
    patient's probability of naming an item of average difficulty under the published 1PL model,
    logit q = slope * (theta - mean item difficulty). The two steps share the failure:
    P1 = q^r, P2 = q^(1-r) with r ~ Beta(profile_beta), so P1 * P2 = q for an average item, and r
    decides whether the patient's errors are mostly meaning (r high) or sound (r low) errors."""
    from dataclasses import replace
    ts, it = cfg.two_step, cfg.items
    theta = rng.choice(_pnt_thetas(ts.thetas_path)) + rng.normal(0.0, ts.theta_jitter)
    b_mean = _bank_mean_t(it.path)
    q = 1.0 / (1.0 + np.exp(-it.irt_slope * (theta - b_mean)))
    q = float(np.clip(q, 1e-3, 1 - 1e-3))
    r = float(rng.beta(*ts.profile_beta))
    logit = lambda x: float(np.log(x / (1 - x)))
    return replace(params, sem_strength=logit(q ** r), phon_strength=logit(q ** (1 - r)))


_BMEAN: dict[str, float] = {}


def _bank_mean_t(path: str) -> float:
    if path not in _BMEAN:
        import csv
        from core.config import resolve
        with open(resolve(path), newline="") as f:
            _BMEAN[path] = float(np.mean([float(r["irt_difficulty_t"]) for r in csv.DictReader(f)]))
    return _BMEAN[path]


def make_patients(cfg: SimConfig, cohort: str, split: str, n: int, root_seed: int,
                  start: int = 0) -> list[PatientSpec]:
    """`cohort` is "population" (training distribution) or a preset name."""
    if split not in cfg.splits:
        raise ValueError(f"unknown split {split!r}; expected one of {list(cfg.splits)}")
    dists = cfg.population if cohort == "population" else cfg.presets[cohort]
    # Cohort name is hashed into the seed so cohorts drawn with the same root seed are independent.
    cohort_id = int.from_bytes(cohort.encode()[:8].ljust(8, b"\0"), "little")
    specs = []
    for k in range(start, start + n):
        ss = np.random.SeedSequence([root_seed, cfg.splits[split], cohort_id, k])
        param_ss, outcome_ss, selection_ss = ss.spawn(3)
        prng = np.random.default_rng(param_ss)
        params = sample_params(dists, cfg.parameters, prng, _model_classes(cfg)[0])
        if (cohort == "population" and cfg.model == "two_step" and cfg.two_step.population_source == "pnt"):
            params = ground_in_pnt(params, cfg, prng)
        specs.append(PatientSpec(cohort, split, k, params, outcome_ss, selection_ss))
    return specs
