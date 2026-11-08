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
        params = sample_params(dists, cfg.parameters, np.random.default_rng(param_ss), _model_classes(cfg)[0])
        specs.append(PatientSpec(cohort, split, k, params, outcome_ss, selection_ss))
    return specs
