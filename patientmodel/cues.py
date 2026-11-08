"""Cues: what a clinician says to help a patient name a picture, as external activation in the SP model.

Four cue kinds, each a `CueInput` to the target's units (see dell_sp.CueInput):
  phonemic_initial  the word's first sound ("it starts with /b/"): activation to the target's onset unit
  phonemic_final    the word's last sound: activation to the target's coda unit, which the formal
                    neighbours share, so it helps them too
  semantic          a meaning cue (definition, associate, category): activation to the target's
                    semantic features, which the semantic neighbour partly shares
  whole_word        the clinician says the word and the patient repeats it: activation to all three
                    of the target's phoneme units and to its word node (Dell et al. 2013's dual-route
                    repetition: lexical plus non-lexical input)

Activation levels are in configs/cues.yaml, calibrated by scripts/calibrate_cues.py to cued-naming
accuracy in Nardo et al. 2017 (Brain 140: 18 chronic patients; noise control 0.60, final phoneme 0.79,
initial phoneme 0.88, whole word 0.93 before treatment). The semantic level is an ASSUMPTION ranked
below the phonemic cues (Pease & Goodglass 1978; Python et al. 2021).

`effort` is the share of the retrieval the cue does for the patient (0 = unaided, 1 = repetition).
The learning layer weights a cued success between an unaided success and errorless study by it:
Howard et al. 1985 found phonemic-cue facilitation short-lived against lasting effects of tasks that
make the patient retrieve, and Middleton et al. 2019 found retrieval worth more than repetition.
Effort values are ASSUMPTIONS stated in the config.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from patientmodel.dell_sp import CueInput

ROOT = Path(__file__).resolve().parents[1]
CUE_KINDS: tuple[str, ...] = ("phonemic_initial", "phonemic_final", "semantic", "whole_word")


@dataclass(frozen=True)
class CueConfig:
    phonemic_initial_onset: float = 20.0
    phonemic_final_coda: float = 20.0
    semantic_features: float = 5.0
    whole_word_phoneme: float = 20.0
    whole_word_lexical: float = 20.0
    effort: dict | None = None

    def effort_of(self, kind: str) -> float:
        default = {"phonemic_initial": 0.5, "phonemic_final": 0.4, "semantic": 0.3, "whole_word": 1.0}
        return float((self.effort or default).get(kind, default[kind]))


def load_cue_config(path: str | Path = ROOT / "configs" / "cues.yaml") -> CueConfig:
    path = Path(path)
    if not path.exists():
        return CueConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    act = raw.get("activation", {})
    return CueConfig(phonemic_initial_onset=float(act.get("phonemic_initial_onset", 20.0)),
                     phonemic_final_coda=float(act.get("phonemic_final_coda", 20.0)),
                     semantic_features=float(act.get("semantic_features", 5.0)),
                     whole_word_phoneme=float(act.get("whole_word_phoneme", 20.0)),
                     whole_word_lexical=float(act.get("whole_word_lexical", 20.0)),
                     effort=raw.get("effort"))


@dataclass(frozen=True)
class Cue:
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in CUE_KINDS:
            raise ValueError(f"unknown cue kind {self.kind!r}; one of {CUE_KINDS}")

    def input(self, cfg: CueConfig) -> CueInput:
        if self.kind == "phonemic_initial":
            return CueInput(onset=cfg.phonemic_initial_onset)
        if self.kind == "phonemic_final":
            return CueInput(coda=cfg.phonemic_final_coda)
        if self.kind == "semantic":
            return CueInput(semantic=cfg.semantic_features)
        return CueInput(onset=cfg.whole_word_phoneme, vowel=cfg.whole_word_phoneme, coda=cfg.whole_word_phoneme,
                        word=cfg.whole_word_lexical)

    def effort(self, cfg: CueConfig) -> float:
        return cfg.effort_of(self.kind)


STANDARD_HIERARCHY: tuple[Cue | None, ...] = (None, Cue("phonemic_initial"), Cue("whole_word"))
