"""Cue mechanics: direction and ordering of effects, learning-layer interaction, hierarchy."""
import numpy as np
import pytest

from patientmodel.cues import STANDARD_HIERARCHY, Cue, CueConfig, load_cue_config
from patientmodel.dell_sp import CueInput, DellSPModel
from patientmodel.learning import ItemMemory, LearningModel, LearningParams


@pytest.fixture(scope="module")
def model():
    return DellSPModel()


def pc(model, s, p, cue=None, n=2500, seed=1):
    return model.simulate(s, p, n, seed=seed, cue=cue)["correct"]


def test_cue_kinds_and_config():
    cfg = load_cue_config()
    assert isinstance(cfg, CueConfig)
    for k in ("phonemic_initial", "phonemic_final", "semantic", "whole_word"):
        ci = Cue(k).input(cfg)
        assert isinstance(ci, CueInput) and 0.0 <= Cue(k).effort(cfg) <= 1.0
    with pytest.raises(ValueError):
        Cue("rhyme")


def test_onset_cue_helps_and_beats_coda_cue(model):
    s, p = 0.02, 0.02
    base = pc(model, s, p)
    onset = pc(model, s, p, CueInput(onset=30))
    coda = pc(model, s, p, CueInput(coda=30))
    assert onset > base + 0.05
    assert onset > coda, "formal neighbours share the coda, so a coda cue must help less than an onset cue"


def test_whole_word_cue_is_strongest(model):
    s, p = 0.015, 0.015
    cfg = load_cue_config()
    base = pc(model, s, p)
    word = pc(model, s, p, Cue("whole_word").input(cfg))
    onset = pc(model, s, p, Cue("phonemic_initial").input(cfg))
    assert word >= onset > base


def test_semantic_cue_helps_a_lesioned_patient(model):
    s, p = 0.02, 0.025
    assert pc(model, s, p, CueInput(semantic=10)) > pc(model, s, p) + 0.03


def test_cued_success_lays_a_weaker_trace():
    lm = LearningModel(LearningParams(w_success=1.0, w_study=0.5))
    always = type("R", (), {"random": staticmethod(lambda: 0.0)})()   # every attempt succeeds
    a, b = ItemMemory(beta=0.0), ItemMemory(beta=0.0)
    lm.retrieval_practice(a, 1.0, always)
    lm.retrieval_practice(b, 1.0, always, boost=2.0, success_weight=0.6)
    assert a.weights == [1.0] and b.weights == [0.6]
    assert lm.probability(ItemMemory(beta=0.0), 1.0, boost=2.0) > lm.probability(ItemMemory(beta=0.0), 1.0)


def test_cue_hierarchy_escalates_and_stops(model):
    from patientmodel.items import Item
    from patientmodel.patient import Patient
    from patientmodel.treatment import TreatablePatient

    class StubLexicon:
        def pron(self, w): return ("K", "AE", "T")

    class StubRealizer:
        lexicon = StubLexicon()
        def realize(self, word, cat, rng):
            from patientmodel.realize import Realization
            return Realization(category=cat, word=None if cat == "nonword" else word, phonemes=("K", "AE", "B"), fallback_from=None)

    item = Item(item_id="x", word="cat", name_agreement=1.0, rt_ms=800.0, log_freq=1.0, syllables=1, difficulty=0.0,
                image=None, pronounceable=True)
    pt = Patient("severe-test", 0.012, 0.012, model, StubRealizer(), rng=np.random.default_rng(0))
    tp = TreatablePatient(pt, probe_trials=800)
    assert tp.cue_boost(item, Cue("whole_word")) > tp.cue_boost(item, Cue("phonemic_initial")) > 0.0
    responses = tp.cue_hierarchy(item, 0.0)
    assert 1 <= len(responses) <= len(STANDARD_HIERARCHY)
    assert [r.cue for r in responses] == [c.kind if c else None for c in STANDARD_HIERARCHY[: len(responses)]]
    assert all(not r.correct for r in responses[:-1])
