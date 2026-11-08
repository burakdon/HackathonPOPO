"""A patient you can treat: the snapshot patient plus item-level learning over days.

`TreatablePatient` wraps a snapshot `Patient` (SP model, realizer, item effects) with one
`ItemMemory` per word. Retrievability of a word at time t comes from the learning model; its
baseline is set so that, with no practice, the word's retrieval probability equals the snapshot
model's P(correct) for that item. When retrieval fails, *which* error comes out is still decided
by the SP model, conditioned on an error, so the error structure stays the snapshot's.

Events:
  attempt(item, t)   the patient tries to name the picture; feedback (the name) follows, as in
                     Middleton's retrieval-practice trials. Lays a success or fail-with-feedback trace.
  study(item, t)     the patient hears and repeats the name (errorless). Lays a study trace.
  probe(item, t)     unaided test, no feedback, no trace.
Time is in days; a session's trials are spaced by `trial_seconds`.

Cues (patientmodel/cues.py): attempt(item, t, cue) raises this attempt's retrieval probability by
the activation offset that reproduces the SP model's cued P(correct) for the item, and weights the
trace of a cued success between an unaided success and errorless study by the cue's effort share.
cue_hierarchy() runs the clinical escalation (unaided, then a first sound, then the word).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from patientmodel.cues import STANDARD_HIERARCHY, Cue
from patientmodel.dell_sp import CATEGORIES, NAMING_CATEGORIES
from patientmodel.items import Item
from patientmodel.learning import SECOND, ItemMemory, LearningModel, LearningParams
from patientmodel.patient import Patient, Response

ROOT = Path(__file__).resolve().parents[1]
FIT_PATH = ROOT / "results" / "learning_fit.json"


def fitted_learning_params(path: Path = FIT_PATH) -> LearningParams:
    """The parameters fitted to Middleton 2019, if the fit has been run; else the defaults."""
    if path.exists():
        d = json.loads(path.read_text())["params"]
        return LearningParams(**{k: v for k, v in d.items() if k in LearningParams.__dataclass_fields__})
    return LearningParams()


@dataclass
class TreatablePatient:
    snapshot: Patient
    learning: LearningModel = field(default_factory=lambda: LearningModel(fitted_learning_params()))
    probe_trials: int = 2000
    _memory: dict[str, ItemMemory] = field(default_factory=dict, init=False)
    _baseline_p: dict[str, float] = field(default_factory=dict, init=False)
    _cued_p: dict[tuple[str, str], float] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ baselines
    def baseline_probability(self, item: Item) -> float:
        if item.item_id not in self._baseline_p:
            probs = self.snapshot.category_probabilities(item, n=self.probe_trials)
            self._baseline_p[item.item_id] = probs["correct"]
        return self._baseline_p[item.item_id]

    def memory(self, item: Item) -> ItemMemory:
        if item.item_id not in self._memory:
            beta = LearningModel.beta_for_probability(self.baseline_probability(item), self.learning.p)
            self._memory[item.item_id] = ItemMemory(beta=beta)
        return self._memory[item.item_id]

    def retrievability(self, item: Item, t: float, cue: Cue | None = None) -> float:
        return self.learning.probability(self.memory(item), t, self.cue_boost(item, cue))

    def cued_probability(self, item: Item, cue: Cue) -> float:
        """Snapshot P(correct) for this item under the cue, from the SP model."""
        key = (item.item_id, cue.kind)
        if key not in self._cued_p:
            self._cued_p[key] = self.snapshot.category_probabilities(item, n=self.probe_trials, cue=cue)["correct"]
        return self._cued_p[key]

    def cue_boost(self, item: Item, cue: Cue | None) -> float:
        """Activation added on a cued attempt: the offset that moves the no-practice retrieval
        probability from the unaided to the cued SP value, in the learning model's units."""
        if cue is None:
            return 0.0
        clip = lambda q: min(max(q, 1e-4), 1 - 1e-4)
        logit = lambda q: float(np.log(clip(q) / (1 - clip(q))))
        return self.learning.p.s * (logit(self.cued_probability(item, cue)) - logit(self.baseline_probability(item)))

    # ------------------------------------------------------------------ what comes out on failure
    def _error_response(self, item: Item, t: float, cue: Cue | None = None) -> Response:
        """Draw an error category from the SP model (under the cue, if any) conditioned on not being
        correct, and realize it."""
        s, p = self.snapshot.effective_weights(item)
        rng = self.snapshot.rng
        ci = cue.input(self.snapshot.cue_cfg) if cue else None
        for _ in range(50):
            cat = CATEGORIES[int(self.snapshot.model.simulate_trials(s, p, 1, rng, ci)[0])]
            if cat != "correct":
                break
        else:
            cat = "nonword"
        real = self.snapshot.realizer.realize(item.word, cat, rng)
        return Response(item=item, category=real.category, word=real.word, phonemes=real.phonemes,
                        duration_s=self.snapshot.duration.sample(real.category, rng), correct=False,
                        fallback_from=real.fallback_from, s_eff=s, p_eff=p, cue=cue.kind if cue else None)

    def _correct_response(self, item: Item, cue: Cue | None = None) -> Response:
        s, p = self.snapshot.effective_weights(item)
        return Response(item=item, category="correct", word=item.word, phonemes=self.snapshot.realizer.lexicon.pron(item.word),
                        duration_s=self.snapshot.duration.sample("correct", self.snapshot.rng), correct=True,
                        fallback_from=None, s_eff=s, p_eff=p, cue=cue.kind if cue else None)

    # ------------------------------------------------------------------ events
    def attempt(self, item: Item, t: float, cue: Cue | None = None) -> Response:
        """Retrieval practice with feedback: try to name (with a cue, if given), then hear the name.
        A cued success lays a trace between an unaided success and errorless study, by the cue's
        effort share."""
        mem = self.memory(item)
        boost = self.cue_boost(item, cue)
        w_ok = None
        if cue is not None:
            lp = self.learning.p
            w_ok = lp.w_success - cue.effort(self.snapshot.cue_cfg) * (lp.w_success - lp.w_study)
        ok = self.learning.retrieval_practice(mem, t, self.snapshot.rng, boost=boost, success_weight=w_ok)
        return self._correct_response(item, cue) if ok else self._error_response(item, t, cue)

    def cue_hierarchy(self, item: Item, t: float, cues: tuple[Cue | None, ...] = STANDARD_HIERARCHY,
                      step_seconds: float = 5.0) -> list[Response]:
        """Clinical escalation: attempt with each cue in turn until one succeeds. Every attempt is a
        learning event; the responses are returned in order."""
        out: list[Response] = []
        for k, cue in enumerate(cues):
            r = self.attempt(item, t + k * step_seconds * SECOND, cue)
            out.append(r)
            if r.correct:
                break
        return out

    def study(self, item: Item, t: float) -> None:
        self.learning.study(self.memory(item), t)

    def probe(self, item: Item, t: float) -> Response:
        """Unaided test without feedback; leaves no trace."""
        ok = self.learning.test(self.memory(item), t, self.snapshot.rng)
        return self._correct_response(item) if ok else self._error_response(item, t)

    def session(self, items: list[Item], day: float, mode: str = "attempt", presentations: int = 1,
                lag: int = 0) -> list[Response]:
        """Run one practice session on `items` starting at `day`. mode: attempt | study.
        Items are interleaved with `lag` intervening trials between presentations of the same item
        (lag 0 = each item once per round, rounds back to back)."""
        order: list[Item] = []
        for _ in range(presentations):
            order.extend(items)
        out: list[Response] = []
        for k, it in enumerate(order):
            t = day + k * self.learning.p.trial_seconds * SECOND * (1 + lag)
            if mode == "study":
                self.study(it, t)
            else:
                out.append(self.attempt(it, t))
        return out
