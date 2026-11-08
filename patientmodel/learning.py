"""Item-level learning and forgetting: how practice changes a word's retrievability over days.

Model. Each word has a memory trace made of its practice events. Following the ACT-R activation
account of spacing (Pavlik & Anderson 2005, Cognitive Science 29), the strength of word i at time t is

    m_i(t) = ln( exp(beta_i) + sum_k  w_k * (t - t_k)^(-d_k) )      over practice events k before t
    d_k    = c * exp(m_i(t_k)) + alpha                               decay of the k-th trace

The patient's pre-therapy retrievability beta_i is a permanent trace, so an untreated word keeps its
baseline and practice can only add to it, decaying back toward the baseline over days (Pavlik &
Anderson's form has no floor because every item in their task started with a study trace). A trace
laid down when the word was already highly active decays faster. That single mechanism
produces the spacing effect: practice after a long gap, when activation has fallen, creates a more
durable trace than practice when the word is still fresh. Retrieval probability is a logistic in
strength:

    P(retrieve_i at t) = 1 / (1 + exp((tau - m_i(t)) / s))

Our extension for therapy: each event has a weight w_k by what happened. A successful effortful
retrieval (retrieval practice) lays a stronger trace than hearing-and-repeating the word
(repetition, errorless) or than a failed attempt followed by the correct name (retrieval practice
with feedback). This is the retrieval-practice effect (Middleton et al. 2016, 2019). The weights
w_success > w_study >= w_fail_feedback are fitted to Middleton's data (see learning_fit.py).

Baseline. beta_i is the word's retrievability for this patient before therapy. Middleton selected
items each participant could recognise but not name, so for their data beta_i sits near the
retrieval floor; in the engine beta_i comes from the snapshot patient (its SP weights and the
item's difficulty) by matching the no-practice retrieval probability to the SP model's P(correct).

Time. The public interface uses days (sessions are days apart; a trial is `trial_seconds`), but the
power law is evaluated in seconds, as in Pavlik & Anderson, with a floor of one second. In days the
seconds-scale gaps inside a session make t^-d explode and the activation-dependent decay with it.

Patient-level parameters: beta (baseline), learn_scale (multiplies all event weights; some patients
learn far less), and the shared c, alpha, tau, s. Everything is explicit; nothing is hidden state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DAY = 1.0
SECOND = 1.0 / 86400.0


@dataclass(frozen=True)
class LearningParams:
    """Shared memory parameters (fitted on Middleton 2019; defaults are Pavlik & Anderson 2005's
    values re-scaled to days, to be replaced by the fit)."""
    c: float = 0.217          # activation-dependent decay
    alpha: float = 0.177      # baseline decay
    tau: float = -0.7         # retrieval threshold
    s: float = 0.255          # logistic noise
    w_success: float = 1.0    # trace weight: successful retrieval (retrieval practice, correct)
    w_fail_feedback: float = 0.5  # trace weight: failed retrieval followed by the correct name
    w_study: float = 0.5      # trace weight: hearing and repeating the word (errorless)
    trial_seconds: float = 8.0
    time_floor_seconds: float = 1.0    # smallest elapsed time counted

    def __post_init__(self) -> None:
        if min(self.c, self.alpha, self.s) <= 0:
            raise ValueError("c, alpha and s must be positive")


@dataclass
class ItemMemory:
    beta: float                       # baseline strength for this word and patient
    times: list[float] = field(default_factory=list)      # event times (days)
    weights: list[float] = field(default_factory=list)    # event weights
    decays: list[float] = field(default_factory=list)     # per-trace decay d_k


class LearningModel:
    def __init__(self, params: LearningParams | None = None, learn_scale: float = 1.0) -> None:
        self.p = params or LearningParams()
        self.learn_scale = learn_scale

    # ------------------------------------------------------------------ strength and probability
    def strength(self, mem: ItemMemory, t: float) -> float:
        """Activation of the word at time t (days)."""
        if not mem.times:
            return mem.beta
        dt = np.maximum((t - np.asarray(mem.times)) / SECOND, self.p.time_floor_seconds)
        total = float(np.exp(mem.beta)) + float(np.sum(np.asarray(mem.weights) * dt ** (-np.asarray(mem.decays))))
        return float(np.log(total))

    def probability(self, mem: ItemMemory, t: float) -> float:
        m = self.strength(mem, t)
        return float(1.0 / (1.0 + np.exp((self.p.tau - m) / self.p.s)))

    # ------------------------------------------------------------------ events
    def record(self, mem: ItemMemory, t: float, weight: float) -> None:
        """Lay down a trace at time t. Its decay depends on the word's activation at that moment."""
        m_now = self.strength(mem, t)
        m_now = max(min(m_now, 10.0), -10.0)
        d = self.p.c * float(np.exp(m_now)) + self.p.alpha
        mem.times.append(t); mem.weights.append(weight * self.learn_scale); mem.decays.append(d)

    def retrieval_practice(self, mem: ItemMemory, t: float, rng: np.random.Generator) -> bool:
        """One retrieval attempt with feedback: the patient tries to name, then hears the name.
        Returns whether the attempt succeeded. Lays a success or fail-with-feedback trace."""
        success = rng.random() < self.probability(mem, t)
        self.record(mem, t, self.p.w_success if success else self.p.w_fail_feedback)
        return success

    def study(self, mem: ItemMemory, t: float) -> None:
        """Errorless exposure: the patient hears the name and repeats it."""
        self.record(mem, t, self.p.w_study)

    def test(self, mem: ItemMemory, t: float, rng: np.random.Generator) -> bool:
        """Unaided test without feedback (does not add a trace). Middleton's retention tests gave
        no feedback; a successful test retrieval would in reality strengthen the word slightly, and
        that is deliberately ignored here to match how the data were collected."""
        return rng.random() < self.probability(mem, t)

    @staticmethod
    def beta_for_probability(p0: float, params: LearningParams) -> float:
        """Baseline strength such that the no-practice retrieval probability is p0."""
        p0 = min(max(p0, 1e-4), 1 - 1e-4)
        return params.tau - params.s * float(np.log(1.0 / p0 - 1.0))
