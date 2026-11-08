"""Two-step patient model: aphasia as weakened links between meaning, words and sounds.

Computational definition (after the two-step interactive model of word retrieval, Dell et al. 1997,
and its semantic-phonological version, Foygel & Dell 2000, in which a patient is characterised by
the strength of semantic-lexical links `s` and lexical-phonological links `p`):

    picture -> [step 1: meaning -> word]  -> [step 2: word -> sounds] -> spoken name

    P1(item, cue) = sigmoid(sem_strength + general + learn1[i] + beta_f * z(frequency)[i]
                            + sem_resp  * stage1_cue[cue] - fatigue)        # select the right word
    P2(item, cue) = sigmoid(phon_strength + general + learn2[i] - beta_l * z(length)[i]
                            + phon_resp * stage2_cue[cue] - fatigue)        # encode its sounds
    p(correct)    = P1 * P2

Errors are not labelled at random: they come from WHERE retrieval failed.
    step 1 fails  -> a related wrong word ("dog" for "cat"):         meaning error, or no response
    step 2 fails  -> the right word, wrong sounds ("cap" for "cat"):  sound error, or no response
                                                                      (tip of the tongue)
Cues act on the step they feed: a semantic cue or sentence frame boosts step 1, a first sound or
first letter boosts step 2 (with a small boost to step 1 through feedback, as in the interactive
model). Because p(correct) is a product, a cue helps most when it targets the step that is failing,
so the error a patient makes predicts which cue will help. This is the interaction a therapist reads
and the original single-logit model could not express.

This is a closed-form, two-stage approximation of the spreading-activation model, not a simulation
of the network itself: an RL agent trains on ~1.5M trials and the oracle, metrics and privileged
reward need the exact p(correct) of all 15 actions at every step, which a stochastic network
simulation cannot provide cheaply. Parameter values are ASSUMPTIONS (configs/sim_two_step.yaml);
the structure, not the magnitudes, is what is taken from the literature.

Common random numbers are kept: every trial consumes exactly four draws, as in sim/patient.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from core.config import ClinicalConfig
from core.schema import CUES
from sim.config import SimConfig
from sim.items import ItemBank, _z
from sim.patient import SimPatient, TrialResult, sigmoid


@dataclass(frozen=True)
class TwoStepParams:
    """Hidden per-patient parameters of the two-step model (see configs/sim_two_step.yaml)."""

    sem_strength: float      # s: meaning -> word link strength (log-odds)
    phon_strength: float     # p: word -> sounds link strength (log-odds)
    sem_resp: float          # responsiveness to cues that feed step 1
    phon_resp: float         # responsiveness to cues that feed step 2
    omission_bias: float     # P(no response | step 1 failed); else a meaning error
    tot_rate: float          # P(no response | step 2 failed) (tip of the tongue); else a sound error
    fatigue_rate: float
    fatigue_recovery: float
    learn_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class TwoStepPatient(SimPatient):
    """Drop-in replacement for SimPatient (same public methods), built from TwoStepParams."""

    def __init__(self, params: TwoStepParams, bank: ItemBank, cfg: SimConfig,
                 clinical: ClinicalConfig, rng: np.random.Generator) -> None:
        ts = cfg.two_step
        if ts is None:
            raise ValueError("model: two_step needs a `two_step:` section in the simulator config")
        self.params, self.bank, self.cfg, self.clinical, self.rng = params, bank, cfg, clinical, rng
        self.general = 0.0                          # generalised gain, shared by both steps
        self.learn1 = np.zeros(len(bank))           # item-specific gain on step 1
        self.learn2 = np.zeros(len(bank))           # item-specific gain on step 2
        self.fatigue = 0.0
        self.fail_streak = 0
        self.base1 = params.sem_strength + ts.beta_frequency * _z(bank.frequency)
        self.base2 = params.phon_strength - ts.beta_length * _z(bank.length)
        self.cue1 = {c: params.sem_resp * ts.stage1_cue[c] for c in CUES}
        self.cue2 = {c: params.phon_resp * ts.stage2_cue[c] for c in CUES}
        self._max_strength = max(clinical.cue_strength.values())
        self._learn_norm = self._expected_gain_max()
        self.initial_retrieval = self.retrieval_probabilities().copy()

    # SimPatient code reads these two attributes; keep them meaningful.
    @property
    def ability(self) -> float:
        return self.general

    @property
    def item_learning(self) -> np.ndarray:
        return self.learn1 + self.learn2

    # ---- hidden-state queries (oracle and metrics only) ----
    def stage_probs(self, idx, cue: str, fatigue: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        f = self.fatigue if fatigue is None else fatigue
        p1 = sigmoid(self.base1[idx] + self.general + self.learn1[idx] + self.cue1[cue] - f)
        p2 = sigmoid(self.base2[idx] + self.general + self.learn2[idx] + self.cue2[cue] - f)
        return p1, p2

    def logit(self, item: int, cue: str) -> float:
        p = self.p_correct(item, cue)
        return float(np.log(p / (1 - p)))

    def p_correct(self, item: int, cue: str) -> float:
        p1, p2 = self.stage_probs(item, cue)
        return float(p1 * p2)

    def p_correct_tier(self, tier: int, cue: str) -> float:
        p1, p2 = self.stage_probs(self.bank.indices_in_tier(tier), cue)
        return float((p1 * p2).mean())

    def retrieval_probabilities(self) -> np.ndarray:
        p1, p2 = self.stage_probs(slice(None), "none", fatigue=0.0)
        return p1 * p2

    def error_distribution(self, item: int, cue: str) -> dict[str, float]:
        """P(correct, meaning, sound, no_response) for one trial (sim-only, used by tests and reports)."""
        p1, p2 = (float(x) for x in self.stage_probs(item, cue))
        f1, f2 = 1 - p1, p1 * (1 - p2)
        om, tot = self.params.omission_bias, self.params.tot_rate
        return {"correct": p1 * p2, "meaning": f1 * (1 - om), "sound": f2 * (1 - tot),
                "no_response": f1 * om + f2 * tot}

    # ---- one trial ----
    def respond(self, item: int, cue: str) -> TrialResult:
        u_correct, u_error, z_rt, u_quit = self.rng.random(), self.rng.random(), self.rng.standard_normal(), self.rng.random()
        p1, p2 = (float(x) for x in self.stage_probs(item, cue))
        p = p1 * p2
        correct = bool(u_correct < p)

        if correct:
            cue_factor = 1.0 - self.cfg.learning.cue_discount * self.clinical.cue_strength[cue] / self._max_strength
            gain = self.params.learn_rate * self.expected_gain(p) / p * cue_factor
            # ASSUMPTION: practice strengthens the bottleneck most; the gain is split between the two
            # steps in proportion to how likely each was to fail.
            w1 = (1 - p1) / max((1 - p1) + (1 - p2), 1e-9)
            self.learn1[item] += gain * w1
            self.learn2[item] += gain * (1 - w1)
            self.general += self.cfg.learning.generalisation * gain
            error_type = "correct"
        else:
            # Which step failed, given that the trial failed; then the error that failure produces.
            f1, f2 = 1 - p1, p1 * (1 - p2)
            om, tot = self.params.omission_bias, self.params.tot_rate
            cdf = np.cumsum([f1 * (1 - om), f1 * om, f2 * (1 - tot), f2 * tot]) / (f1 + f2)
            error_type = ("meaning", "no_response", "sound", "no_response")[int(np.searchsorted(cdf, u_error, side="right").clip(0, 3))]

        self.fail_streak = 0 if correct else self.fail_streak + 1
        fc = self.cfg.fatigue
        self.fatigue = min(fc.cap, self.fatigue * (1 - self.params.fatigue_recovery)
                           + self.params.fatigue_rate * (1 + fc.streak_gain * self.fail_streak))

        rt_cfg = self.cfg.response_time
        log_rt = (np.log(rt_cfg.base_s) + rt_cfg.difficulty_gain * (1 - p)
                  + rt_cfg.fatigue_gain * self.fatigue + rt_cfg.sigma * z_rt)
        rt = float(np.exp(log_rt)) + (0.0 if correct else rt_cfg.fail_extra_s)

        d = self.cfg.dropout
        hazard = d.base + d.streak_gain * max(0, self.fail_streak - 2)
        quit_ = bool(d.enabled and u_quit < hazard)
        return TrialResult(correct, error_type, round(rt, 2), p, quit_)
