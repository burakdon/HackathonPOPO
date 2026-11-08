"""Gymnasium environment: a course of therapy on one synthetic patient from the engine.

What the agent controls on every trial: which word to practise (by an observable rule) and which cue
to give. What it is judged on: how many of the practised words the patient can still name, unaided,
a week and a month after the course (the engine's learning layer predicts this from the practice
history; patientmodel/learning.py, fitted to Middleton et al. 2019).

Actions (16 = 4 item rules x 4 cues):
  item rule   due       the word practised longest ago (round-robin, maximal spacing)
              repeat    the word just practised (massed; masked on a session's first trial)
              weakest   the word with the lowest observed accuracy so far (the one the patient fails)
              strongest the word with the highest observed accuracy (maintenance)
  cue         none | phonemic_initial | semantic | whole_word   (patientmodel/cues.py)

Observation (136 floats, all from what a real session would record): the last `history` trials
(cue, rule, outcome, response type) and summary counts (session and overall accuracy, share of
words attempted, recency, share of words below 50% observed accuracy, where in the course we are).
Hidden from the agent: the patient's lesion weights, each word's memory trace, true probabilities.

Reward: `privileged` is the change, after each trial, in the mean predicted unaided retention of the
practised words on the first test day. It telescopes to the episode objective exactly, so it is dense
without being a different objective. `observable` rewards unaided correct responses (and cued ones
less), which a real session could compute; policies trained on either are judged on retention.

Patients: PopulationSampler with the fitted prior (or a severity profile); patient k of a split is
seeded from (root_seed, split, k), so every policy meets the same patient with the same random
stream (common random numbers; streams diverge only when the policies' choices diverge).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from patientmodel.cues import Cue
from patientmodel.dell_sp import CATEGORIES, DellSPModel, load_dell_sp_config
from patientmodel.fit import load_or_build_grid
from patientmodel.items import Item, load_item_bank
from patientmodel.learning import SECOND
from patientmodel.patient import Patient
from patientmodel.population import PopulationSampler
from patientmodel.realize import Realization
from patientmodel.treatment import TreatablePatient

ROOT = Path(__file__).resolve().parents[1]
ITEM_RULES: tuple[str, ...] = ("due", "repeat", "weakest", "strongest")
CUE_KINDS: tuple[str | None, ...] = (None, "phonemic_initial", "semantic", "whole_word")
ACTIONS: tuple[tuple[str, str | None], ...] = tuple((r, c) for r in ITEM_RULES for c in CUE_KINDS)
N_ACTIONS = len(ACTIONS)
RESPONSE_TYPES: tuple[str, ...] = CATEGORIES          # correct + six error types
PER_TRIAL = 1 + len(CUE_KINDS) + len(ITEM_RULES) + len(RESPONSE_TYPES)
CURRENT = 8


def action_index(rule: str, cue: str | None) -> int:
    return ITEM_RULES.index(rule) * len(CUE_KINDS) + CUE_KINDS.index(cue)


@dataclass
class EngineEnvConfig:
    n_items: int = 20
    baseline_range: tuple[float, float] = (0.05, 0.55)
    schedule_days: tuple[float, ...] = (0.0, 3.0, 7.0, 10.0)
    trials_per_session: int = 40
    test_days: tuple[float, ...] = (17.0, 38.0)
    history: int = 8
    probe_trials: int = 500
    reward: str = "privileged"
    whole_word_only_after_error: bool = False
    cohort: str = "population"
    splits: dict = field(default_factory=lambda: {"train": 0, "val": 1, "test": 2})


def load_engine_env_config(path: str | Path = ROOT / "configs" / "engine_env.yaml") -> EngineEnvConfig:
    raw = yaml.safe_load(open(path)) or {}
    for k in ("baseline_range", "schedule_days", "test_days"):
        if k in raw:
            raw[k] = tuple(raw[k])
    return EngineEnvConfig(**raw)


class CategoryRealizer:
    """Realizer stand-in for training: keeps the response category, produces no word form. The full
    word-level realizer (patientmodel/realize.py) is for transcripts and demos, not for RL steps."""
    class _Lex:
        @staticmethod
        def pron(word: str) -> tuple[str, ...]:
            return ()
    lexicon = _Lex()

    def realize(self, word: str, cat: str, rng) -> Realization:
        return Realization(category=cat, word=word if cat == "correct" else None, phonemes=None, fallback_from=None)


class _Shared:
    """Model, grid and item bank loaded once per process."""
    model: DellSPModel | None = None
    table = None
    bank: list[Item] | None = None

    @classmethod
    def get(cls):
        if cls.model is None:
            cfg, fit_cfg = load_dell_sp_config()
            cls.model = DellSPModel(cfg)
            cls.table = load_or_build_grid(cls.model, fit_cfg["grid_step"], fit_cfg["trials_per_cell"], fit_cfg["seed"])
            cls.bank = [i for i in load_item_bank() if i.pronounceable]
        return cls.model, cls.table, cls.bank


@dataclass
class ItemStats:
    attempts: int = 0
    correct: int = 0
    last_trial: int = -1          # global trial index of the last practice, -1 = never
    session_presentations: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts if self.attempts else 0.5


class EngineTherapyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EngineEnvConfig | None = None, split: str = "train", root_seed: int = 0,
                 realizer=None, strict_masks: bool = True) -> None:
        super().__init__()
        self.cfg = cfg or load_engine_env_config()
        self.split, self.root_seed, self.strict_masks = split, root_seed, strict_masks
        self.model, self.table, self.bank = _Shared.get()
        self.sampler = PopulationSampler(self.table)
        self.realizer = realizer or CategoryRealizer()
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.cfg.history * PER_TRIAL + CURRENT,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.n_sessions = len(self.cfg.schedule_days)
        self.total_trials = self.n_sessions * self.cfg.trials_per_session

    # ------------------------------------------------------------------ patient
    def _build_patient(self, k: int) -> tuple[TreatablePatient, list[Item]]:
        ss = np.random.SeedSequence([self.root_seed, self.cfg.splits[self.split], k])
        seeds = ss.generate_state(3)
        if self.cfg.cohort == "population":
            spec = self.sampler.sample(1, seed=int(seeds[0]), prefix=f"{self.split}{k}")[0]
        else:
            spec = self.sampler.from_profile(self.cfg.cohort, 1, seed=int(seeds[0]))[0]
        pt = Patient(spec.patient_id, spec.s, spec.p, self.model, self.realizer, aq=spec.aq,
                     rng=np.random.default_rng(int(seeds[1])))
        tp = TreatablePatient(pt, probe_trials=self.cfg.probe_trials)
        # Personally hard words: scan the harder half of the bank in a patient-specific order until
        # n_items fall inside the baseline range (Middleton treated words the participant could not name).
        order = np.random.default_rng(int(seeds[2])).permutation(len(self.bank))
        hard = sorted(order, key=lambda i: -self.bank[i].difficulty)[: len(self.bank) // 2]
        lo, hi = self.cfg.baseline_range
        chosen: list[Item] = []
        for i in np.random.default_rng(int(seeds[2]) + 1).permutation(hard):
            it = self.bank[i]
            if lo <= tp.baseline_probability(it) <= hi:
                chosen.append(it)
                if len(chosen) == self.cfg.n_items:
                    break
        if len(chosen) < self.cfg.n_items:        # very mild or very severe patient: take the nearest
            rest = [self.bank[i] for i in hard if self.bank[i] not in chosen]
            rest.sort(key=lambda it: min(abs(tp.baseline_probability(it) - lo), abs(tp.baseline_probability(it) - hi)))
            chosen += rest[: self.cfg.n_items - len(chosen)]
        return tp, chosen

    # ------------------------------------------------------------------ objective
    def retention(self, day: float) -> float:
        """Mean predicted unaided P(correct) over the practised words at `day` (sim-only)."""
        return float(np.mean([self.tp.retrievability(it, day) for it in self.items]))

    # ------------------------------------------------------------------ observable state
    def observable(self) -> dict:
        """Everything a real session would know, for rule-based policies."""
        return {"last_trials": self.history, "items": self.stats, "session": self.session, "n_sessions": self.n_sessions,
                "trial_in_session": self.trial_in_session, "trials_per_session": self.cfg.trials_per_session,
                "n_items": len(self.items), "last_item": self.last_item}

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        trials = self.history[-self.cfg.history:]
        offset = (self.cfg.history - len(trials)) * PER_TRIAL
        for j, t in enumerate(trials):
            b = offset + j * PER_TRIAL
            obs[b] = 1.0
            obs[b + 1 + CUE_KINDS.index(t["cue"])] = 1.0
            obs[b + 1 + len(CUE_KINDS) + ITEM_RULES.index(t["rule"])] = 1.0
            obs[b + 1 + len(CUE_KINDS) + len(ITEM_RULES) + RESPONSE_TYPES.index(t["category"])] = 1.0
        c = self.cfg.history * PER_TRIAL
        sess = [t for t in self.history if t["session"] == self.session]
        obs[c + 0] = self.session / max(self.n_sessions - 1, 1)
        obs[c + 1] = self.trial_in_session / self.cfg.trials_per_session
        obs[c + 2] = self.cfg.schedule_days[self.session] / max(self.cfg.test_days)
        obs[c + 3] = float(np.mean([t["correct"] for t in sess])) if sess else 0.0
        obs[c + 4] = float(np.mean([t["correct"] for t in self.history])) if self.history else 0.0
        obs[c + 5] = float(np.mean([s.attempts > 0 for s in self.stats]))
        rec = [min(self.trial_index - s.last_trial, self.cfg.trials_per_session) / self.cfg.trials_per_session
               for s in self.stats if s.last_trial >= 0]
        obs[c + 6] = float(np.mean(rec)) if rec else 1.0
        obs[c + 7] = float(np.mean([s.accuracy < 0.5 for s in self.stats if s.attempts > 0])) if rec else 0.0
        return obs

    # ------------------------------------------------------------------ masks and item rules
    def action_masks(self) -> np.ndarray:
        mask = np.ones(N_ACTIONS, dtype=bool)
        if self.last_item is None or self.trial_in_session == 0:
            for c in CUE_KINDS:
                mask[action_index("repeat", c)] = False
        if self.cfg.whole_word_only_after_error:
            last = self.history[-1] if self.history else None
            if last is None or last["correct"]:
                for r in ITEM_RULES:
                    mask[action_index(r, "whole_word")] = False
        return mask

    def _pick_item(self, rule: str) -> int:
        n = len(self.items)
        if rule == "repeat" and self.last_item is not None:
            return self.last_item
        attempted = [i for i in range(n) if self.stats[i].attempts > 0]
        if rule == "weakest" and attempted:
            return min(attempted, key=lambda i: (self.stats[i].accuracy, self.stats[i].last_trial))
        if rule == "strongest" and attempted:
            return max(attempted, key=lambda i: (self.stats[i].accuracy, -self.stats[i].last_trial))
        return min(range(n), key=lambda i: (self.stats[i].last_trial, i))      # due

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        k = int(options["patient_index"]) if "patient_index" in options else int(self.np_random.integers(2**31 - 1))
        self.patient_index = k
        self.tp, self.items = self._build_patient(k)
        self.stats = [ItemStats() for _ in self.items]
        self.history: list[dict] = []
        self.session, self.trial_in_session, self.trial_index = 0, 0, 0
        self.last_item: int | None = None
        self.start_retention = {d: self.retention(d) for d in self.cfg.test_days}
        self._value = self.retention(self.cfg.test_days[0])
        return self._obs(), {"patient_index": k, "aq": self.tp.snapshot.aq, "s": self.tp.snapshot.s, "p": self.tp.snapshot.p,
                             "start_retention": self.start_retention}

    def _now(self) -> float:
        return self.cfg.schedule_days[self.session] + self.trial_in_session * self.tp.learning.p.trial_seconds * SECOND

    def step(self, action: int):
        mask = self.action_masks()
        if not mask[int(action)]:
            if self.strict_masks:
                raise ValueError(f"masked action {action}: {ACTIONS[int(action)]} at session {self.session} trial {self.trial_in_session}")
            action = int(np.flatnonzero(mask)[0])
        rule, cue_kind = ACTIONS[int(action)]
        i = self._pick_item(rule)
        item = self.items[i]
        cue = Cue(cue_kind) if cue_kind else None
        r = self.tp.attempt(item, self._now(), cue)
        st = self.stats[i]
        st.attempts += 1; st.correct += int(r.correct); st.last_trial = self.trial_index; st.session_presentations += 1
        rec = {"item": i, "word": item.word, "rule": rule, "cue": cue_kind, "correct": r.correct, "category": r.category,
               "session": self.session, "trial": self.trial_index, "day": round(self._now(), 4)}
        self.history.append(rec)
        self.last_item = i
        self.trial_index += 1; self.trial_in_session += 1

        if self.cfg.reward == "privileged":
            v = self.retention(self.cfg.test_days[0])
            reward = 100.0 * (v - self._value)
            self._value = v
        else:
            effort = cue.effort(self.tp.snapshot.cue_cfg) if cue else 0.0
            reward = (1.0 - effort) if r.correct else 0.0

        terminated = False
        if self.trial_in_session >= self.cfg.trials_per_session:
            self.session += 1; self.trial_in_session = 0; self.last_item = None
            for s in self.stats:
                s.session_presentations = 0
            if self.session >= self.n_sessions:
                terminated = True
                self.session = self.n_sessions - 1          # keep indices valid for the final observation
        info: dict = {"response": rec}
        if terminated:
            info["retention"] = {d: self.retention(d) for d in self.cfg.test_days}
            info["start_retention"] = self.start_retention
            info["gain"] = {d: info["retention"][d] - self.start_retention[d] for d in self.cfg.test_days}
            info["history"] = self.history
        return self._obs(), float(reward), terminated, False, info
