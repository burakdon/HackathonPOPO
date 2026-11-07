"""Prompt dataset and reward for GRPO, built offline from the simulator (no API calls).

States come from sessions where the contextual bandit plays simulated TRAINING patients. At each
visited state we store the State JSON, the action mask, and the patient's true mean p(correct) for
every (tier, cue) at that moment (hidden from the model). The reward of a completion is then the
expected band-tracking reward of the action it names.
"""
from __future__ import annotations

import numpy as np

from agents.llm_prompt import parse_action
from agents.logistic_bandit import LogisticThompsonBandit
from agents.selection import clinical_gate
from core.config import ClinicalConfig, load_clinical, load_logistic_bandit
from core.masks import ACTIONS, action_index, action_mask
from core.schema import MAX_LAST_TRIALS, Outcome, State
from sim.config import SimConfig, load_sim
from sim.items import load_items
from sim.population import make_patients


def build_dataset(n: int, seed: int, sim_cfg: SimConfig | None = None,
                  clinical: ClinicalConfig | None = None) -> list[dict]:
    cfg, clinical = sim_cfg or load_sim(), clinical or load_clinical()
    bank, lcfg = load_items(cfg.items), load_logistic_bandit()
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    k = 0
    while len(out) < n:
        spec = make_patients(cfg, "population", "train", 1, root_seed=seed, start=k)[0]
        patient, selector = spec.build(bank, cfg, clinical)
        agent = LogisticThompsonBandit(lcfg, clinical, seed=seed * 100_000 + k)
        tier, history = cfg.session.start_tier, []
        for t in range(cfg.session.n_trials):
            state = State(last_trials=history[-MAX_LAST_TRIALS:], current_tier=tier, trial_count=t,
                          session_minutes=round(t * 0.15, 2))
            mask = clinical_gate(state, action_mask(tier, clinical.approved_cues, clinical.max_tier_step),
                                 lcfg.cue_only_after_error)
            if rng.random() < 0.25:   # subsample states so one session does not dominate
                out.append({"state": state.model_dump(), "mask": mask.tolist(),
                            "p_true": [patient.p_correct_tier(tt, c) for tt, c in ACTIONS],
                            "patient": k, "trial": t})
            a = agent.act(state, mask)
            r = patient.respond(selector.next(a.tier), a.cue)
            agent.update(state, a, Outcome(correct=r.correct, error_type=r.error_type, response_time=r.response_time))
            history.append({"word": "x", "tier": a.tier, "cue": a.cue, "outcome": r.correct,
                            "error_type": r.error_type, "response_time": r.response_time})
            tier = a.tier
            if len(out) >= n:
                break
        k += 1
    return out


def reward(text: str, example: dict, clinical: ClinicalConfig, cue_cost: float, tier_bonus: float,
           format_penalty: float) -> float:
    parsed = parse_action(text)
    if parsed is None:
        return format_penalty
    i = action_index(*parsed)
    if not example["mask"][i]:
        return format_penalty
    tier, cue = parsed
    return (-abs(example["p_true"][i] - clinical.target) - cue_cost * clinical.cue_strength[cue]
            + tier_bonus * tier)
