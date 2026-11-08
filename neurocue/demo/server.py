"""NeuroCue patient console: a small FastAPI service over the engine, and the page that drives it.

  uv run python -m neurocue.demo.server            (http://localhost:8765)

Everything the page shows is computed by the same code the validation ran: the patient is
patientmodel.Patient on the Dell model with the full word realizer; the diagnosis is the grid
posterior the fitter uses; the retention curves are the fitted learning layer inside the therapy
environment; the override is agents/ditto.py on the trained MaskablePPO policy.
"""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from neurocue.baselines import BASELINES
from neurocue.env import ACTIONS, EngineTherapyEnv, action_index, load_engine_env_config
from patientmodel.dell_sp import NAMING_CATEGORIES
from patientmodel.items import Item, load_item_bank
from patientmodel.lexicon import build_lexicon
from patientmodel.patient import Patient
from patientmodel.population import PopulationConfig, PopulationSampler, expected_accuracy
from patientmodel.realize import Realizer
from patientmodel.treatment import TreatablePatient

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
PPO_PATH = ROOT / "models" / "engine_ppo_privileged_seed0.zip"
LESION_PRIOR = {"semantic": 1.0, "balanced": 0.0, "phonological": -1.0}   # log p/s centre per lesion type

# ARPAbet -> rough English respelling, so a browser speech engine can say a nonword.
RESPELL = {"AA": "ah", "AE": "a", "AH": "u", "AO": "aw", "AW": "ow", "AY": "igh", "B": "b", "CH": "ch", "D": "d", "DH": "th",
           "EH": "e", "ER": "er", "EY": "ay", "F": "f", "G": "g", "HH": "h", "IH": "i", "IY": "ee", "JH": "j", "K": "k",
           "L": "l", "M": "m", "N": "n", "NG": "ng", "OW": "oh", "OY": "oy", "P": "p", "R": "r", "S": "s", "SH": "sh",
           "T": "t", "TH": "th", "UH": "oo", "UW": "oo", "V": "v", "W": "w", "Y": "y", "Z": "z", "ZH": "zh"}


def respell(phonemes) -> str:
    return "".join(RESPELL.get(p, p.lower()) for p in phonemes) if phonemes else ""


class Shared:
    env: EngineTherapyEnv
    realizer: Realizer
    pictures: list[Item]
    ppo = None

    @classmethod
    def load(cls) -> None:
        cfg = load_engine_env_config(); cfg.probe_trials = 500
        cls.env = EngineTherapyEnv(cfg, split="test", root_seed=7)
        bank = cls.env.bank
        lex = build_lexicon(extra_words=[i.word for i in bank])
        cls.realizer = Realizer(lex)
        cls.pictures = sorted([i for i in load_item_bank(require_image=True) if i.pronounceable and i.word in lex.index],
                              key=lambda i: i.word)
        if PPO_PATH.exists():
            from sb3_contrib import MaskablePPO
            cls.ppo = MaskablePPO.load(str(PPO_PATH), device="cpu")


@dataclass
class DemoPatient:
    id: str
    aq: float
    lesion: str
    seed: int
    tp: TreatablePatient
    order: list[int]
    cursor: int = 0
    counts: np.ndarray = field(default_factory=lambda: np.zeros(len(NAMING_CATEGORIES)))
    responses: list[dict] = field(default_factory=list)
    ppo_run: dict | None = None
    tuned = None


PATIENTS: dict[str, DemoPatient] = {}
app = FastAPI(title="NeuroCue patient console")


def make_patient(aq: float, lesion: str, seed: int) -> TreatablePatient:
    from dataclasses import replace
    pc = replace(PopulationConfig(), log_ps_ratio_mean=LESION_PRIOR[lesion], log_ps_ratio_sd=0.25)
    sampler = PopulationSampler(Shared.env.table, pc)
    rng = np.random.default_rng(seed)
    s, p = sampler.weights_for_accuracy(float(expected_accuracy(aq, pc)), rng)
    pt = Patient(f"demo-{seed}", s, p, Shared.env.model, Shared.realizer, aq=aq, rng=np.random.default_rng(seed + 1))
    return TreatablePatient(pt, probe_trials=500)


def posterior(counts: np.ndarray) -> dict:
    t = Shared.env.table
    ll = (counts[None, None, :] * np.log(np.clip(t.probs, 1e-4, None))).sum(-1)
    w = np.exp(ll - ll.max()); w /= w.sum()
    i, j = np.unravel_index(int(w.argmax()), w.shape)
    ms = float((w.sum(1) * t.s_values).sum()); mp = float((w.sum(0) * t.p_values).sum())
    acc = float(t.probs[i, j, NAMING_CATEGORIES.index("correct")])
    lr = np.log(t.p_values[j] / max(t.s_values[i], 1e-6))
    kind = "semantic" if lr > 0.3 else "phonological" if lr < -0.3 else "balanced"
    return {"s_values": t.s_values.tolist(), "p_values": t.p_values.tolist(), "grid": (w / w.max()).tolist(),
            "map": {"s": float(t.s_values[i]), "p": float(t.p_values[j])}, "mean": {"s": ms, "p": mp},
            "n": int(counts.sum()), "predicted_accuracy": acc, "lesion_read": kind}


class NewPatient(BaseModel):
    aq: float = 60.0
    lesion: str = "balanced"
    seed: int = 1


@app.on_event("startup")
def _startup() -> None:
    Shared.load()


@app.post("/api/patient")
def new_patient(req: NewPatient) -> dict:
    if req.lesion not in LESION_PRIOR:
        raise HTTPException(400, "lesion must be semantic | balanced | phonological")
    tp = make_patient(req.aq, req.lesion, req.seed)
    pid = uuid.uuid4().hex[:8]
    order = np.random.default_rng(req.seed + 2).permutation(len(Shared.pictures)).tolist()
    PATIENTS[pid] = DemoPatient(pid, req.aq, req.lesion, req.seed, tp, order)
    probe = [Shared.pictures[k] for k in order[:12]]
    acc = float(np.mean([tp.baseline_probability(it) for it in probe]))
    return {"id": pid, "aq": req.aq, "lesion": req.lesion, "s": tp.snapshot.s, "p": tp.snapshot.p, "expected_accuracy": acc,
            "ppo_available": Shared.ppo is not None}


def _get(pid: str) -> DemoPatient:
    if pid not in PATIENTS:
        raise HTTPException(404, "unknown patient")
    return PATIENTS[pid]


@app.post("/api/patient/{pid}/name")
def name_picture(pid: str) -> dict:
    d = _get(pid)
    item = Shared.pictures[d.order[d.cursor % len(d.order)]]; d.cursor += 1
    r = d.tp.snapshot.name(item)
    cat = r.category if r.category in NAMING_CATEGORIES else "nonword"
    d.counts[NAMING_CATEGORIES.index(cat)] += 1
    phon = list(r.phonemes) if r.phonemes else None
    rec = {"n": len(d.responses) + 1, "target": item.word, "item_id": item.item_id, "image": f"/api/image/{item.item_id}",
           "category": r.category, "correct": r.correct, "word": r.word, "phonemes": phon,
           "speak": r.word if r.word else respell(phon), "duration_s": round(r.duration_s, 2),
           "p_correct": d.tp.baseline_probability(item)}
    d.responses.append(rec)
    return {"response": rec, "posterior": posterior(d.counts)}


@app.get("/api/patient/{pid}/truth")
def truth(pid: str) -> dict:
    d = _get(pid)
    return {"s": d.tp.snapshot.s, "p": d.tp.snapshot.p, "aq": d.aq, "lesion": d.lesion}


@app.get("/api/image/{item_id}")
def image(item_id: str):
    for it in Shared.pictures:
        if it.item_id == item_id and it.image:
            return FileResponse(str(it.image))
    raise HTTPException(404, "no picture")


def run_course(policy, tp: TreatablePatient, item_seed: int, record_obs: bool = False) -> dict:
    env = Shared.env
    obs, info = env.reset(options={"patient": tp, "item_seed": item_seed})
    rng = np.random.default_rng(0)
    trials, obs_log, mask_log = [], [], []
    start = info["start_retention"]
    done = False
    while not done:
        mask = env.action_masks()
        if hasattr(policy, "predict"):
            a, _ = policy.predict(obs, action_masks=mask, deterministic=True); a = int(a)
        else:
            a = policy.act(env.observable(), mask, rng)
        if record_obs:
            obs_log.append(obs.copy()); mask_log.append(mask.copy())
        obs, _, done, _, info = env.step(a)
        rec = dict(info["response"]); rec["retention"] = info["retention_now"]; rec["action"] = a
        trials.append(rec)
    out = {"trials": trials, "start": {str(int(k)): v for k, v in start.items()},
           "final": {str(int(k)): v for k, v in info["retention"].items()},
           "words": [it.word for it in env.items]}
    if record_obs:
        out["_obs"], out["_masks"] = obs_log, mask_log
    return out


class CourseReq(BaseModel):
    policies: list[str] = ["spaced_hierarchy", "ppo"]


@app.post("/api/patient/{pid}/course")
def course(pid: str, req: CourseReq) -> dict:
    d = _get(pid)
    out = {}
    for name in req.policies:
        tp = make_patient(d.aq, d.lesion, d.seed)        # a fresh copy: same parameters, same random stream
        if name == "ppo":
            if Shared.ppo is None:
                out[name] = {"unavailable": True}; continue
            res = run_course(Shared.ppo, tp, d.seed, record_obs=True)
            d.ppo_run = res
            res = {k: v for k, v in res.items() if not k.startswith("_")}
        elif name == "tuned":
            if d.tuned is None:
                out[name] = {"unavailable": True}; continue
            res = run_course(d.tuned, tp, d.seed)
        else:
            res = run_course(BASELINES[name](), tp, d.seed)
        out[name] = res
    return out


class OverrideReq(BaseModel):
    trial: int
    cue: str | None      # None | phonemic_initial | semantic | whole_word


@app.post("/api/patient/{pid}/override")
def override(pid: str, req: OverrideReq) -> dict:
    d = _get(pid)
    if Shared.ppo is None or d.ppo_run is None:
        raise HTTPException(400, "run the trained policy's course first")
    from agents.ditto import Demo, action_probs, ditto
    t = req.trial
    if not 0 <= t < len(d.ppo_run["trials"]):
        raise HTTPException(400, "trial out of range")
    rule = d.ppo_run["trials"][t]["rule"]
    a = action_index(rule, req.cue)
    obs, mask = d.ppo_run["_obs"][t], d.ppo_run["_masks"][t]
    if not mask[a]:
        raise HTTPException(400, "that cue is not allowed in this state")
    base = d.tuned if d.tuned is not None else Shared.ppo
    # Gentle update: with one demonstration and a small network, a strong update rewrites the policy
    # everywhere (measured: lr 2e-3 x 6 rounds changed all 160 states). lr 1e-3 x 4 rounds changed 16 of
    # 160, all of them states that followed an error, none elsewhere.
    tuned, log = ditto(base, [Demo(obs=obs, mask=mask, action=a)], rounds=4, samples_per_state=8, beta=0.5, lr=1e-3,
                       steps_per_round=10, seed=0)
    d.tuned = tuned
    # Where did the policy change? Compare the chosen action over every state of the recorded course.
    O, M = np.stack(d.ppo_run["_obs"]), np.stack(d.ppo_run["_masks"])
    before = action_probs(Shared.ppo, O, M).argmax(1); after = action_probs(tuned, O, M).argmax(1)
    changed = [int(i) for i in np.flatnonzero(before != after)]
    same_context = [i for i in range(len(O)) if i != t and d.ppo_run["trials"][i]["category"] == d.ppo_run["trials"][t - 1]["category"]] if t > 0 else []
    tp = make_patient(d.aq, d.lesion, d.seed)
    res = run_course(tuned, tp, d.seed)
    return {"demo_prob": log[-1]["demo_prob"], "changed_states": changed, "n_states": int(len(O)),
            "changed_in_same_context": len([i for i in changed if i in same_context]), "n_same_context": len(same_context),
            "course": res}


app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
