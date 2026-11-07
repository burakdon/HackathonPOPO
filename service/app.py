"""FastAPI agent service the frontend calls.

Run:  uv run uvicorn service.app:app --host 0.0.0.0 --port 8000   (0.0.0.0: reachable from the LAN)

One online-learning agent instance per session_id, held in memory. Each session's random seed
is derived deterministically from (config seed, session_id), so replaying the same requests
reproduces the same actions. Trade-off: sessions are lost on restart; acceptable for a demo,
and a session that restarts simply begins again from the prior.
"""
from __future__ import annotations

import json
import os
import threading
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.base import Policy
from agents.beta_bandit import BetaThompsonBandit
from agents.logistic_bandit import LogisticThompsonBandit
from core.config import load_beta_bandit, load_clinical, load_logistic_bandit, load_service, resolve
from core.masks import action_mask
from core.schema import Action, NextTrial, OverrideRequest, State, UpdateRequest
from sim.config import load_sim
from sim.items import ItemSelector, load_items


def session_seed(base_seed: int, session_id: str) -> int:
    """Stable per-session seed. crc32, not hash(): Python's hash() is salted per process."""
    return int(np.random.SeedSequence([base_seed, zlib.crc32(session_id.encode())]).generate_state(1)[0])


class SessionStore:
    """Thread-safe map session_id -> agent. FastAPI runs sync endpoints in a thread pool."""

    def __init__(self, factory) -> None:
        self._factory = factory
        self._agents: dict[str, Policy] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Policy:
        with self._lock:
            if session_id not in self._agents:
                self._agents[session_id] = self._factory(session_id)
            return self._agents[session_id]

    def __len__(self) -> int:
        return len(self._agents)

    def lock(self) -> threading.Lock:
        return self._lock


def agent_factory(agent: str, cfg, clinical):
    """session_id -> fresh agent for that session, seeded deterministically from the session id."""
    if agent == "logistic_bandit":
        lcfg = load_logistic_bandit(cfg.logistic_bandit_config)
        return lambda sid: LogisticThompsonBandit(lcfg, clinical, seed=session_seed(lcfg.seed, sid))
    bcfg = load_beta_bandit(cfg.bandit_config)
    return lambda sid: BetaThompsonBandit(bcfg, clinical, seed=session_seed(bcfg.seed, sid))


class DittoTrainRequest(BaseModel):
    variant: str | None = None      # paper | legacy (default: configs/ditto.yaml live.variant)


def create_app(config_path: str | Path | None = None,
               overrides_path: str | Path | None = None,
               agent: str | None = None,
               accepted_path: str | Path | None = None) -> FastAPI:
    cfg = load_service(config_path or os.environ.get("AGENT_SERVICE_CONFIG", "configs/service.yaml"))
    agent = agent or os.environ.get("AGENT")
    if agent:
        cfg = cfg.model_copy(update={"agent": agent})
    clinical = load_clinical(cfg.clinical_config)
    overrides_file = resolve(overrides_path or cfg.overrides_path)
    accepted_file = resolve(accepted_path or cfg.accepted_path)

    live = None
    if cfg.agent == "ppo":
        from service.live_ditto import LiveDitto
        live = LiveDitto(cfg.ppo_model, cfg.ditto_config, clinical)
        store = SessionStore(lambda sid: live.policy)      # stateless policy shared by all sessions
    else:
        store = SessionStore(agent_factory(cfg.agent, cfg, clinical))
    # For the accepted-choice log: the agent's last action per session, and which trials were overridden.
    acted: dict[str, tuple[int, Action]] = {}
    overridden: set[tuple[str, int]] = set()
    # Item bank and review-probe rule shared with the simulator, for POST /next_trial.
    sim_cfg = load_sim()
    bank = load_items(sim_cfg.items)
    selectors: dict[str, ItemSelector] = {}
    write_lock = threading.Lock()

    app = FastAPI(title="Aphasia Trainer agent service", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins,
                       allow_origin_regex=cfg.cors_origin_regex,
                       allow_methods=["*"], allow_headers=["*"])
    app.state.store = store
    app.state.agent = cfg.agent
    app.state.live = live
    app.state.clinical = clinical

    def sid(state: State) -> str:
        return state.session_id or cfg.default_session_id

    def append_jsonl(path: Path, record: dict) -> int:
        with write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
            with open(path) as f:
                return sum(1 for _ in f)

    def count_lines(path: Path) -> int:
        return sum(1 for line in open(path) if line.strip()) if path.exists() else 0

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "agent": live.label() if live else cfg.agent, "active_sessions": len(store)}

    @app.post("/act", response_model=Action)
    def act(state: State) -> Action:
        session = sid(state)
        agent = store.get(session)
        mask = action_mask(state.current_tier, clinical.approved_cues, clinical.max_tier_step)
        with store.lock():
            action = agent.act(state, mask)
            acted[session] = (state.trial_count, action)
        return action

    @app.post("/next_trial", response_model=NextTrial)
    def next_trial(state: State) -> NextTrial:
        """The whole next trial in one call: a review probe if one is due (chosen by the shared rule,
        no cue), otherwise the agent's tier and cue plus the item chosen within that tier. Call it
        once per trial, then POST /update with `probe` set as returned."""
        session = sid(state)
        with store.lock():
            if session not in selectors:
                selectors[session] = ItemSelector(bank, np.random.default_rng(session_seed(0, session)),
                                                  sim_cfg.review_probes)
            sel = selectors[session]
            sel.t = state.trial_count          # keep the selector's clock in step with the app
            if sel.probe_due():
                i = sel.next_probe()
                return NextTrial(word=bank.words[i], image_file=bank.image_files[i], tier=int(bank.tier[i]),
                                 cue="none", probe=True,
                                 rationale=f"Review probe: unassisted recall of a word practised earlier.")
        action = act(state)
        with store.lock():
            i = sel.next(action.tier)
        return NextTrial(word=bank.words[i], image_file=bank.image_files[i], tier=action.tier, cue=action.cue,
                         probe=False, rationale=action.rationale)

    @app.post("/update")
    def update(req: UpdateRequest) -> dict:
        session = sid(req.state)
        if req.probe:      # review probes are not the agent's choice: nothing to learn from
            return {"status": "ok", "session_id": session, "agent_updated": False}
        agent = store.get(session)
        with store.lock():
            agent.update(req.state, req.action, req.outcome)
            shown = acted.get(session)
            accepted = (req.reviewed and shown is not None and shown[0] == req.state.trial_count
                        and (session, req.state.trial_count) not in overridden
                        and shown[1].tier == req.action.tier and shown[1].cue == req.action.cue)
        if accepted:     # the clinician saw the agent's choice and did not override it: an anchor for DITTO
            append_jsonl(accepted_file, {"timestamp": datetime.now(timezone.utc).isoformat(), "session_id": session,
                                         "state": req.state.model_dump(), "action": req.action.model_dump()})
        return {"status": "ok", "session_id": session, "agent_updated": True, "accepted_logged": accepted}

    @app.post("/override")
    def override(req: OverrideRequest) -> dict:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": sid(req.state),
            "state": req.state.model_dump(),
            "agent_action": req.agent_action.model_dump(),
            "clinician_action": req.clinician_action.model_dump(),
            "note": req.note,
        }
        with store.lock():
            overridden.add((record["session_id"], req.state.trial_count))
        n = append_jsonl(overrides_file, record)
        return {"status": "stored", "session_id": record["session_id"], "n_overrides": n}

    # ---- live DITTO loop (agent == "ppo"); see service/live_ditto.py ----
    def need_live():
        if live is None:
            raise HTTPException(409, f"live DITTO needs the ppo agent (running {cfg.agent}); "
                                     f"start the service with AGENT=ppo")
        return live

    @app.get("/ditto/status")
    def ditto_status() -> dict:
        return {"agent": live.label() if live else cfg.agent, "live_ditto_available": live is not None,
                "tuned": bool(live and live.tuned), "n_overrides": count_lines(overrides_file),
                "n_accepted": count_lines(accepted_file),
                "last_train": None if not live or not live.last_report else
                {k: v for k, v in live.last_report.items() if k != "rounds"}}

    @app.post("/ditto/train")
    def ditto_train(req: DittoTrainRequest | None = None) -> dict:
        lv = need_live()
        from train.ditto_ppo import read_jsonl
        try:
            return lv.train(read_jsonl(overrides_file), read_jsonl(accepted_file), (req or DittoTrainRequest()).variant)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/ditto/reset")
    def ditto_reset() -> dict:
        need_live().reset()
        return {"status": "reset", "agent": live.label()}

    return app


app = create_app()
