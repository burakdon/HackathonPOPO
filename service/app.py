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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.base import Policy
from agents.beta_bandit import BetaThompsonBandit
from agents.logistic_bandit import LogisticThompsonBandit
from core.config import load_beta_bandit, load_clinical, load_logistic_bandit, load_service, resolve
from core.masks import action_mask
from core.schema import Action, OverrideRequest, State, UpdateRequest


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


def create_app(config_path: str | Path | None = None,
               overrides_path: str | Path | None = None,
               agent: str | None = None) -> FastAPI:
    cfg = load_service(config_path or os.environ.get("AGENT_SERVICE_CONFIG", "configs/service.yaml"))
    if agent:
        cfg = cfg.model_copy(update={"agent": agent})
    clinical = load_clinical(cfg.clinical_config)
    overrides_file = resolve(overrides_path or cfg.overrides_path)

    store = SessionStore(agent_factory(cfg.agent, cfg, clinical))
    write_lock = threading.Lock()

    app = FastAPI(title="Aphasia Trainer agent service", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins,
                       allow_origin_regex=cfg.cors_origin_regex,
                       allow_methods=["*"], allow_headers=["*"])
    app.state.store = store
    app.state.agent = cfg.agent
    app.state.clinical = clinical

    def sid(state: State) -> str:
        return state.session_id or cfg.default_session_id

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "agent": cfg.agent, "active_sessions": len(store)}

    @app.post("/act", response_model=Action)
    def act(state: State) -> Action:
        agent = store.get(sid(state))
        mask = action_mask(state.current_tier, clinical.approved_cues, clinical.max_tier_step)
        with store.lock():
            return agent.act(state, mask)

    @app.post("/update")
    def update(req: UpdateRequest) -> dict:
        session = sid(req.state)
        agent = store.get(session)
        with store.lock():
            agent.update(req.state, req.action, req.outcome)
        return {"status": "ok", "session_id": session}

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
        with write_lock:
            overrides_file.parent.mkdir(parents=True, exist_ok=True)
            with open(overrides_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            with open(overrides_file) as f:
                n = sum(1 for _ in f)
        return {"status": "stored", "session_id": record["session_id"], "n_overrides": n}

    return app


app = create_app()
