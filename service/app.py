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
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from agents.base import Policy
from agents.beta_bandit import BetaThompsonBandit
from agents.logistic_bandit import LogisticThompsonBandit
from core.config import load_beta_bandit, load_clinical, load_logistic_bandit, load_service, resolve
from core.masks import action_mask
from core.schema import (Action, LearnRequest, LearnResult, NextTrial, OverrideRequest, State,
                         UpdateRequest)
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

    def replace(self, session_id: str, agent: Policy) -> None:
        with self._lock:
            self._agents[session_id] = agent

    def lock(self) -> threading.Lock:
        return self._lock


def agent_factory(agent: str, cfg, clinical):
    """session_id -> fresh agent for that session, seeded deterministically from the session id."""
    if agent == "llm":
        # Language-model policy behind the original /act contract. The model is loaded once per
        # worker (shared across sessions); each session gets its own bandit as the explicit fallback.
        from agents.llm_policy import LLMActPolicy
        from agents.registry import ModelRegistry
        from core.practice import load_practice
        if not cfg.llm_model:
            raise ValueError("agent: llm needs llm_model in configs/service.yaml")
        pcfg = load_practice(cfg.practice_config)
        registry = ModelRegistry(pcfg)
        bcfg = load_beta_bandit(cfg.bandit_config)
        return lambda sid: LLMActPolicy(registry, cfg.llm_model, pcfg, clinical,
                                        BetaThompsonBandit(bcfg, clinical, seed=session_seed(bcfg.seed, sid)), bcfg)
    if agent == "ppo":
        from sb3_contrib import MaskablePPO

        from agents.ppo import PPOPolicy
        from envs.aphasia_env import load_env_config
        env_cfg = load_env_config()
        shared = MaskablePPO.load(str(resolve(cfg.ppo_model)), device="cpu")   # loaded once, read-only
        return lambda sid: PPOPolicy(shared, clinical, env_cfg)
    if agent == "logistic_bandit":
        lcfg = load_logistic_bandit(cfg.logistic_bandit_config)
        return lambda sid: LogisticThompsonBandit(lcfg, clinical, seed=session_seed(lcfg.seed, sid))
    bcfg = load_beta_bandit(cfg.bandit_config)
    return lambda sid: BetaThompsonBandit(bcfg, clinical, seed=session_seed(bcfg.seed, sid))


def create_app(config_path: str | Path | None = None,
               overrides_path: str | Path | None = None,
               agent: str | None = None,
               speech_provider=None,
               cueagain_root: str | Path | None = None,
               cueagain_decider=None,
               rule_drafter=None,
               cueagain_trainer=None,
               cueagain_previewer=None) -> FastAPI:
    """`cueagain_root` / `cueagain_decider`: where the patient practice API (service/patient_api.py) keeps
    its data (default runs/cueagain, or $CUEAGAIN_DATA) and which decision model it asks (default: each
    patient's active model). Tests pass a temporary directory and the rules decider."""
    cfg = load_service(config_path or os.environ.get("AGENT_SERVICE_CONFIG", "configs/service.yaml"))
    # Optional overrides (e.g. to run several agents side by side): argument, then environment variable.
    agent = agent or os.environ.get("AGENT_SERVICE_AGENT")
    if agent:
        cfg = cfg.model_copy(update={"agent": agent})
    if os.environ.get("AGENT_SERVICE_PPO_MODEL"):
        cfg = cfg.model_copy(update={"ppo_model": os.environ["AGENT_SERVICE_PPO_MODEL"]})
    clinical = load_clinical(cfg.clinical_config)
    overrides_file = resolve(overrides_path or cfg.overrides_path)

    store = SessionStore(agent_factory(cfg.agent, cfg, clinical))
    # Per-session log for live learning: states the agent acted on, and clinician overrides.
    acted: dict[str, list[State]] = {}
    overrides_log: dict[str, list[tuple[State, Action]]] = {}
    live = None
    if cfg.agent == "ppo":
        from sb3_contrib import MaskablePPO

        from envs.aphasia_env import load_env_config
        from service.live_ditto import LiveDitto
        live = LiveDitto(MaskablePPO.load(str(resolve(cfg.ppo_model)), device="cpu"), clinical,
                         load_env_config(), cfg.live_ditto)
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
    app.state.clinical = clinical

    def sid(state: State) -> str:
        return state.session_id or cfg.default_session_id

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "agent": cfg.agent, "active_sessions": len(store)}

    @app.post("/act", response_model=Action)
    def act(state: State, response: Response = None) -> Action:
        session = sid(state)
        agent = store.get(session)
        mask = action_mask(state.current_tier, clinical.approved_cues, clinical.max_tier_step)
        with store.lock():
            action = agent.act(state, mask)
            acted.setdefault(session, []).append(state)
        # Which model actually produced the action, without changing the Action body (headers).
        if response is not None:
            last = getattr(agent, "last_decision", None)
            response.headers["X-Decision-Source"] = last.source if last else cfg.agent
            if last is not None:
                response.headers["X-Requested-Model"] = last.requested_model
                if last.fallback_reason:
                    response.headers["X-Fallback-Reason"] = last.fallback_reason[:200]
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

    seen_updates: dict[tuple[str, str], str] = {}     # (session, attempt_id) -> content fingerprint

    @app.post("/update")
    def update(req: UpdateRequest) -> dict:
        session = sid(req.state)
        if req.attempt_id is not None:   # idempotent: an attempt is learned from at most once
            import hashlib
            fp = hashlib.sha256(req.model_dump_json(exclude={"attempt_id"}).encode()).hexdigest()
            with store.lock():
                prev = seen_updates.get((session, req.attempt_id))
                if prev is not None:
                    if prev != fp:
                        raise HTTPException(409, f"attempt {req.attempt_id} was already submitted with different content")
                    return {"status": "duplicate", "session_id": session, "agent_updated": False}
                seen_updates[(session, req.attempt_id)] = fp
        if req.probe:      # review probes are not the agent's choice: nothing to learn from
            return {"status": "ok", "session_id": session, "agent_updated": False}
        agent = store.get(session)
        with store.lock():
            agent.update(req.state, req.action, req.outcome)
        return {"status": "ok", "session_id": session, "agent_updated": True}

    @app.post("/learn_from_overrides", response_model=LearnResult)
    def learn_from_overrides(req: LearnRequest) -> LearnResult:
        """One DITTO-style round on this session's overrides; hot-swaps this session's policy only."""
        if live is None:
            raise HTTPException(409, "live learning requires the ppo agent (configs/service.yaml)")
        logged = overrides_log.get(req.session_id, [])
        if not logged:
            raise HTTPException(400, "no overrides logged in this session")
        overridden = {s.trial_count for s, _ in logged}
        accepted = [s for s in acted.get(req.session_id, []) if s.trial_count not in overridden]
        try:
            tuned, report = live.learn(logged, accepted)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        from agents.ppo import PPOPolicy
        store.replace(req.session_id, PPOPolicy(tuned, clinical, live.env_cfg))
        return LearnResult(session_id=req.session_id, **report)

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
        overrides_log.setdefault(record["session_id"], []).append((req.state, req.clinician_action))
        with write_lock:
            overrides_file.parent.mkdir(parents=True, exist_ok=True)
            with open(overrides_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            with open(overrides_file) as f:
                n = sum(1 for _ in f)
        return {"status": "stored", "session_id": record["session_id"], "n_overrides": n}

    # ---- Practice flow (patient practice, speech, clinician feedback, model versions) and the web app
    from agents.registry import ModelRegistry
    from core.practice import load_practice
    from service.practice import practice_router
    pcfg = load_practice(cfg.practice_config)
    registry = ModelRegistry(pcfg)
    app.state.registry = registry
    app.include_router(practice_router(pcfg, clinical, load_beta_bandit(cfg.bandit_config), registry,
                                       speech_provider=speech_provider))

    # Patient practice API of docs/API_CONTRACT.md: automatic scoring, model-chosen next steps.
    from core.contract import NewPatient
    from core.practice import load_items as load_pictures
    from service.engine import Engine
    from service.patient_api import default_model, model_decider, patient_router
    speech_holder = {"p": speech_provider}

    def get_speech():
        if speech_holder["p"] is None:
            try:
                from speech.whisper_local import WhisperLocalProvider
                speech_holder["p"] = WhisperLocalProvider()
            except ImportError as e:
                from speech.base import UnavailableProvider
                speech_holder["p"] = UnavailableProvider(f"speech extra not installed ({e.name})")
        return speech_holder["p"]

    engine = Engine(resolve(cueagain_root or os.environ.get("CUEAGAIN_DATA", "runs/cueagain")),
                    load_pictures(pcfg.items_path), decider=cueagain_decider or model_decider(registry, lambda v: engine.adapter_version(v)),
                    hint_audio_dir=resolve("frontend/hints"),
                    default_model="rules" if cueagain_decider else default_model(registry))
    if not engine.patients:
        engine.add_patient(NewPatient(name="Ana (demo)", first="Ana", about="Demo patient for trying the app."),
                           pid="demo-ana", simulated=True)
    from service.preview import ModelPreviewer
    from service.training import SubprocessTrainer, TrainingManager
    engine.training = TrainingManager(
        engine, cueagain_trainer or SubprocessTrainer(resolve(pcfg.llm.runs_dir), engine.root / "training"),
        cueagain_previewer or ModelPreviewer(cueagain_decider or model_decider(registry, lambda v: engine.adapter_version(v))))
    app.state.engine = engine
    if rule_drafter is None:
        from service.rule_drafter import ClaudeRuleDrafter
        rule_drafter = ClaudeRuleDrafter()
    app.include_router(patient_router(engine, get_speech, registry, rule_drafter))
    app.mount("/media/images", StaticFiles(directory=str(resolve("frontend/images"))), name="media-images")
    app.mount("/media/hints", StaticFiles(directory=str(resolve("frontend/hints")), check_dir=False), name="media-hints")
    if resolve("product").exists():
        app.mount("/product", StaticFiles(directory=str(resolve("product")), html=True), name="product")
    frontend = resolve("frontend")
    if frontend.exists():
        app.mount("/app", StaticFiles(directory=str(frontend), html=True), name="app")

        @app.get("/", include_in_schema=False)
        def root():
            return RedirectResponse("/app/")

    return app


app = create_app()
