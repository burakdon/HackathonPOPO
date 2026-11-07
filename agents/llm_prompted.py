"""Prompted frontier-model baseline (Gemini): the language model chooses tier and cue from the prompt
alone (agents/llm_prompt.py), with no training. Answers that cannot be parsed or name a masked action
fall back to the current tier with no cue and are counted, so the baseline is never credited with an
action it did not validly choose. Responses are cached on disk keyed by the prompt hash, which makes
reruns free and exactly reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os

import httpx
import numpy as np

from agents.llm_prompt import SYSTEM_PROMPT, parse_action, user_prompt
from core.config import ClinicalConfig, load_yaml, resolve
from core.masks import action_index
from core.schema import Action, Outcome, State

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def load_dotenv(path: str = ".env") -> None:
    p = resolve(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class GeminiPolicy:
    name = "gemini"

    def __init__(self, clinical: ClinicalConfig, seed: int = 0, cfg: dict | None = None, client=None) -> None:
        self.clinical, self.cfg = clinical, cfg or load_yaml("configs/llm_baseline.yaml")
        load_dotenv()
        self.key = os.environ.get("GEMINI_API_KEY")
        self.client = client or httpx.Client(timeout=self.cfg["timeout_s"])
        self.cache_path = resolve(self.cfg["cache_path"])
        self.cache = self._load_cache()
        self.stats = {"calls": 0, "cache_hits": 0, "invalid": 0}

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path.exists():
            return {}
        return {r["hash"]: r["text"] for r in map(json.loads, self.cache_path.read_text().splitlines())}

    def _generate(self, user: str) -> str:
        h = hashlib.sha256(f"{self.cfg['model']}|{SYSTEM_PROMPT}|{user}".encode()).hexdigest()
        if h in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[h]
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY is not set (add it to .env)")
        body = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": self.cfg["temperature"],
                                     "maxOutputTokens": self.cfg["max_output_tokens"],
                                     "responseMimeType": "application/json"}}
        r = self.client.post(API.format(model=self.cfg["model"]), params={"key": self.key}, json=body)
        r.raise_for_status()
        parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        self.stats["calls"] += 1
        self.cache[h] = text
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"hash": h, "text": text}) + "\n")
        return text

    def act(self, state: State, mask: np.ndarray) -> Action:
        text = self._generate(user_prompt(state, mask))
        parsed = parse_action(text)
        if parsed is None or not mask[action_index(*parsed)]:
            self.stats["invalid"] += 1
            return Action(tier=state.current_tier, cue="none", rationale="LLM answer invalid; fallback")
        return Action(tier=parsed[0], cue=parsed[1], rationale=f"Prompted LLM: {text[:120]}")

    def update(self, state: State, action: Action, outcome: Outcome) -> None:
        pass
