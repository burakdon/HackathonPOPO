"""Prompted LLM baseline, with the HTTP API mocked (no key or network needed)."""
import httpx
import numpy as np

from agents.llm_prompted import GeminiPolicy
from core.config import load_clinical, load_yaml
from core.masks import action_mask
from core.schema import State


def fake_client(answer: str, calls: list):
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": answer}]}}]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def make(tmp_path, answer, calls, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg = {**load_yaml("configs/llm_baseline.yaml"), "cache_path": str(tmp_path / "cache.jsonl")}
    return GeminiPolicy(load_clinical(), cfg=cfg, client=fake_client(answer, calls))


def test_valid_answer_is_used_and_cached(tmp_path, monkeypatch):
    calls = []
    pol = make(tmp_path, '{"tier": 2, "cue": "none"}', calls, monkeypatch)
    st = State(current_tier=1, trial_count=0, session_minutes=0.0)
    mask = action_mask(1, load_clinical().approved_cues, 1)
    a = pol.act(st, mask)
    assert (a.tier, a.cue) == (2, "none")
    pol.act(st, mask)                                  # same prompt -> cache hit, no second call
    assert len(calls) == 1 and pol.stats["cache_hits"] == 1
    again = make(tmp_path, '{"tier": 1, "cue": "none"}', calls, monkeypatch)   # cache persists on disk
    assert (again.act(st, mask).tier, len(calls)) == (2, 1)


def test_masked_or_garbled_answers_fall_back_and_are_counted(tmp_path, monkeypatch):
    st = State(current_tier=1, trial_count=0, session_minutes=0.0)
    mask = action_mask(1, load_clinical().approved_cues, 1)
    for i, answer in enumerate(['{"tier": 3, "cue": "none"}', "I think tier two"]):
        pol = make(tmp_path / str(i), answer, [], monkeypatch)
        a = pol.act(st, mask)
        assert (a.tier, a.cue) == (1, "none") and pol.stats["invalid"] == 1
