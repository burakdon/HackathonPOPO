"""Shared language-model runtime: one base model per worker process, adapters loaded on top of it.

Used by live inference (agents/llm_policy.py), evaluation (evaluation/compare.py) and, for the
prompt template, by training (training/ditto.py). Loading the 1.7B base once and switching LoRA
adapters keeps memory at one model no matter how many sessions or versions are in use.

Heavy imports (torch, transformers, peft) happen lazily, so the baseline service runs without the
`llm` extra installed.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from core.practice import LLMConfig

# Template settings recorded in every run manifest. Qwen3: thinking disabled consistently in
# training, evaluation and serving (the template then inserts an empty <think></think> block).
TEMPLATE_SETTINGS = {"add_generation_prompt": True, "enable_thinking": False}
# Option scoring runs all options of a decision in one batch unless that batch would exceed this
# many tokens; then it scores one option at a time. A prompt with all 119 training demonstrations
# (~23k tokens) times 6 options ran out of memory on a 40 GB A100 when batched.
MAX_SCORE_BATCH_TOKENS = 16_000


def chat_prompt(tokenizer, msgs: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(msgs, tokenize=False, **TEMPLATE_SETTINGS)


def pick_device(pref: str = "auto") -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def llm_available() -> tuple[bool, str | None]:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        return False, f"llm extra not installed ({e.name}); run `uv sync --extra llm`"
    return True, None


class LLMRuntime:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.adapters: dict[str, str] = {}      # adapter name -> path

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = pick_device(self.cfg.device)
        dtype = getattr(torch, self.cfg.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model, revision=self.cfg.revision)
        model = AutoModelForCausalLM.from_pretrained(self.cfg.base_model, revision=self.cfg.revision, dtype=dtype)
        self.model = model.to(self.device).eval()

    def load(self) -> None:
        with self._lock:
            self._load()

    def ensure_adapter(self, name: str, path: str | Path) -> None:
        with self._lock:
            self._load()
            if name in self.adapters:
                return
            from peft import PeftModel
            if not self.adapters:
                self.model = PeftModel.from_pretrained(self.model, str(path), adapter_name=name).eval()
            else:
                self.model.load_adapter(str(path), adapter_name=name)
            self.adapters[name] = str(path)

    def _adapter_ctx(self, adapter: str | None):
        """Activate `adapter` (None = the unadapted base). Caller holds the lock."""
        import contextlib
        if adapter is not None and adapter not in self.adapters:
            raise KeyError(f"adapter {adapter!r} is not loaded")
        if adapter is None:
            return self.model.disable_adapter() if self.adapters else contextlib.nullcontext()
        self.model.set_adapter(adapter)
        return contextlib.nullcontext()

    def score(self, msgs: list[dict[str, str]], candidates: list[str], adapter: str | None = None) -> tuple[list[float], float]:
        """Summed log-probability of each candidate answer (closed by the end-of-turn token) given the
        prompt, in one batched forward pass. Uses the training code's encoder so scores match the
        quantities DITTO optimises. Returns (log-probs, seconds)."""
        import torch

        from training.ditto import Encoder, answer_logps      # lazy: training imports this module
        with self._lock:
            self._load()
            enc = Encoder(self.tokenizer, self.device)
            prompt = enc.prompt_ids(msgs)
            ctx = self._adapter_ctx(adapter)
            items = [(prompt, enc.answer_ids(c)) for c in candidates]
            batch = items if len(prompt) * len(items) <= MAX_SCORE_BATCH_TOKENS else None
            t0 = time.perf_counter()
            with ctx, torch.inference_mode():
                if batch is not None:
                    lps = [float(x) for x in answer_logps(self.model, enc, batch).float().cpu()]
                else:      # long prompt: one option per forward pass to bound memory
                    lps = [float(answer_logps(self.model, enc, [it]).float().cpu()[0]) for it in items]
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
            return lps, time.perf_counter() - t0

    def generate(self, msgs: list[dict[str, str]], adapter: str | None = None) -> tuple[str, float]:
        """Greedy decoding with `adapter` active (None = the unadapted base). Returns (text, seconds)."""
        import torch
        with self._lock:
            self._load()
            prompt = chat_prompt(self.tokenizer, msgs)
            ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            ctx = self._adapter_ctx(adapter)
            t0 = time.perf_counter()
            with ctx, torch.inference_mode():
                out = self.model.generate(**ids, max_new_tokens=self.cfg.max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          pad_token_id=self.tokenizer.eos_token_id)
            dt = time.perf_counter() - t0
            text = self.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            return text.strip(), dt


_shared: dict[tuple, LLMRuntime] = {}
_shared_lock = threading.Lock()


def shared_runtime(cfg: LLMConfig) -> LLMRuntime:
    """One runtime per (model, revision, device, dtype) per process."""
    key = (cfg.base_model, cfg.revision, cfg.device, cfg.dtype, cfg.decoding)
    with _shared_lock:
        if key not in _shared:
            _shared[key] = LLMRuntime(cfg)
        return _shared[key]
