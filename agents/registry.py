"""Model versions available to the practice flow, and the policy object for each.

Versions:
  bandit                    Beta-Bernoulli Thompson bandit baseline (no language model)
  base                      unadapted language model with the normal task instruction
  prompted                  the same model with the 20 training demonstrations in its prompt
  sft:<run_id>              the run's supervised-only adapter (isolates the effect of DITTO rounds)
  ditto:<run_id>            the run's final DITTO adapter
  ditto:<run_id>@round<k>   an intermediate DITTO round (evaluation)

Only runs whose manifest says `status: completed` can be selected. The language model is loaded
once per process (agents/llm_runtime.shared_runtime) and adapters are added to it on demand.
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import threading
from pathlib import Path

from agents.llm_policy import LLMDecisionPolicy
from agents.llm_runtime import llm_available, shared_runtime
from core.config import resolve
from core.practice import DecisionContext, PracticeConfig, parse_action, render_context

TRAIN_DEMOS = "data/demonstrations/demonstrator_a_v1.jsonl"


def few_shot_examples(path: str = TRAIN_DEMOS, history_in_prompt: int = 4) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in resolve(path).read_text().splitlines() if line.strip()]
    out = []
    for r in rows:
        if r["split"] != "train":      # never dev or test situations
            continue
        ctx = DecisionContext(**r["context"])
        out.append((render_context(ctx, [parse_action(o) for o in r["options"]], history_in_prompt), r["preferred"]))
    return out


class ModelRegistry:
    def __init__(self, pcfg: PracticeConfig) -> None:
        self.pcfg = pcfg
        self.runs_dir = resolve(pcfg.llm.runs_dir)
        self._policies: dict[str, LLMDecisionPolicy] = {}
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)   # one generation at a time

    # ---- discovery
    def runs(self) -> list[dict]:
        out = []
        if not self.runs_dir.exists():
            return out
        for m in sorted(self.runs_dir.glob("*/manifest.json")):
            try:
                out.append(json.loads(m.read_text()))
            except json.JSONDecodeError:
                continue
        return out

    def run(self, run_id: str) -> dict:
        p = self.runs_dir / run_id / "manifest.json"
        if not p.exists():
            raise KeyError(f"no training run {run_id!r}")
        return json.loads(p.read_text())

    def versions(self) -> list[dict]:
        ok, why = llm_available()
        llm = self.pcfg.llm
        vs = [{"id": "bandit", "label": "Bandit baseline (no language model)", "kind": "baseline",
               "available": True, "detail": "Beta-Bernoulli Thompson sampling over tier x hint"}]
        vs.append({"id": "base", "label": f"{llm.base_model}, unadapted", "kind": "llm", "available": ok,
                   "unavailable_reason": why})
        vs.append({"id": "prompted", "label": f"{llm.base_model} + training demonstrations in the prompt",
                   "kind": "llm", "available": ok, "unavailable_reason": why})
        for m in self.runs():
            done = m.get("status") == "completed"
            reason = why if not ok else (None if done else f"run status: {m.get('status')}")
            if m.get("smoke"):
                continue
            vs.append({"id": f"sft:{m['run_id']}", "label": f"SFT only ({m['run_id']})", "kind": "adapter",
                       "run_id": m["run_id"], "available": ok and done, "unavailable_reason": reason})
            vs.append({"id": f"ditto:{m['run_id']}", "label": f"DITTO adapter ({m['run_id']})", "kind": "adapter",
                       "run_id": m["run_id"], "available": ok and done, "unavailable_reason": reason,
                       "status": m.get("status")})
        return vs

    # ---- policies
    def adapter_path(self, version: str) -> tuple[str, Path, dict]:
        kind, rest = version.split(":", 1)
        run_id, _, rnd = rest.partition("@")
        m = self.run(run_id)
        if m.get("status") != "completed":
            raise ValueError(f"run {run_id} is not completed (status {m.get('status')})")
        if m.get("base_model") != self.pcfg.llm.base_model or m.get("base_revision") != self.pcfg.llm.revision:
            raise ValueError(f"run {run_id} was trained on {m.get('base_model')}@{m.get('base_revision')}, "
                             f"but the service serves {self.pcfg.llm.base_model}@{self.pcfg.llm.revision}")
        sub = "sft" if kind == "sft" else (rnd or m["final_adapter"])
        path = self.runs_dir / run_id / sub
        if not path.exists():
            raise FileNotFoundError(f"adapter files missing: {path}")
        name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{kind}-{run_id}-{sub}")    # PEFT forbids "." in adapter names
        return name, path, m

    def policy(self, version: str) -> LLMDecisionPolicy:
        with self._lock:
            if version in self._policies:
                return self._policies[version]
            ok, why = llm_available()
            if not ok:
                raise RuntimeError(why)
            llm = self.pcfg.llm
            runtime = shared_runtime(llm)
            detail = {"base_model": llm.base_model, "base_revision": llm.revision, "template": "enable_thinking=False"}
            few_shot, adapter = None, None
            if version == "base":
                pass
            elif version == "prompted":
                few_shot = few_shot_examples(TRAIN_DEMOS, self.pcfg.history_in_prompt)
                detail["few_shot_examples"] = len(few_shot)
            elif version.startswith(("sft:", "ditto:")):
                adapter, path, m = self.adapter_path(version)
                runtime.ensure_adapter(adapter, path)
                rel = path.relative_to(resolve(".")) if path.is_relative_to(resolve(".")) else path
                detail.update({"adapter": str(rel), "run_id": m["run_id"],
                               "trained_on": m["demonstrations"]["example_ids"]})
            else:
                raise KeyError(f"unknown model version {version!r}")
            pol = LLMDecisionPolicy(runtime, version, adapter, detail, self.pcfg.history_in_prompt,
                                    llm.timeout_s, few_shot, self._executor)
            self._policies[version] = pol
            return pol
