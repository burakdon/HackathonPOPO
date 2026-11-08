"""Golden test for ports of the delivered DITTO model: 5 fixed situations and the exact decision the
reference code returns for each (docs/ditto_golden_test.json, described in docs/DITTO_HANDOFF.md).

    uv run --extra llm python -m scripts.ditto_golden_check            # compare with the stored answers
    uv run --extra llm python -m scripts.ditto_golden_check --write    # (re)record them

Each case is a DecisionContext and the options the server offers. The model version, decoding and
prompt format come from configs/practice.yaml (ditto:sft200-g2-s0, option scoring, prompt v2).
"""
from __future__ import annotations

import argparse
import json

from agents.llm_policy import BanditDecisionPolicy
from agents.registry import ModelRegistry
from core.config import load_beta_bandit, load_clinical, resolve
from core.practice import DecisionContext, format_action, load_practice, messages, parse_action

GOLDEN = "docs/ditto_golden_test.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="record the current outputs as the expected ones")
    args = ap.parse_args()
    doc = json.loads(resolve(GOLDEN).read_text())
    pcfg = load_practice()
    reg = ModelRegistry(pcfg)
    version = doc["model_version"]
    pol = reg.policy(version)
    fallback = BanditDecisionPolicy(load_beta_bandit(), load_clinical(), seed=0)
    ok = True
    for case in doc["cases"]:
        ctx = DecisionContext(**case["context"])
        opts = [parse_action(o) for o in case["options"]]
        d = pol.decide(ctx, opts, fallback)
        keys = [format_action(o) for o in opts]
        lps, _ = pol.runtime.score(messages(ctx, opts, pcfg.history_in_prompt, fmt=pcfg.llm.prompt_format),
                                   keys, pol.adapter)
        got = {"action": d.action.key(), "source": d.source,
               "logprobs": {k: round(v, 2) for k, v in zip(keys, lps)}}
        if args.write:
            case["expected"] = got
        else:
            same = got["action"] == case["expected"]["action"] and got["source"] == version
            ok &= same
            print(f"{case['id']}: {'OK  ' if same else 'DIFF'} got {got['action']} expected {case['expected']['action']}")
        print(f"  {case['id']}: {got['action']}  ({d.latency_ms:.0f} ms)  logprobs {got['logprobs']}")
    if args.write:
        resolve(GOLDEN).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {GOLDEN}")
    elif not ok:
        raise SystemExit("golden test failed: decisions differ")
    else:
        print("golden test passed: all 5 decisions identical")


if __name__ == "__main__":
    main()
