"""Demo check: how do the live agents react to a healthy 'judge' over HTTP?

Three scripted responders play a session through POST /next_trial and POST /update (review probes
included), against one or more running agent services:
  always_correct   answers every trial correctly
  mostly_correct   ~90% correct (seeded)
  error_streak     correct, then 3 deliberate errors in a row (sound errors), then correct again
For each, the tier/cue trace is printed with a few adaptation statistics.

Usage:  uv run python -m scripts.judge_demo --url http://localhost:8771 --name ppo_sim [--url ... --name ...]
"""
from __future__ import annotations

import argparse

import httpx
import numpy as np

CUE_ABBR = {"none": "-", "first_sound": "S", "first_letter": "L", "sentence_frame": "F", "semantic": "M"}
N_TRIALS = 30
STREAK_AT = 12          # error_streak: trials 12, 13, 14 are errors (counting all trials)


def responder(name: str, k: int, rng: np.random.Generator) -> bool:
    if name == "always_correct":
        return True
    if name == "mostly_correct":
        return bool(rng.random() < 0.9)
    return not (STREAK_AT <= k < STREAK_AT + 3)


def play(url: str, who: str, n: int = N_TRIALS, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    hist, tier, out = [], 1, []
    with httpx.Client(base_url=url, timeout=10.0) as c:
        sid = f"judge-{who}-{seed}"
        for k in range(n):
            st = {"session_id": sid, "last_trials": hist[-10:], "current_tier": tier, "trial_count": k,
                  "session_minutes": round(k * 0.15, 2)}
            nt = c.post("/next_trial", json=st).raise_for_status().json()
            ok = responder(who, k, rng)
            err = "correct" if ok else "sound"
            c.post("/update", json={"state": st, "action": {"tier": nt["tier"], "cue": nt["cue"]},
                                    "outcome": {"correct": ok, "error_type": err, "response_time": 2.0 if ok else 4.0},
                                    "probe": nt["probe"]}).raise_for_status()
            hist.append({"word": nt["word"], "tier": nt["tier"], "cue": nt["cue"], "outcome": ok, "error_type": err,
                         "response_time": 2.0 if ok else 4.0, "probe": nt["probe"]})
            if not nt["probe"]:
                tier = nt["tier"]
            out.append({**nt, "correct": ok})
    return out


def summary(trace: list[dict]) -> str:
    agent = [t for t in trace if not t["probe"]]
    tiers = [t["tier"] for t in agent]
    changes = sum(a != b for a, b in zip(tiers, tiers[1:]))
    first3 = next((i for i, t in enumerate(trace) if not t["probe"] and t["tier"] == 3), None)
    cues_after_err = sum(1 for a, b in zip(trace, trace[1:]) if not a["correct"] and not b["probe"] and b["cue"] != "none")
    return (f"max tier {max(tiers)}, tier changes {changes}, first tier-3 trial "
            f"{'-' if first3 is None else first3}, mean tier {np.mean(tiers):.2f}, cues after errors {cues_after_err}")


def line(trace: list[dict]) -> str:
    """One token per trial: tier digit + cue letter (- = no cue), lower-case p = review probe, x = error."""
    return " ".join(("p" if t["probe"] else "") + f"{t['tier']}{CUE_ABBR[t['cue']]}" + ("" if t["correct"] else "x")
                    for t in trace)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", required=True)
    ap.add_argument("--name", action="append", required=True)
    args = ap.parse_args()
    print("token = tier + cue (- none, S first sound, L first letter, F sentence frame, M semantic); "
          "p = review probe; x = error\n")
    for who in ("always_correct", "mostly_correct", "error_streak"):
        print(f"== {who}")
        for url, name in zip(args.url, args.name):
            tr = play(url, who)
            print(f"  {name:18} {summary(tr)}\n  {'':18} {line(tr)}")
        print()


if __name__ == "__main__":
    main()
