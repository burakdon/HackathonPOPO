"""End-to-end smoke test: a simulated patient plays a full session through the live HTTP service.

Exercises exactly the loop the frontend will run: POST /act -> show trial -> POST /update.
Step 1a uses the scaffold's SimPatient (docs/scaffold/rl_starter.py); Step 1b switches to sim/.

Usage:  uv run python -m scripts.service_smoke --url http://localhost:8000 --profile mild
"""
from __future__ import annotations

import argparse

import httpx

from docs.scaffold.rl_starter import make_profiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--profile", default="mild", choices=["mild", "moderate", "severe"])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    patient = make_profiles(seed=args.seed)[args.profile]
    session_id = f"smoke-{args.profile}-{args.seed}"
    history: list[dict] = []
    tier, minutes = 2, 0.0
    with httpx.Client(base_url=args.url, timeout=5.0) as c:
        for k in range(args.trials):
            state = {"session_id": session_id, "last_trials": history[-10:], "current_tier": tier,
                     "trial_count": k, "session_minutes": round(minutes, 2)}
            a = c.post("/act", json=state).raise_for_status().json()
            out = patient.respond(a["tier"], a["cue"])
            c.post("/update", json={"state": state, "action": a, "outcome": {
                "correct": out["correct"], "error_type": out["error_type"],
                "response_time": out["response_time"]}}).raise_for_status()
            history.append({"word": f"item_{k}", "tier": a["tier"], "cue": a["cue"],
                            "outcome": out["correct"], "error_type": out["error_type"],
                            "response_time": out["response_time"]})
            tier, minutes = a["tier"], minutes + out["response_time"] / 60 + 0.1
            roll = sum(t["outcome"] for t in history[-8:]) / len(history[-8:])
            print(f"{k:2d}  tier {a['tier']}  cue {a['cue']:<14} "
                  f"{'correct' if out['correct'] else out['error_type']:<12} roll8 {roll:.2f}")
        health = c.get("/health").json()

    acc = [t["outcome"] for t in history]
    roll8 = [sum(acc[i - 7:i + 1]) / 8 for i in range(7, len(acc))]
    in_band = sum(0.75 <= r <= 0.90 for r in roll8) / len(roll8)
    print(f"\nsession {session_id}: accuracy {sum(acc) / len(acc):.2f}, "
          f"time in band {in_band:.2f}, final tier {tier}, "
          f"latent ability gain {patient.ability - patient.start_ability:+.3f}")
    print("health:", health)


if __name__ == "__main__":
    main()
