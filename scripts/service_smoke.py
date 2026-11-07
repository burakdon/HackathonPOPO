"""End-to-end smoke test: a simulated patient plays a full session through the live HTTP service.

Exercises exactly the loop the frontend will run: POST /act -> show trial -> POST /update.
The patient is a held-out (test split) member of a preset cohort from sim/.

Usage:  uv run python -m scripts.service_smoke --url http://localhost:8000 --cohort mild_anomic
"""
from __future__ import annotations

import argparse

import httpx

from core.config import load_clinical
from eval.metrics import rolling_accuracy
from sim.config import load_sim
from sim.items import load_items
from sim.population import make_patients


def main() -> None:
    cfg, clinical = load_sim(), load_clinical()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--cohort", default="mild_anomic", choices=list(cfg.presets))
    ap.add_argument("--patient", type=int, default=0, help="index of the held-out patient")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bank = load_items(cfg.items)
    spec = make_patients(cfg, args.cohort, "test", 1, root_seed=args.seed, start=args.patient)[0]
    patient, selector = spec.build(bank, cfg, clinical)
    session_id = f"smoke-{args.cohort}-{args.patient}-{args.seed}"
    start = patient.mean_retrieval()
    history: list[dict] = []
    tier, minutes = cfg.session.start_tier, 0.0
    with httpx.Client(base_url=args.url, timeout=5.0) as c:
        agent = c.get("/health").raise_for_status().json()["agent"]
        for k in range(cfg.session.n_trials):
            state = {"session_id": session_id, "last_trials": history[-10:], "current_tier": tier,
                     "trial_count": k, "session_minutes": round(minutes, 2)}
            a = c.post("/act", json=state).raise_for_status().json()
            item = selector.next(a["tier"])
            r = patient.respond(item, a["cue"])
            c.post("/update", json={"state": state, "action": a, "outcome": {
                "correct": r.correct, "error_type": r.error_type,
                "response_time": r.response_time}}).raise_for_status()
            history.append({"word": bank.words[item], "tier": a["tier"], "cue": a["cue"],
                            "outcome": r.correct, "error_type": r.error_type, "response_time": r.response_time})
            tier, minutes = a["tier"], minutes + (r.response_time + cfg.response_time.overhead_s) / 60
            recent = history[-clinical.window:]
            print(f"{k:2d}  tier {a['tier']}  cue {a['cue']:<14} {bank.words[item]:<12} "
                  f"{'correct' if r.correct else r.error_type:<12} p_true {r.p_correct:.2f}  "
                  f"roll{clinical.window} {sum(t['outcome'] for t in recent) / len(recent):.2f}")

    acc = [t["outcome"] for t in history]
    roll = rolling_accuracy(acc, clinical.window)
    lo, hi = clinical.band
    print(f"\nagent {agent}, session {session_id}: accuracy {sum(acc) / len(acc):.2f}, "
          f"time in band {((roll >= lo) & (roll <= hi)).mean():.2f}, "
          f"tiers used {sorted(set(t['tier'] for t in history))}, final tier {tier}, "
          f"learning gain {100 * (patient.mean_retrieval() - start):+.2f} pts (sim-only)")


if __name__ == "__main__":
    main()
