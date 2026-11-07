"""Rehearse the live DITTO demo end to end over HTTP, with a simulated clinician.

Three phases, all through the service exactly as the dashboard would drive it:
  1. BEFORE: the base PPO policy runs sessions on evaluation patients; we count how often it already
     makes the clinician's choice where her preference applies ("after a sound error, first-sound cue").
  2. TEACH: sessions on OTHER patients; the clinician overrides the agent whenever her preference
     applies and it disagrees, until she has given --overrides overrides; every other choice she
     accepts (logged as an anchor). Supervision stops at the last override, then POST /ditto/train.
  3. AFTER: the same evaluation patients and the same random streams as phase 1, no overrides.

Usage:
  AGENT=ppo uv run uvicorn service.app:app --port 8000      # in one terminal
  uv run python -m scripts.ditto_live_demo --url http://localhost:8000 --overrides 3
  uv run python -m scripts.ditto_live_demo --inprocess --overrides 3      # no server needed
"""
from __future__ import annotations

import argparse
import json

from core.config import load_clinical
from sim.config import load_sim
from sim.items import load_items
from sim.population import make_patients

RULE = {"after_error": "sound", "cue": "first_sound"}


def play(client, spec, bank, cfg, clinical, session_id: str, clinician: dict | None, n_trials: int) -> dict:
    """One session. clinician = {"budget": remaining overrides} or None (no overrides).
    Returns counts of states where the rule applies and how often the shown/agent action follows it."""
    patient, selector = spec.build(bank, cfg, clinical)
    history, tier, minutes = [], cfg.session.start_tier, 0.0
    applies = agent_follows = overrides = 0
    for k in range(n_trials):
        if clinician is not None and clinician["budget"] <= 0:
            # Stop supervising at the last override. Choices shown after this point were not reviewed,
            # so they must not be logged as accepted (they would contradict her preference).
            break
        state = {"session_id": session_id, "last_trials": history[-10:], "current_tier": tier,
                 "trial_count": k, "session_minutes": round(minutes, 2)}
        a = client.post("/act", json=state).json()
        shown = a
        rule_applies = bool(history) and history[-1]["error_type"] == RULE["after_error"]
        if rule_applies:
            applies += 1
            agent_follows += a["cue"] == RULE["cue"]
            if clinician and clinician["budget"] > 0 and a["cue"] != RULE["cue"]:
                shown = {"tier": a["tier"], "cue": RULE["cue"], "rationale": "clinician override"}
                client.post("/override", json={"state": state, "agent_action": a, "clinician_action": shown,
                                               "note": "after a sound error I give the first sound"})
                clinician["budget"] -= 1
                overrides += 1
        item = selector.next(shown["tier"])
        r = patient.respond(item, shown["cue"])
        client.post("/update", json={"state": state, "action": shown, "reviewed": clinician is not None, "outcome": {
            "correct": r.correct, "error_type": r.error_type, "response_time": r.response_time}})
        history.append({"word": bank.words[item], "tier": shown["tier"], "cue": shown["cue"], "outcome": r.correct,
                        "error_type": r.error_type, "response_time": r.response_time})
        tier, minutes = shown["tier"], minutes + (r.response_time + cfg.response_time.overhead_s) / 60
        if r.quit:
            break
    return {"applies": applies, "agent_follows": agent_follows, "overrides": overrides}


def phase(client, specs, bank, cfg, clinical, tag, clinician=None, n_trials=40) -> dict:
    tot = {"applies": 0, "agent_follows": 0, "overrides": 0}
    for j, spec in enumerate(specs):
        for key, v in play(client, spec, bank, cfg, clinical, f"{tag}-{j}", clinician, n_trials).items():
            tot[key] += v
        if clinician is not None and clinician["budget"] <= 0:
            break
    tot["follow_rate"] = tot["agent_follows"] / tot["applies"] if tot["applies"] else float("nan")
    return tot


def run_demo(client, n_overrides: int = 3, n_eval: int = 10, variant: str | None = None, cohort: str = "moderate_broca") -> dict:
    cfg, clinical = load_sim(), load_clinical()
    bank = load_items(cfg.items)
    eval_specs = make_patients(cfg, cohort, "test", n_eval, root_seed=0, start=0)
    teach_specs = make_patients(cfg, cohort, "test", 20, root_seed=0, start=100)
    client.post("/ditto/reset")
    before = phase(client, eval_specs, bank, cfg, clinical, "before")
    taught = phase(client, teach_specs, bank, cfg, clinical, "teach", clinician={"budget": n_overrides})
    train = client.post("/ditto/train", json={"variant": variant}).json()
    after = phase(client, eval_specs, bank, cfg, clinical, "after")
    return {"before": before, "taught": taught, "train": {k: v for k, v in train.items() if k != "rounds"},
            "after": after}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--inprocess", action="store_true", help="run the service in-process (fresh log files)")
    ap.add_argument("--overrides", type=int, default=3)
    ap.add_argument("--eval-patients", type=int, default=10)
    ap.add_argument("--variant", default=None, choices=[None, "paper", "legacy"])
    args = ap.parse_args()
    if args.inprocess:
        import tempfile
        from fastapi.testclient import TestClient
        from service.app import create_app
        tmp = tempfile.mkdtemp()
        client = TestClient(create_app(agent="ppo", overrides_path=f"{tmp}/overrides.jsonl",
                                       accepted_path=f"{tmp}/accepted.jsonl"))
    else:
        import httpx
        client = httpx.Client(base_url=args.url, timeout=60.0)
        if client.get("/health").json()["agent"] not in ("ppo", "ppo+ditto"):
            raise SystemExit("start the service with AGENT=ppo for the live DITTO demo")
    res = run_demo(client, args.overrides, args.eval_patients, args.variant)
    b, a, t = res["before"], res["after"], res["train"]
    print(json.dumps(t.get("per_override", t), indent=1))
    print(f"\nClinician gave {res['taught']['overrides']} overrides "
          f"(and accepted the agent's other choices: {t.get('n_anchors')} anchors).")
    print(f"DITTO ({t.get('variant')}, {t.get('seconds')} s): agreement on the override states "
          f"{t.get('agreement_before', float('nan')):.2f} -> {t.get('agreement_after', float('nan')):.2f}")
    print(f"Held-out sessions, states after a sound error: agent gives the first-sound cue "
          f"{b['agent_follows']}/{b['applies']} ({b['follow_rate']:.0%}) before -> "
          f"{a['agent_follows']}/{a['applies']} ({a['follow_rate']:.0%}) after")


if __name__ == "__main__":
    main()
