"""DITTO-style demo: does the PPO policy adopt a clinician's preference from ~10 overrides, and only
where it applies?

1. Run the trained policy on validation patients and record every decision state.
2. Overrides: states where the simulated clinician's rule applies and the policy disagrees (or the
   real overrides in runs/overrides.jsonl with --overrides). The clinician keeps the policy's tier
   and changes the cue.
3. DITTO-style DPO (agents/ditto.py), frozen reference = the original policy.
4. Evaluate on states from DISJOINT validation patients: adoption of the rule in matching states,
   leakage into states after other error types (where cues are also allowed), agreement with the
   original policy in all other states, and full-simulation metrics on the three test cohorts.
Writes the tuned model and results/ditto_report.json + results/ditto_report.md.

Usage:  uv run python -m train.ditto_ppo [--overrides runs/overrides.jsonl]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sb3_contrib import MaskablePPO

from agents.ditto import Demo, action_probs, ditto
from agents.selection import clinical_gate
from core.config import load_clinical, load_yaml, resolve
from core.masks import ACTION_CUES, ACTION_TIERS, action_index, action_mask
from core.observation import ObservationBuilder
from core.schema import State
from envs.aphasia_env import AphasiaEnv, load_env_config

CUE_OF = np.array(ACTION_CUES)


def collect_states(model: MaskablePPO, env: AphasiaEnv, patients: range) -> list[dict]:
    """Every decision state the deterministic policy visits on these patients."""
    out = []
    for k in patients:
        obs, _ = env.reset(options={"patient_index": k})
        done = False
        while not done:
            mask = env.action_masks()
            state = env._state()
            last = state.last_trials[-1] if state.last_trials else None
            out.append({"obs": obs.copy(), "mask": mask.copy(), "patient": k,
                        "last_error": None if last is None else last.error_type})
            a, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, done, _, _ = env.step(int(a))
    return out


def categorise(states: list[dict], rule: dict) -> dict[str, list[dict]]:
    cue_i = lambda s: s["mask"][[action_index(t, rule["cue"]) for t in (1, 2, 3)]].any()
    cats = {"matching": [], "other_errors": [], "unrelated": []}
    for s in states:
        if s["last_error"] == rule["after_error"] and cue_i(s):
            cats["matching"].append(s)
        elif s["last_error"] in ("sound", "meaning", "no_response") and cue_i(s):
            cats["other_errors"].append(s)
        else:
            cats["unrelated"].append(s)
    return cats


def summarise(model, ref, states: list[dict], rule_cue: str) -> dict[str, float]:
    if not states:
        return {"n": 0}
    obs, masks = np.stack([s["obs"] for s in states]), np.stack([s["mask"] for s in states])
    p, p_ref = action_probs(model, obs, masks), action_probs(ref, obs, masks)
    is_cue = CUE_OF == rule_cue
    return {"n": len(states),
            "p_rule_cue": float(p[:, is_cue].sum(1).mean()),
            "argmax_rule_cue": float(np.mean(is_cue[p.argmax(1)])),
            "argmax_same_as_original": float(np.mean(p.argmax(1) == p_ref.argmax(1))),
            "total_variation_vs_original": float(0.5 * np.abs(p - p_ref).sum(1).mean())}


def demos_from_records(records: list[dict], clinical, env_cfg) -> tuple[list[Demo], list[dict]]:
    """Override records (as written by POST /override) -> demonstrations, plus the records skipped.

    The mask must be the one the policy acts under: the clinical mask AND the cue-only-after-error
    gate (agents/ppo.py, envs/aphasia_env.py). A clinician action outside it (e.g. a cue right after
    a correct answer) can never be produced by the policy, so it is skipped and reported rather than
    silently trained on as an impossible target."""
    ob = ObservationBuilder(clinical, env_cfg.history, env_cfg.nominal_trials, env_cfg.rt_cap_s)
    demos, skipped = [], []
    for rec in records:
        state = State(**rec["state"])
        mask = clinical_gate(state, action_mask(state.current_tier, clinical.approved_cues, clinical.max_tier_step),
                             env_cfg.cue_only_after_error)
        a = action_index(rec["clinician_action"]["tier"], rec["clinician_action"]["cue"])
        if mask[a]:
            demos.append(Demo(ob(state), mask, a))
        else:
            skipped.append(rec)
    return demos, skipped


def read_jsonl(path) -> list[dict]:
    p = resolve(path)
    return [json.loads(line) for line in open(p) if line.strip()] if p.exists() else []


def demos_from_overrides(path, clinical, env_cfg) -> list[Demo]:
    demos, skipped = demos_from_records(read_jsonl(path), clinical, env_cfg)
    if skipped:
        print(f"skipped {len(skipped)} overrides the policy cannot produce under the clinical mask")
    return demos


def sim_metrics(model_path: str, clinical, n: int = 50) -> dict[str, dict[str, float]]:
    from agents.ppo import PPOPolicy
    from eval.metrics import session_metrics
    from sim.config import load_sim
    from sim.items import load_items
    from sim.population import make_patients
    from sim.session import run_session
    cfg = load_sim()
    bank = load_items(cfg.items)
    pol = PPOPolicy(model_path, clinical, load_env_config())
    out = {}
    for cohort in cfg.presets:
        rows = []
        for spec in make_patients(cfg, cohort, "test", n, 0):
            patient, selector = spec.build(bank, cfg, clinical)
            rows.append(session_metrics(run_session(pol, patient, selector, cfg, clinical), clinical))
        out[cohort] = {k: float(np.nanmean([r[k] for r in rows])) for k in
                       ("time_in_band", "control_error", "learning_gain", "probe_recall", "frustration_events")}
    return out


def main() -> None:
    cfg = load_yaml("configs/ditto.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--overrides", default=None, help="use real overrides (JSONL from the service) instead")
    args = ap.parse_args()
    clinical, env_cfg = load_clinical(), load_env_config()
    rule = cfg["clinician_rule"]
    ref = MaskablePPO.load(str(resolve(cfg["base_model"])), device="cpu")
    env = AphasiaEnv(cohort="population", split="val")

    pool = collect_states(ref, env, range(*cfg["override_patients"]))
    if args.overrides:
        demos = demos_from_overrides(args.overrides, clinical, env_cfg)
        source = args.overrides
    else:
        rng = np.random.default_rng(cfg["seed"])
        cand = []
        for s in categorise(pool, rule)["matching"]:
            a_pol = int(np.argmax(action_probs(ref, s["obs"][None], s["mask"][None])[0]))
            a_cli = action_index(int(ACTION_TIERS[a_pol]), rule["cue"])
            if ACTION_CUES[a_pol] != rule["cue"] and s["mask"][a_cli]:
                cand.append(Demo(s["obs"], s["mask"], a_cli))
        pick = rng.choice(len(cand), size=min(cfg["n_overrides"], len(cand)), replace=False)
        demos = [cand[i] for i in pick]
        source = f"simulated clinician: after a '{rule['after_error']}' error use '{rule['cue']}'"
    print(f"{len(demos)} overrides ({source})")

    # Accepted choices: states from the same sessions where the clinician's rule does not apply, so a
    # clinician following it would have accepted the agent's choice.
    anchors = [(s["obs"], s["mask"]) for s in categorise(pool, rule)["other_errors"] + categorise(pool, rule)["unrelated"]]
    variants = [("DITTO-style (demonstrations only)", 0.0, None),
                ("DITTO-style + accepted-choice anchors", cfg["anchor_weight"], cfg["output_model"])]
    held_out = categorise(collect_states(ref, env, range(*cfg["eval_patients"])), rule)
    names = {"matching": f"after a {rule['after_error']} error (rule applies)",
             "other_errors": "after other errors (cues allowed, rule does not apply)",
             "unrelated": "all other states"}
    report = {"source": source, "n_overrides": len(demos), "n_anchor_states": len(anchors), "variants": {}}
    sim_before = sim_metrics(cfg["base_model"], clinical)
    lines = ["<!-- Generated by: uv run python -m train.ditto_ppo. Do not edit by hand. -->",
             f"DITTO-style tuning from {len(demos)} overrides ({source}); evaluated on held-out states from "
             f"disjoint validation patients and on the three test cohorts.", ""]
    for label, weight, save_path in variants:
        tuned, log = ditto(ref, demos, cfg["rounds"], cfg["samples_per_state"], cfg["beta"], cfg["learning_rate"],
                           cfg["gradient_steps_per_round"], cfg["seed"], anchors=anchors, anchor_weight=weight)
        print(label)
        for row in log:
            print("  round", row)
        tmp_path = save_path or "runs/ditto_plain.zip"
        resolve(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        tuned.save(str(resolve(tmp_path)), exclude=["tensorboard_log"])
        ev = {cat: {"before": summarise(ref, ref, st, rule["cue"]), "after": summarise(tuned, ref, st, rule["cue"])}
              for cat, st in held_out.items()}
        sim_after = sim_metrics(tmp_path, clinical)
        report["variants"][label] = {"anchor_weight": weight, "rounds": log, "evaluation": ev,
                                     "simulation": {"before": sim_before, "after": sim_after}}
        lines += [f"**{label}** (anchor weight {weight})", "",
                  "| held-out states | n | P(rule cue) before -> after | top action = rule cue before -> after | "
                  "top action unchanged | total variation vs original |", "|---|---|---|---|---|---|"]
        for cat, e in ev.items():
            b, a = e["before"], e["after"]
            if b["n"]:
                lines.append(f"| {names[cat]} | {b['n']} | {b['p_rule_cue']:.2f} -> {a['p_rule_cue']:.2f} | "
                             f"{b['argmax_rule_cue']:.2f} -> {a['argmax_rule_cue']:.2f} | "
                             f"{a['argmax_same_as_original']:.2f} | {a['total_variation_vs_original']:.3f} |")
        lines += ["", "| cohort | in band before -> after | control error | learning gain x100 | probe recall |",
                  "|---|---|---|---|---|"]
        for c in sim_before:
            b, a = sim_before[c], sim_after[c]
            lines.append(f"| {c} | {b['time_in_band']:.2f} -> {a['time_in_band']:.2f} | {b['control_error']:.3f} -> "
                         f"{a['control_error']:.3f} | {100 * b['learning_gain']:.2f} -> {100 * a['learning_gain']:.2f} | "
                         f"{b['probe_recall']:.2f} -> {a['probe_recall']:.2f} |")
        lines.append("")
    resolve("results/ditto_report.json").write_text(json.dumps(report, indent=2))
    resolve("results/ditto_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[1:]))


if __name__ == "__main__":
    main()
