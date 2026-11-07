"""How many overrides does the therapist need, and does the DITTO recipe matter?

For each simulated clinician rule ("after a <error> error, give the <cue> cue"), each number of
overrides n and each seed, three ways of learning from the same n overrides are compared on
held-out decision states from DISJOINT validation patients:

  - SFT: behaviour cloning on the demonstrations (the paper's main baseline);
  - DITTO-style (legacy): demonstrations > all sampled actions, uniform, reference = original policy;
  - DITTO (paper): SFT initialisation, reference = SFT policy, 70/20/10 online/replay/intermodel.

All three use the same accepted-choice anchors and the same number of gradient steps. Metrics:
  adoption  share of held-out states where the rule applies whose top action uses the rule's cue
  leakage   share of held-out states after OTHER errors whose top action changed (lower is better)
  kept      share of all remaining held-out states whose top action is unchanged (higher is better)

Writes results/ditto_sweep.csv, results/ditto_sweep.md and results/ditto_sweep.png.
Usage:  uv run python -m train.ditto_sweep [--quick]
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np
from sb3_contrib import MaskablePPO

from agents.ditto import Demo, action_probs, behaviour_cloning, ditto
from core.config import load_yaml, resolve
from core.masks import ACTION_CUES, ACTION_TIERS, action_index
from envs.aphasia_env import AphasiaEnv
from train.ditto_ppo import categorise, collect_states, summarise

RULES = [{"after_error": "sound", "cue": "first_sound"},
         {"after_error": "meaning", "cue": "semantic"},
         {"after_error": "no_response", "cue": "sentence_frame"}]
METHODS = ("SFT", "DITTO-style (legacy)", "DITTO (paper)")


def candidates(ref, pool_cats: dict, rule: dict) -> list[Demo]:
    """Override states: the rule applies and the policy's top action does not use the rule's cue.
    The simulated clinician keeps the policy's tier and changes the cue."""
    out = []
    for s in pool_cats["matching"]:
        a_pol = int(np.argmax(action_probs(ref, s["obs"][None], s["mask"][None])[0]))
        a_cli = action_index(int(ACTION_TIERS[a_pol]), rule["cue"])
        if ACTION_CUES[a_pol] != rule["cue"] and s["mask"][a_cli]:
            out.append(Demo(s["obs"], s["mask"], a_cli))
    return out


def run(method: str, ref, demos, anchors, cfg) -> MaskablePPO:
    common = dict(anchors=anchors, anchor_weight=cfg["anchor_weight"])
    steps = cfg["rounds"] * cfg["gradient_steps_per_round"]
    if method == "SFT":
        return behaviour_cloning(ref, demos, steps, cfg["learning_rate"], cfg["seed_"], **common)
    if method == "DITTO-style (legacy)":
        return ditto(ref, demos, cfg["rounds"], cfg["samples_per_state"], cfg["beta"], cfg["learning_rate"],
                     cfg["gradient_steps_per_round"], cfg["seed_"], **common)[0]
    p = cfg["paper"]
    return ditto(ref, demos, cfg["rounds"], cfg["samples_per_state"], cfg["beta"], cfg["learning_rate"],
                 cfg["gradient_steps_per_round"], cfg["seed_"], sft_steps=p["sft_steps"], mixture=p["mixture"],
                 batch_size=p["batch_size"], **common)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 rule, n in {3, 10}, 1 seed (smoke test)")
    args = ap.parse_args()
    cfg = load_yaml("configs/ditto.yaml")
    ns, seeds, rules = cfg["sweep"]["n_overrides"], cfg["sweep"]["seeds"], RULES
    if args.quick:
        ns, seeds, rules = [3, 10], [0], RULES[:1]
    ref = MaskablePPO.load(str(resolve(cfg["base_model"])), device="cpu")
    env = AphasiaEnv(cohort="population", split="val")
    pool = collect_states(ref, env, range(*cfg["override_patients"]))
    held = collect_states(ref, env, range(*cfg["eval_patients"]))
    rows, t0 = [], time.time()
    for rule in rules:
        pool_cats, held_cats = categorise(pool, rule), categorise(held, rule)
        anchors = [(s["obs"], s["mask"]) for s in pool_cats["other_errors"] + pool_cats["unrelated"]]
        cand = candidates(ref, pool_cats, rule)
        base = summarise(ref, ref, held_cats["matching"], rule["cue"])["argmax_rule_cue"]
        for n in ns:
            for seed in seeds:
                pick = np.random.default_rng([seed, n]).choice(len(cand), size=min(n, len(cand)), replace=False)
                demos = [cand[i] for i in pick]
                for method in METHODS:
                    tuned = run(method, ref, demos, anchors, {**cfg, "seed_": seed})
                    m = summarise(tuned, ref, held_cats["matching"], rule["cue"])
                    o = summarise(tuned, ref, held_cats["other_errors"], rule["cue"])
                    u = summarise(tuned, ref, held_cats["unrelated"], rule["cue"])
                    rows.append({"rule": f"{rule['after_error']}->{rule['cue']}", "n_overrides": len(demos),
                                 "seed": seed, "method": method, "adoption_before": base,
                                 "adoption": m["argmax_rule_cue"], "leakage": 1 - o["argmax_same_as_original"],
                                 "kept": u["argmax_same_as_original"],
                                 "n_matching": m["n"], "n_other": o["n"], "n_unrelated": u["n"]})
            print(f"{rule['after_error']}->{rule['cue']} n={n} done ({time.time() - t0:.0f}s)", flush=True)

    out = "results/ditto_sweep_quick" if args.quick else "results/ditto_sweep"
    with open(resolve(out + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    write_report(rows, ns, out, cfg)
    plot(rows, ns, out)


def _agg(rows, method, n, key, rule=None):
    v = [r[key] for r in rows if r["method"] == method and r["n_overrides"] == n and (rule is None or r["rule"] == rule)]
    return float(np.mean(v)), float(np.std(v)), len(v)


def write_report(rows, ns, out, cfg) -> None:
    rules = sorted({r["rule"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    p = cfg["paper"]
    lines = ["<!-- Generated by: uv run python -m train.ditto_sweep. Do not edit by hand. -->",
             f"Learning a clinician's cue preference from n overrides. {len(rules)} simulated clinician rules x "
             f"{len(seeds)} seeds; held-out states from disjoint validation patients; mean ± sd over rules x seeds. "
             f"Same anchors (weight {cfg['anchor_weight']}) and gradient budget "
             f"({cfg['rounds']} x {cfg['gradient_steps_per_round']} steps) for every method. "
             f"Paper variant: {p['sft_steps']} SFT steps, mixture {p['mixture']}.", "",
             "**Adoption** (top action uses the clinician's cue where the rule applies; higher is better)", "",
             "| n overrides | " + " | ".join(METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
    for key, title in (("adoption", None), ("leakage", "**Leakage** (top action changed after OTHER errors; lower is better)"),
                       ("kept", "**Kept** (top action unchanged in all other states; higher is better)")):
        if title:
            lines += ["", title, "", "| n overrides | " + " | ".join(METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
        for n in ns:
            cells = []
            for m in METHODS:
                mu, sd, k = _agg(rows, m, n, key)
                cells.append(f"{mu:.2f} ± {sd:.2f}" if k else "")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "Adoption before tuning (original policy), per rule: " + ", ".join(
        f"{r} {np.mean([x['adoption_before'] for x in rows if x['rule'] == r]):.2f}" for r in rules) + ".", "",
        "Per rule, adoption at the largest n:", "", "| rule | " + " | ".join(METHODS) + " |",
        "|---" * (len(METHODS) + 1) + "|"]
    for r in rules:
        lines.append(f"| {r} | " + " | ".join(f"{_agg(rows, m, ns[-1], 'adoption', r)[0]:.2f}" for m in METHODS) + " |")
    resolve(out + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[1:]))


def plot(rows, ns, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in ((axes[0], "adoption", "Adopts the clinician's cue where it applies (higher is better)"),
                           (axes[1], "leakage", "Changes behaviour after other errors (lower is better)")):
        for m, marker in zip(METHODS, "osD"):
            mu = [_agg(rows, m, n, key)[0] for n in ns]
            sd = [_agg(rows, m, n, key)[1] for n in ns]
            ax.errorbar(ns, mu, yerr=sd, marker=marker, capsize=3, label=m)
        ax.set_xlabel("number of clinician overrides")
        ax.set_xticks(ns)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(resolve(out + ".png"), dpi=150)


if __name__ == "__main__":
    main()
