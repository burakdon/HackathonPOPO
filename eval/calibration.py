"""Simulator calibration against published group-level statistics (data/calibration_targets.yaml).

  uv run python -m eval.calibration [--out results/calibration_report.md]

Computes the simulator measure for each target (see `measure` in the YAML), checks it against the
published acceptance range, and writes a report. tests/test_calibration.py asserts every target.
"""
from __future__ import annotations

import argparse

import numpy as np

from core.config import load_clinical, load_yaml, resolve
from sim.config import load_sim
from sim.items import load_items
from sim.patient import PatientParams, SimPatient, sigmoid
from sim.population import make_patients

N = 500


def _patients(cfg, cl, bank, cohort, n=N):
    return [s.build(bank, cfg, cl)[0] for s in make_patients(cfg, cohort, "test", n, 0)]


def _error_shares(pts) -> np.ndarray:
    """Expected share of (semantic, phonological, omission) among errors, averaged over patients.
    After a failure the simulator draws (sound, meaning, no_response) with probabilities proportional
    to (phon_impair, sem_impair, omission_bias); weight each patient by its error rate."""
    tot = np.zeros(3)
    for p in pts:
        err_rate = 1 - p.retrieval_probabilities().mean()
        w = np.array([p.params.sem_impair, p.params.phon_impair, p.params.omission_bias])
        tot += err_rate * w / w.sum()
    return tot / tot.sum()


def _normalised_accuracy(pts) -> float:
    """Accuracy with omissions removed from the denominator (as in Dell et al. 2007)."""
    vals = []
    for p in pts:
        acc = p.retrieval_probabilities().mean()
        w = np.array([p.params.phon_impair, p.params.sem_impair, p.params.omission_bias])
        omit = (1 - acc) * w[2] / w.sum()
        vals.append(acc / (1 - omit))
    return float(np.mean(vals))


def _phon_vs_sem(pts) -> float:
    s = _error_shares(pts)
    return float(s[1] / (s[0] + s[1]))


def measure(cfg=None) -> dict[str, float]:
    cfg, cl = cfg or load_sim(), load_clinical()
    bank = load_items(cfg.items)
    pop = _patients(cfg, cl, bank, "population")
    acc = np.array([p.retrieval_probabilities().mean() for p in pop])
    shares = _error_shares(pop)
    z = lambda v: (v - v.mean()) / v.std()
    m = {
        "population_accuracy_mean": float(acc.mean()),
        "population_accuracy_spread": float(np.percentile(acc, 95) - np.percentile(acc, 5)),
        "_population_accuracy_p5_p95": (float(np.percentile(acc, 5)), float(np.percentile(acc, 95))),
        "error_share_semantic": float(shares[0]),
        "error_share_phonological": float(shares[1]),
        "error_share_omission": float(shares[2]),
        "anomic_normalised_accuracy": _normalised_accuracy(_patients(cfg, cl, bank, "mild_anomic")),
        "broca_normalised_accuracy": _normalised_accuracy(_patients(cfg, cl, bank, "moderate_broca")),
        "anomic_phonological_vs_semantic": _phon_vs_sem(_patients(cfg, cl, bank, "mild_anomic")),
        "broca_phonological_vs_semantic": _phon_vs_sem(_patients(cfg, cl, bank, "moderate_broca")),
        # difficulty = beta_f * (-z(freq)) + beta_l * z(len): conditional slope per Zipf unit at fixed length
        "frequency_logodds_per_log10": float(cfg.items.beta_frequency / bank.frequency.std()),
        "phonemic_cue_logodds": float(np.mean([p.cue_gain["first_sound"] for p in pop])),
        "semantic_cue_logodds": float(np.mean([p.cue_gain["semantic"] for p in pop])),
        "ability_sd_logit": float(np.std([p.params.ability for p in pop])),
    }
    drops = []
    for c in ("mild_anomic", "moderate_broca", "severe"):
        curves = []
        for spec in make_patients(cfg, c, "test", 100, 0):
            p = SimPatient(PatientParams(**{**spec.params.to_dict(), "learn_rate": 0.0}), bank, cfg, cl,
                           np.random.default_rng(spec.index))
            items = bank.indices_in_tier(2)
            start = np.mean(sigmoid(p.ability - bank.difficulty - p.fatigue))
            for t in range(cfg.session.n_trials):
                p.respond(int(items[t % len(items)]), "none")
            curves.append(start - np.mean(sigmoid(p.ability - bank.difficulty - p.fatigue)))
        drops.append(float(np.mean(curves)))
    m["fatigue_direction"] = min(drops)       # smallest cohort drop must still be > 0
    return m


def check(m: dict, targets: dict) -> list[dict]:
    rows = []
    for key, t in targets["targets"].items():
        lo, hi = t["range"]
        v = m[key]
        ok = lo <= v <= hi
        if key == "population_accuracy_spread":
            p5, p95 = m["_population_accuracy_p5_p95"]
            ok = ok and 0.03 <= p5 and p95 <= 0.98
        if key == "fatigue_direction":
            ok = v > 0
        rows.append({"target": key, "value": v, "low": lo, "high": hi, "pass": ok,
                     "source": t["source"] if isinstance(t["source"], str) else "+".join(t["source"])})
    return rows


def report(rows: list[dict], title: str) -> str:
    lines = [f"**{title}**", "", "| target | simulator | published range | source | result |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['target']} | {r['value']:.3f} | [{r['low']}, {r['high']}] | {r['source']} | "
                     f"{'pass' if r['pass'] else '**OFF**'} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Calibration")
    args = ap.parse_args()
    rows = check(measure(), load_yaml("data/calibration_targets.yaml"))
    print(report(rows, args.title))
    print(f"\n{sum(r['pass'] for r in rows)}/{len(rows)} targets pass")


if __name__ == "__main__":
    main()
