"""Simulate Middleton, Rawson & Verkuilen (2019) and fit the learning model to its item-level data.

Protocol (Method section): per participant, 192 personally unnameable items, 48 per condition
(retrieval practice vs repetition x spaced lag 24 vs massed lag 1), trained in two cycles of four
sessions, two sessions a week. Each item: once per session, 3 trials per session with 1 or 24
intervening trials. Repetition: 3 study trials (hear, repeat, feedback). Retrieval practice:
session 1 = 1 study + 2 attempts with feedback; sessions 2-4 = 3 attempts with feedback. Tests
without feedback 7 days (retention) and 28 days (follow-up) after the last session. The
within-training measure in the data file is accuracy on the first attempt of sessions 2, 3 and 4
("Day" 1-3), which is how we read `spacingbysessioninteraction.txt`.

Fitted parameters: shared c, alpha, s, w_study, w_fail_feedback (w_success fixed at 1 as the scale;
tau fixed at Pavlik & Anderson's value since it trades off against beta); per participant beta.
A per-participant learning scale was tried and is not identifiable from four participants, so
individual differences enter only through beta; P3's reversal (repetition > retrieval practice) is
therefore not captured and is reported as such. Objective: binomial negative log-likelihood of the 4 x 4 x (3 + 2) observed
accuracies. Simulation uses common random numbers so the objective is smooth enough for Nelder-Mead.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from patientmodel.learning import DAY, SECOND, ItemMemory, LearningModel, LearningParams

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "middleton2019_osf"
OUT = ROOT / "results" / "learning_fit.json"
SUBJECTS = ("P1", "P2", "P3", "P4")
CONDITIONS = tuple((tr, sp) for tr in ("retrievalpractice", "repetition") for sp in ("Spaced", "Massed"))
SESSION_DAYS = (0.0, 3.0, 7.0, 10.0)
RETENTION_DAY = SESSION_DAYS[-1] + 7.0
FOLLOWUP_DAY = SESSION_DAYS[-1] + 28.0
TRIAL_SECONDS = 12.0            # 8 s attempt + feedback and inter-trial interval (ASSUMPTION)


def load_observed() -> dict:
    """Observed successes/trials per (subject, condition, measure)."""
    def rows(name: str) -> list[dict]:
        raw = (DATA / name).read_text().replace("\r", "\n")
        return list(csv.DictReader([l for l in raw.splitlines() if l.strip()], delimiter="\t"))
    obs: dict[tuple, list[int]] = {}
    for name, measure in (("week.txt", "retention"), ("month.txt", "followup")):
        for r in rows(name):
            obs.setdefault((r["Subject"], (r["Training"], r["Spacing"]), measure), [0, 0])
            obs[(r["Subject"], (r["Training"], r["Spacing"]), measure)][0] += int(r["ResponseAccuracy"])
            obs[(r["Subject"], (r["Training"], r["Spacing"]), measure)][1] += 1
    for r in rows("spacingbysessioninteraction.txt"):
        key = (r["Subject"], ("retrievalpractice", r["Spacing"]), f"day{r['Day']}")
        obs.setdefault(key, [0, 0])
        obs[key][0] += int(r["ResponseAccuracy"]); obs[key][1] += 1
    return obs


@dataclass
class SubjectParams:
    beta: float
    learn_scale: float


def simulate_subject(params: LearningParams, sp: SubjectParams, n_items: int, seed: int) -> dict[tuple, float]:
    """Predicted accuracy per (condition, measure) for one simulated participant."""
    model = LearningModel(params, learn_scale=sp.learn_scale)
    rng = np.random.default_rng(seed)
    out: dict[tuple, float] = {}
    for cond in CONDITIONS:
        training, spacing = cond
        lag = 24 if spacing == "Spaced" else 1
        hits = {m: 0 for m in ("day1", "day2", "day3", "retention", "followup")}
        for _ in range(n_items):
            mem = ItemMemory(beta=sp.beta)
            for si, day in enumerate(SESSION_DAYS):
                start = rng.integers(0, 60)                      # position of the first trial in its block
                times = [day + (start + k * (lag + 1)) * TRIAL_SECONDS * SECOND for k in range(3)]
                for k, t in enumerate(times):
                    if training == "repetition" or (si == 0 and k == 0):
                        model.study(mem, t)
                    else:
                        ok = model.retrieval_practice(mem, t, rng)
                        if k == 0 and 1 <= si <= 3:
                            hits[f"day{si}"] += ok
            hits["retention"] += model.test(mem, RETENTION_DAY, rng)
            hits["followup"] += model.test(mem, FOLLOWUP_DAY, rng)
        for m, h in hits.items():
            if training == "repetition" and m.startswith("day"):
                continue
            out[(cond, m)] = h / n_items
    return out


def predict_all(params: LearningParams, subs: dict[str, SubjectParams], n_items: int = 48, seed: int = 0) -> dict[tuple, float]:
    pred: dict[tuple, float] = {}
    for j, s in enumerate(SUBJECTS):
        for (cond, m), v in simulate_subject(params, subs[s], n_items, seed + 1000 * j).items():
            pred[(s, cond, m)] = v
    return pred


def nll(pred: dict[tuple, float], obs: dict[tuple, list[int]]) -> float:
    total = 0.0
    for key, (k, n) in obs.items():
        p = min(max(pred.get(key, 0.5), 1e-3), 1 - 1e-3)
        total -= k * np.log(p) + (n - k) * np.log(1 - p)
    return float(total)


N_SHARED = 5


def unpack(x: np.ndarray) -> tuple[LearningParams, dict[str, SubjectParams]]:
    c, alpha, s, w_study, w_fail = (float(np.exp(v)) for v in x[:N_SHARED])
    params = LearningParams(c=c, alpha=alpha, s=s, w_success=1.0, w_fail_feedback=w_fail, w_study=w_study, trial_seconds=TRIAL_SECONDS)
    subs = {sub: SubjectParams(beta=float(x[N_SHARED + j]), learn_scale=1.0) for j, sub in enumerate(SUBJECTS)}
    return params, subs


def fit(n_items: int = 96, seed: int = 0, maxiter: int = 1500) -> dict:
    from scipy.optimize import minimize
    obs = load_observed()
    base = LearningParams()
    beta0 = LearningModel.beta_for_probability(0.03, base)
    x0 = np.array([np.log(base.c), np.log(base.alpha), np.log(base.s), np.log(0.6), np.log(0.6)] + [beta0] * len(SUBJECTS))
    f = lambda x: nll(predict_all(*unpack(x), n_items=n_items, seed=seed), obs)
    res = minimize(f, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 0.5, "adaptive": True})
    params, subs = unpack(res.x)
    pred = predict_all(params, subs, n_items=400, seed=seed + 7)
    report = {
        "nll": float(res.fun), "iterations": int(res.nit), "converged": bool(res.success),
        "params": {k: float(v) for k, v in params.__dict__.items()},
        "subjects": {s: {"beta": sp.beta, "learn_scale": sp.learn_scale} for s, sp in subs.items()},
        "fit_table": [{"subject": s, "training": c[0], "spacing": c[1], "measure": m, "observed": round(k / n, 3), "n": n,
                       "predicted": round(pred[(s, c, m)], 3)} for (s, c, m), (k, n) in sorted(obs.items())],
    }
    return report


def pooled(table: list[dict], by: tuple[str, ...]) -> list[dict]:
    g: dict[tuple, list] = {}
    for r in table:
        g.setdefault(tuple(r[k] for k in by), []).append(r)
    return [{**dict(zip(by, k)), "observed": round(sum(r["observed"] * r["n"] for r in v) / sum(r["n"] for r in v), 3),
             "predicted": round(sum(r["predicted"] * r["n"] for r in v) / sum(r["n"] for r in v), 3), "n": sum(r["n"] for r in v)}
            for k, v in sorted(g.items())]


def write_markdown(rep: dict, path: Path) -> None:
    p = rep["params"]; tab = rep["fit_table"]
    def cell(tr, sp, m):
        rows = [r for r in tab if r["training"] == tr and r["spacing"] == sp and r["measure"] == m]
        if not rows:
            return "-"
        n = sum(r["n"] for r in rows)
        return f"{sum(r['observed'] * r['n'] for r in rows) / n:.2f} / {sum(r['predicted'] * r['n'] for r in rows) / n:.2f}"
    L = ["<!-- Generated by: uv run python -m patientmodel.learning_fit. Do not edit by hand. -->",
         "Learning model fitted to Middleton, Rawson & Verkuilen (2019): 4 participants, 192 items each, retrieval practice vs "
         "repetition x spaced (lag 24) vs massed (lag 1), four sessions, tests at 7 and 28 days. Simulation of the published "
         "protocol; binomial likelihood on 56 observed cells.\n",
         f"Shared parameters: c = {p['c']:.3f}, alpha = {p['alpha']:.3f}, s = {p['s']:.3f}, tau = {p['tau']:.2f} (fixed); trace weights: "
         f"successful retrieval 1.00 (scale), failed retrieval + feedback {p['w_fail_feedback']:.2f}, study {p['w_study']:.2f}. "
         "Per participant baseline beta: " + ", ".join(f"{s} {d['beta']:.2f}" for s, d in rep["subjects"].items()) +
         f". Negative log-likelihood {rep['nll']:.0f}.\n",
         "**Pooled over participants (observed / predicted)**\n",
         "| training | spacing | session 2 | session 3 | session 4 | retention 7 d | follow-up 28 d |", "|---|---|---|---|---|---|---|"]
    for tr in ("retrievalpractice", "repetition"):
        for sp in ("Spaced", "Massed"):
            L.append(f"| {tr} | {sp} | " + " | ".join(cell(tr, sp, m) for m in ("day1", "day2", "day3", "retention", "followup")) + " |")
    L += ["\n**Retention at 7 days by participant (observed / predicted)**\n", "| participant | repetition | retrieval practice |", "|---|---|---|"]
    for s in SUBJECTS:
        def c2(tr):
            rows = [r for r in tab if r["subject"] == s and r["training"] == tr and r["measure"] == "retention"]
            n = sum(r["n"] for r in rows)
            return f"{sum(r['observed'] * r['n'] for r in rows) / n:.2f} / {sum(r['predicted'] * r['n'] for r in rows) / n:.2f}"
        L.append(f"| {s} | {c2('repetition')} | {c2('retrievalpractice')} |")
    worst = max(abs(r["observed"] - r["predicted"]) for r in pooled(tab, ("training", "spacing", "measure")))
    L += [f"\nWhat the fit captures: the spacing effect at both delays, the retrieval-practice advantage at both delays, the rise "
          f"across sessions, and the forgetting between 7 and 28 days; the worst pooled cell is off by {worst:.2f}. What it does "
          "not: individual differences in the retrieval-practice effect. P3 shows the reverse pattern and P4 an unusually large "
          "advantage; a shared model with only a per-participant baseline cannot produce either, and a per-participant learning "
          "scale was not identifiable from four participants. Treat the trace weights as population averages.",
          "\nAssumptions: sessions on days 0, 3, 7, 10; 12 s per trial; the within-training measure is the first attempt of "
          "sessions 2-4; tau fixed at Pavlik & Anderson (2005)'s value; items start at a retrieval probability of 0.03 before "
          "training (they were selected as unnameable); the power law runs in seconds with a one-second floor."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    import time
    t0 = time.time()
    rep = fit()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))
    write_markdown(rep, OUT.with_suffix(".md"))
    print(f"fit in {time.time() - t0:.0f}s, nll {rep['nll']:.1f}, {rep['iterations']} iterations, converged={rep['converged']}")
    print("shared:", {k: round(v, 3) for k, v in rep["params"].items() if k in ("c", "alpha", "s", "w_study", "w_fail_feedback")})
    print("subjects:", {s: {k: round(v, 3) for k, v in d.items()} for s, d in rep["subjects"].items()})
    print("\npooled by training x spacing x measure (observed / predicted):")
    for r in pooled(rep["fit_table"], ("training", "spacing", "measure")):
        print(f"  {r['training']:18s} {r['spacing']:7s} {r['measure']:10s} {r['observed']:.3f} / {r['predicted']:.3f}  (n={r['n']})")
    print("\nby subject x training, retention:")
    for r in pooled([x for x in rep["fit_table"] if x["measure"] == "retention"], ("subject", "training")):
        print(f"  {r['subject']} {r['training']:18s} {r['observed']:.3f} / {r['predicted']:.3f}")
