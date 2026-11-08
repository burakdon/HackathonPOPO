"""Learn the clinician's response-type coding from the adjudicated rows and apply it to the rest.

The rule-based coder (scripts/code_psst_responses.py) was checked against a clinician on the noun
rows it was unsure about. Its semantic and phonological-nonword calls held up; its formal, mixed and
unrelated calls mostly did not. Here the adjudicated rows are taken as ground truth, a random forest
is trained on features of the target/response pair plus the machine's own code, and its accuracy is
estimated by leave-one-target-out cross-validation (rows for the same picture are correlated, so a
row-wise split would flatter it). The fitted model then codes every noun error row the clinician has
not seen and the machine was not confident about, writing `learned_category` and `learned_prob` into
the adjudication sheet. Precedence downstream: final_category (human) > learned_category > category.

Caveat kept in the output: the training rows are the machine's uncertain rows, so the model is
calibrated for rows like them, not for the high-confidence rows it is not applied to.

Run after the coder: uv run python -m scripts.learn_coder
"""
from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

import numpy as np

from patientmodel.lexicon import formally_related, phoneme_edit_distance, semantic_similarity
from scripts.code_psst_responses import ADJ, OUT, Coder, strip

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "psst_coder_learned.md"
MACHINE_CATS = ["correct_variant", "semantic", "formal", "mixed", "unrelated", "nonword_phon", "nonword_other",
                "fragment", "multiword", "no_response", "unclassified"]
CONF = {"low": 0, "medium": 1, "high": 2}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
SEED = 20261108


def features(coder: Coder, r: dict) -> list[float]:
    """Numeric description of one target/response pair; order is fixed, see FEATURE_NAMES."""
    prompt = r["prompt"].lower()
    tgts = [strip(p) for p in coder.cmu.get(prompt, [])]
    tgt = tgts[0] if tgts else ()
    raw = tuple(p for p in r["transcript"].split() if not p.startswith("<"))
    tr = coder.strip_fillers(raw)
    lt, lr = len(tgt), len(tr)
    ed = phoneme_edit_distance(tgt, tr) if tgt and tr else max(lt, lr)
    mx = max(lt, lr, 1)
    pre = 0
    while pre < min(lt, lr) and tgt[pre] == tr[pre]:
        pre += 1
    shared = sum((collections.Counter(tgt) & collections.Counter(tr)).values())
    same_pos = sum(a == b for a, b in zip(tgt, tr))
    tc = [p for p in tgt if p not in VOWELS]; rc = [p for p in tr if p not in VOWELS]
    shared_cons = sum((collections.Counter(tc) & collections.Counter(rc)).values())
    tv = [p for p in tgt if p in VOWELS]; rv = [p for p in tr if p in VOWELS]
    contains = any(tr[i:i + lt] == tgt for i in range(lr - lt + 1)) if 0 < lt <= lr else False
    contained = any(tgt[i:i + lr] == tr for i in range(lt - lr + 1)) if 0 < lr <= lt else False
    word, fuzzy = coder.best_word(tr) if tr else (None, False)
    has_word = word is not None
    zipf = coder.zipf(word, "en") if has_word else 0.0
    variant = float(has_word and (word == prompt or word.startswith(prompt) or prompt.startswith(word)))
    wup, shared_hyp, cos = 0.0, False, 0.0
    if has_word and word != prompt:
        wup, shared_hyp = semantic_similarity(prompt, word, coder.wn, coder.wn.NOUN)
        cos = coder.cosine(prompt, word)
    seg = coder.segment(tr, min_len=3) if lr >= 6 else None
    cat_onehot = [float(r["category"] == c) for c in MACHINE_CATS]
    return [
        lt, lr, lr - lt, abs(lr - lt), ed, ed / mx, float(bool(tr) and bool(tgt) and tr[0] == tgt[0]),
        float(bool(tr) and bool(tgt) and tr[-1] == tgt[-1]), pre, pre / max(lt, 1), float(pre == lr and lr < lt),
        shared, shared / max(lt, 1), same_pos, same_pos / max(lt, 1), shared_cons, shared_cons / max(len(tc), 1),
        float(bool(tv) and bool(rv) and tv[0] == rv[0]), float(contains), float(contained),
        float(has_word), float(fuzzy), zipf, variant, wup, float(shared_hyp), cos,
        float(bool(tgt) and bool(tr) and formally_related(tgt, tr)), float(len(seg)) if seg else 0.0,
        len(raw) - lr, CONF[r["confidence"]], *cat_onehot,
    ]


FEATURE_NAMES = ["len_target", "len_response", "len_diff", "abs_len_diff", "edit_distance", "edit_distance_norm",
                 "same_onset", "same_final", "common_prefix", "common_prefix_ratio", "is_prefix_fragment",
                 "shared_phonemes", "shared_ratio", "same_position", "same_position_ratio", "shared_consonants",
                 "shared_consonant_ratio", "same_first_vowel", "contains_target", "inside_target",
                 "is_word", "is_fuzzy_word", "word_zipf", "is_target_variant", "wordnet_wup", "shared_hypernym",
                 "embedding_cosine", "pnt_formal", "segment_words", "fillers_stripped", "machine_confidence",
                 *[f"machine_{c}" for c in MACHINE_CATS]]


def make_model(seed: int = SEED):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=600, min_samples_leaf=1, class_weight="balanced_subsample",
                                  random_state=seed, n_jobs=-1)


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-target-out predictions."""
    from sklearn.model_selection import LeaveOneGroupOut
    pred = np.empty_like(y)
    for tr_idx, te_idx in LeaveOneGroupOut().split(X, y, groups):
        m = make_model().fit(X[tr_idx], y[tr_idx])
        pred[te_idx] = m.predict(X[te_idx])
    return pred


def confusion_table(truth, pred, title: str) -> list[str]:
    cats = sorted(set(truth) | set(pred))
    conf = collections.defaultdict(collections.Counter)
    for t, p in zip(truth, pred):
        conf[t][p] += 1
    lines = [f"\n{title}\n", "| clinician \\ predicted | n | recall | " + " | ".join(cats) + " |", "|---|---|---|" + "---|" * len(cats)]
    for t in sorted(conf, key=lambda k: -sum(conf[k].values())):
        n = sum(conf[t].values())
        lines.append(f"| {t} | {n} | {conf[t][t] / n:.0%} | " + " | ".join(str(conf[t].get(c, 0)) for c in cats) + " |")
    return lines


def main() -> None:
    sheet = list(csv.DictReader(open(ADJ)))
    coded = {r["utterance_id"]: r for r in csv.DictReader(open(OUT))}
    # Use the current machine code (post-fix) as the feature, the clinician's label as truth.
    train = [dict(coded[r["utterance_id"]], final_category=r["final_category"].strip())
             for r in sheet if r["test"] == "BNT" and r["final_category"].strip()]
    if len(train) < 50:
        sys.exit(f"only {len(train)} adjudicated noun rows; not enough to learn from")
    coder = Coder()
    X = np.array([features(coder, r) for r in train], dtype=float)
    y = np.array([r["final_category"] for r in train])
    groups = np.array([r["prompt"].lower() for r in train])
    machine = np.array([r["category"] for r in train])

    cv_pred = cross_validate(X, y, groups)
    acc_cv, acc_machine = float(np.mean(cv_pred == y)), float(np.mean(machine == y))
    rng = np.random.default_rng(SEED)
    boots = [np.mean((cv_pred == y)[idx]) for idx in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Five-way SP grouping, which is what the engine comparison uses.
    sp = {"semantic": "semantic", "formal": "formal", "mixed": "mixed", "unrelated": "unrelated", "nonword_phon": "nonword",
          "nonword_other": "nonword", "fragment": "nonword"}
    sp_mask = np.array([t in sp for t in y])
    acc_sp_cv = float(np.mean([sp.get(p) == sp[t] for p, t in zip(cv_pred[sp_mask], y[sp_mask])]))
    acc_sp_machine = float(np.mean([sp.get(p) == sp[t] for p, t in zip(machine[sp_mask], y[sp_mask])]))

    model = make_model().fit(X, y)
    imp = sorted(zip(model.feature_importances_, FEATURE_NAMES), reverse=True)[:12]

    # Apply: noun error rows without a human code where the machine is below high confidence.
    targets = [coded[r["utterance_id"]] for r in sheet
               if r["test"] == "BNT" and not r["final_category"].strip() and coded[r["utterance_id"]]["confidence"] != "high"]
    learned: dict[str, tuple[str, float]] = {}
    if targets:
        Xa = np.array([features(coder, r) for r in targets], dtype=float)
        proba = model.predict_proba(Xa)
        for r, p in zip(targets, proba):
            k = int(np.argmax(p))
            learned[r["utterance_id"]] = (model.classes_[k], float(p[k]))
    before = collections.Counter(r["category"] for r in targets)
    after = collections.Counter(learned[r["utterance_id"]][0] for r in targets)

    fields = list(sheet[0].keys())
    for extra in ("learned_category", "learned_prob"):
        if extra not in fields:
            fields.append(extra)
    with open(ADJ, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sheet:
            lc, lp = learned.get(r["utterance_id"], ("", ""))
            w.writerow({**{k: r.get(k, "") for k in fields}, "learned_category": lc, "learned_prob": f"{lp:.2f}" if lp != "" else ""})

    lines = ["<!-- Generated by: uv run python -m scripts.learn_coder. Do not edit by hand. -->",
             "# Learned response-type coder\n",
             f"Trained on {len(train)} clinician-adjudicated noun rows ({len(set(groups))} target pictures), random forest on "
             f"{X.shape[1]} features of the target/response pair plus the machine's own code. Accuracy is by "
             f"leave-one-target-out cross-validation, so every row is predicted by a model that never saw its picture.\n",
             "| | all 9 categories | five SP error types |", "|---|---|---|",
             f"| rule-based machine code | {acc_machine:.0%} | {acc_sp_machine:.0%} |",
             f"| learned, cross-validated | {acc_cv:.0%} (95% bootstrap {lo:.0%} to {hi:.0%}) | {acc_sp_cv:.0%} |",
             "\nThe training rows are the ones the rule-based coder was unsure about, so these numbers describe performance "
             "on rows like those, which is where the model is applied; the machine's high-confidence rows (semantic with a "
             "WordNet relation, phonological nonwords within three edits, prefix fragments) are left as coded, because the "
             "clinician kept those categories when she saw them.",
             *confusion_table(y, cv_pred, "**Cross-validated confusion** (rows: clinician; columns: learned model)"),
             "\n**Most informative features**\n", "| feature | importance |", "|---|---|",
             *[f"| {n} | {v:.3f} |" for v, n in imp],
             f"\n**Applied** to {len(targets)} unadjudicated noun error rows below high confidence (written to the adjudication "
             f"sheet as `learned_category`, `learned_prob`). Mean confidence {np.mean([v[1] for v in learned.values()]):.2f}; "
             f"{sum(v[1] < 0.5 for v in learned.values())} rows below 0.5.\n",
             "| category | machine code | learned code |", "|---|---|---|",
             *[f"| {c} | {before.get(c, 0)} | {after.get(c, 0)} |" for c in sorted(set(before) | set(after), key=lambda c: -(before.get(c, 0) + after.get(c, 0)))]]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreport -> {REPORT}; sheet updated -> {ADJ}")


if __name__ == "__main__":
    main()
