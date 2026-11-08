"""Code every PSST naming response into a response type, automatically, with a confidence.

PSST labels each response only correct / incorrect. The engine and the speech scorer need the
six Philadelphia Naming Test categories plus the things the PNT folds into "other". Rules, applied
to the stress-stripped ARPAbet transcript against the target's CMUdict pronunciation:

  correct            PSST says correct
  correct_variant    transcript is the target plus a plural/-ing/possessive suffix (PSST scores strict)
  semantic           a single dictionary word related in meaning: WordNet Wu-Palmer >= .6 between the
                     two most common senses of each (same part of speech as the test), or a shared
                     direct hypernym, or sentence-embedding cosine >= .5 (associative relations WordNet
                     lacks: hammock -> swing, volcano -> fire); not formally related
  formal             a single dictionary word meeting the PNT formal criterion, not semantic
  mixed              both of the above
  unrelated          a single dictionary word, neither
  nonword_phon       not a word; within +-2 phonemes of the target's length; formally related
                     (shares onset, coda, a same-position phoneme or two phonemes anywhere)
  nonword_other      not a word; near the target's length; shares nothing with it
  fragment           not a word; at least 2 phonemes shorter than the target; formally related
                     (a partial attempt, e.g. "B R AH SH" for toothbrush)
  multiword          longer than the target by 3+ phonemes and segmentable into dictionary words:
                     a description, circumlocution or several attempts
  unclassified       nothing above applies

Dictionary words: only CMUdict entries that are real vocabulary (Zipf frequency >= 2.5 and known to
WordNet, or Zipf >= 4), matched
exactly or, failing that, within one phoneme edit for words of 3+ phonemes ("CH EY R" -> chair).
Leading filled pauses (uh, um, "the", "a") are stripped before matching.

Confidence: high when one rule fires cleanly on a single-word match or a near-length nonword;
medium when the decision rests on the length window or on greedy segmentation; low for everything
else. Rows below high confidence go to the adjudication sheet for a clinician.

Outputs: data/derived/psst_coded.csv (all utterances), data/derived/psst_adjudication.csv (rows for
human review, BNT first, with the machine's guess and reasoning and empty `final_category`,`notes`).
Usage:  uv run python -m scripts.code_psst_responses
"""
from __future__ import annotations

import collections
import csv
import glob
import re
from pathlib import Path

from patientmodel.lexicon import formally_related, phoneme_edit_distance, wordnet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "psst_coded.csv"
ADJ = ROOT / "data" / "derived" / "psst_adjudication.csv"
SUFFIXES = [("S",), ("Z",), ("IH", "Z"), ("AH", "Z"), ("IH", "NG"), ("D",), ("T",), ("IH", "D"), ("ER",), ("ER", "Z")]
FIELDS = ["utterance_id", "session", "test", "split", "aq_index", "prompt", "correctness", "transcript", "target_pron",
          "matched_word", "category", "confidence", "reason"]


def strip(ps) -> tuple[str, ...]:
    return tuple(re.sub(r"\d", "", p) for p in ps)


class Coder:
    def __init__(self) -> None:
        import cmudict
        self.cmu = cmudict.dict()
        from wordfreq import zipf_frequency
        self.zipf = zipf_frequency
        self.inv: dict[tuple, set[str]] = collections.defaultdict(set)
        self.by_len: dict[int, list[tuple]] = collections.defaultdict(list)
        for w, prons in self.cmu.items():
            if w.isalpha() and len(w) >= 2 and zipf_frequency(w, "en") >= 2.5:
                for p in prons:
                    sp = strip(p)
                    if sp not in self.inv:
                        self.by_len[len(sp)].append(sp)
                    self.inv[sp].add(w)
        self.wn = wordnet()
        # Drop dictionary entries that are proper names or junk: keep a word only if WordNet knows
        # it or it is common (Zipf >= 4), so "bieber", "hammett", "cacace" do not count as words.
        keep = lambda w: zipf_frequency(w, "en") >= 4.0 or bool(self.wn.synsets(w))
        for sp in list(self.inv):
            ws = {w for w in self.inv[sp] if keep(w)}
            if ws:
                self.inv[sp] = ws
            else:
                del self.inv[sp]
        self.by_len = collections.defaultdict(list)
        for sp in self.inv:
            self.by_len[len(sp)].append(sp)
        self._emb_model = None
        self._emb: dict[str, object] = {}

    FILLERS = [("AH", "M"), ("DH", "AH"), ("AH",), ("EH", "M"), ("IH", "T", "S")]

    def strip_fillers(self, tr: tuple) -> tuple:
        changed = True
        while changed and len(tr) > 2:
            changed = False
            for f in self.FILLERS:
                if tr[: len(f)] == f and len(tr) - len(f) >= 2:
                    tr = tr[len(f):]; changed = True
        return tr

    def best_word(self, pron: tuple) -> tuple[str | None, bool]:
        """(word, fuzzy): an exact dictionary match, else the most frequent word within one phoneme
        edit for pronunciations of 3+ phonemes."""
        ws = self.inv.get(pron)
        if ws:
            return max(ws, key=lambda w: self.zipf(w, "en")), False
        if len(pron) >= 3:
            best, best_z = None, -1.0
            for L in (len(pron) - 1, len(pron), len(pron) + 1):
                for cand in self.by_len.get(L, []):
                    if phoneme_edit_distance(pron, cand) <= 1:
                        w = max(self.inv[cand], key=lambda w: self.zipf(w, "en"))
                        z = self.zipf(w, "en")
                        if z > best_z:
                            best, best_z = w, z
            if best is not None and best_z >= 3.0:
                return best, True
        return None, False

    def cosine(self, a: str, b: str) -> float:
        if self._emb_model is None:
            from sentence_transformers import SentenceTransformer
            self._emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        for w in (a, b):
            if w not in self._emb:
                self._emb[w] = self._emb_model.encode([w], normalize_embeddings=True, show_progress_bar=False)[0]
        return float(self._emb[a] @ self._emb[b])

    def related(self, a: str, b: str, pos) -> tuple[bool, str]:
        """Semantic relatedness with the evidence used: WordNet on the two most common senses of the
        test's part of speech, else embedding cosine as an associative backstop."""
        wn = self.wn
        sa, sb = wn.synsets(a, pos=pos)[:2], wn.synsets(b, pos=pos)[:2]
        for x in sa:
            for y in sb:
                if x == y or set(x.hypernyms()) & set(y.hypernyms()):
                    return True, "WordNet shared hypernym"
                v = x.wup_similarity(y)
                if v is not None and v >= 0.6:
                    return True, f"WordNet wup {v:.2f}"
        c = self.cosine(a, b)
        if c >= 0.5:
            return True, f"embedding cosine {c:.2f}"
        return False, f"no WordNet relation, cosine {c:.2f}"

    def segment(self, t: tuple, min_len: int = 2) -> list[tuple] | None:
        n = len(t)
        best: list = [None] * (n + 1)
        best[0] = []
        for i in range(n):
            if best[i] is None:
                continue
            for j in range(i + min_len, min(n, i + 10) + 1):
                if t[i:j] in self.inv and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                    best[j] = best[i] + [t[i:j]]
        return best[n]

    def code(self, r: dict) -> dict:
        prompt = r["prompt"].lower()
        pos = self.wn.VERB if r["test"] == "VNT" else self.wn.NOUN
        tgts = [strip(p) for p in self.cmu.get(prompt, [])]
        tgt = tgts[0] if tgts else ()
        raw = tuple(p for p in r["transcript"].split() if not p.startswith("<"))
        tr = self.strip_fillers(raw)
        out = {k: r.get(k, "") for k in ("utterance_id", "session", "test", "split", "aq_index", "prompt", "correctness", "transcript")}
        out["target_pron"] = " ".join(tgt)
        out["matched_word"] = ""

        def done(cat, conf, reason, word=""):
            out.update(category=cat, confidence=conf, reason=reason, matched_word=word)
            return out

        if r["correctness"] == "True":
            return done("correct", "high", "PSST scored correct")
        if not tr:
            return done("no_response", "high", "empty transcript")
        if not tgt:
            return done("unclassified", "low", "target not in CMUdict")
        for t in tgts:
            if tr == t:
                return done("correct_variant", "medium", "matches a target pronunciation although PSST scored it incorrect")
            for suf in SUFFIXES:
                if tr == t + suf:
                    return done("correct_variant", "high", f"target plus suffix {' '.join(suf)}")
        ed_target = phoneme_edit_distance(tgt, tr)
        if len(tr) <= len(tgt) - 2 and tgt[: len(tr)] == tr:
            return done("fragment", "high", f"first {len(tr)} phonemes of the target, then stops")
        word, fuzzy = self.best_word(tr)
        if word is not None and fuzzy and ed_target <= 1:
            word = None                                   # a near-miss of the target is a phonological error, not a word choice
        if word is not None:
            tag = f"{'~' if fuzzy else ''}{word}"
            if word == prompt or (word.startswith(prompt) or prompt.startswith(word)) and abs(len(prompt) - len(word)) <= 2:
                return done("correct_variant", "medium", f"morphological variant {tag}", word)
            form = formally_related(tgt, tr)
            sem, why = self.related(prompt, word, pos)
            base = "medium" if fuzzy else "high"
            if sem and form:
                return done("mixed", base, f"word {tag}: {why}; formally related", word)
            if sem:
                return done("semantic", base if "WordNet" in why else "medium", f"word {tag}: {why}", word)
            if form:
                conf = base if (tr[0] == tgt[0] or ed_target <= 2) else "medium"
                return done("formal", conf, f"word {tag}: PNT formal criterion, edit distance {ed_target}; {why}", word)
            return done("unrelated", "medium", f"word {tag}: {why}, not formally related", word)
        dl = len(tr) - len(tgt)
        form = formally_related(tgt, tr)
        if abs(dl) <= 2:
            if form:
                return done("nonword_phon", "high" if ed_target <= 3 else "medium", f"nonword, edit distance {ed_target}, formally related")
            return done("nonword_other", "medium", "nonword near target length, shares nothing with target")
        if dl < -2:
            return done("fragment" if form else "nonword_other", "medium", f"{-dl} phonemes shorter than target" + (", formally related" if form else ""))
        seg = self.segment(tr, min_len=3)
        if seg and len(seg) >= 2:
            words = [self.best_word(s)[0] for s in seg]
            return done("multiword", "medium" if len(seg) <= 4 else "low", "segments as: " + " ".join(w or "?" for w in words))
        return done("multiword", "low", f"{dl} phonemes longer than target, not segmentable into words")


def main() -> None:
    rows = []
    for t in sorted(glob.glob(str(ROOT / "data/raw/psst/*/**/utterances.tsv"), recursive=True)):
        split = t.split("/psst/")[1].split("/")[0]
        with open(t) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                r["split"] = split
                rows.append(r)
    coder = Coder()
    coded = [coder.code(r) for r in rows]
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(coded)
    adj = [c for c in coded if c["confidence"] != "high"]
    adj.sort(key=lambda c: (c["test"] != "BNT", c["confidence"] != "medium", c["prompt"]))
    with open(ADJ, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["final_category", "notes"])
        w.writeheader()
        for c in adj:
            w.writerow({**c, "final_category": "", "notes": ""})
    cnt = collections.Counter((c["test"], c["category"]) for c in coded if c["correctness"] != "True")
    conf = collections.Counter((c["test"], c["confidence"]) for c in coded if c["correctness"] != "True")
    print(f"coded {len(coded)} utterances -> {OUT}")
    for test in ("BNT", "VNT"):
        n = sum(v for (t, _), v in cnt.items() if t == test)
        print(f"\n{test} incorrect ({n}):")
        for (t, cat), v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            if t == test:
                print(f"  {cat:16s} {v:5d}  {v / n:5.1%}")
        print("  confidence: " + ", ".join(f"{k[1]} {v}" for k, v in sorted(conf.items()) if k[0] == test))
    print(f"\nadjudication sheet: {len(adj)} rows ({sum(c['test'] == 'BNT' for c in adj)} BNT) -> {ADJ}")


if __name__ == "__main__":
    main()
