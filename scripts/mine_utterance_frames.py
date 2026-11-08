"""Mine how real PSST responses are framed: the function-word material before and after the
response word ("okay that's a ... home", "it's a ... house", "... no"), by severity band.

Only function words from a fixed list are kept, so the output holds phrases like "it's a" with
counts, never a transcript. Writes data/calibration/psst_utterance_frames.json, which
patientmodel/utterance.py samples from when the demo avatar speaks.

Run: uv run python -m scripts.mine_utterance_frames
"""
from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

import cmudict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "calibration" / "psst_utterance_frames.json"
FRAME_WORDS = ["okay", "ok", "oh", "um", "uh", "ah", "eh", "hmm", "it's", "its", "it", "is", "that's", "thats", "that",
               "this", "a", "an", "the", "there's", "theres", "there", "here's", "here", "i", "think", "know", "don't",
               "no", "not", "yeah", "yes", "well", "like", "maybe", "see", "let's", "lets", "so", "and", "or", "what",
               "called", "you", "would", "be", "say", "mean", "something", "kind", "of", "some", "sort", "to", "me",
               "wait", "good", "right", "hm", "mm"]
SPELL = {"its": "it's", "thats": "that's", "theres": "there's", "lets": "let's", "ok": "okay", "ah": "uh", "hm": "hmm", "mm": "mm"}


def strip(ps):
    return tuple(re.sub(r"\d", "", p) for p in ps)


def band(aq: float) -> str:
    return "AQ<50" if aq < 50 else "AQ 50-69" if aq < 70 else "AQ 70-84" if aq < 85 else "AQ 85+"


def main() -> None:
    cmu = cmudict.dict()
    inv: dict[tuple, str] = {}
    for w in FRAME_WORDS:
        for pr in cmu.get(w, []):
            inv.setdefault(strip(pr), SPELL.get(w, w))
    inv[("AH", "M")] = "um"; inv[("AH",)] = "uh"; inv[("EY",)] = "eh"; inv[("OW",)] = "oh"; inv[("M",)] = "mm"

    def segment(ph: tuple) -> list[str] | None:
        best: list = [None] * (len(ph) + 1); best[0] = []
        for i in range(len(ph)):
            if best[i] is None:
                continue
            for j in range(i + 1, min(len(ph), i + 6) + 1):
                if ph[i:j] in inv and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                    best[j] = best[i] + [inv[ph[i:j]]]
        return best[len(ph)]

    rows = [r for r in csv.DictReader(open(ROOT / "data/derived/psst_coded.csv")) if r["test"] == "BNT"]
    pre: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    post: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n_band: collections.Counter = collections.Counter()
    for r in rows:
        raw = tuple(p for p in r["transcript"].split() if not p.startswith("<"))
        if not raw:
            continue
        b = band(float(r["aq_index"])); n_band[b] += 1
        word = r["prompt"].lower() if r["correctness"] == "True" else r["matched_word"].lstrip("~")
        prons = [strip(p) for p in cmu.get(word, [])] if word else []
        hit = None
        for pr in prons:
            for i in range(len(raw) - len(pr), -1, -1):      # last occurrence of the word
                if raw[i:i + len(pr)] == pr:
                    hit = (i, i + len(pr)); break
            if hit:
                break
        if not hit:
            continue
        before, after = raw[: hit[0]], raw[hit[1]:]
        if before:
            seg = segment(before)
            pre[b][" ".join(seg) if seg else "<false start>"] += 1
        if after:
            seg = segment(after)
            if seg:
                post[b][" ".join(seg)] += 1
    out = {"source": "PSST BNT transcripts (TalkBank), function-word material around the response word; counts only",
           "n_responses": dict(n_band),
           "prefix": {b: dict(pre[b].most_common(30)) for b in n_band},
           "suffix": {b: dict(post[b].most_common(15)) for b in n_band}}
    OUT.write_text(json.dumps(out, indent=1))
    for b in ["AQ<50", "AQ 50-69", "AQ 70-84", "AQ 85+"]:
        tot = sum(pre[b].values())
        print(f"{b}: {n_band[b]} responses, {tot} with a prefix ({tot / max(n_band[b], 1):.0%}), {sum(post[b].values())} with a suffix")
        print("   prefixes:", ", ".join(f"{k} ({v})" for k, v in pre[b].most_common(10)))
        print("   suffixes:", ", ".join(f"{k} ({v})" for k, v in post[b].most_common(6)))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
