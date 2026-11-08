"""Accuracy of the automatic scorer on synthetic clips (docs/OPEN_DECISIONS.md D9).

Labeled clips are generated for every picture with several macOS voices, run through the real speech model
(WhisperLocalProvider) and the real scorer (speech.scoring.score), and compared with what each clip should
score as. Nothing is changed in the scorer: this only measures it.

Categories (the expected outcome in brackets):
  correct_word        the picture's name                          (correct)
  alt_word            an accepted alternative, when it has one    (correct)
  plural              the name as a plural, where one is natural  (correct)
  filler_word         "um, <name>"                                (correct)
  wrong_word          a different real word                       (incorrect, meaning)
  near_miss           the name with a different first sound       (incorrect, sound)
  its_a               "it's a <name>"                             (unclear: more than one word)
  dont_know           "I don't know"                              (incorrect, no_response)
  silence             a generated silent WAV                      (unclear)

Writes results/scorer_synthetic.md (confusion table, error-type accuracy, the headline false-accept rate with
examples, latency and the caveat) and results/scorer_synthetic.csv (every clip: what was said, heard, expected
and actual). Clips are kept under runs/scorer_clips/ (gitignored).

Run:  uv run --extra llm --extra speech python -m scripts.scorer_report [--voices Samantha Daniel Karen Moira]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import subprocess
import wave
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from core.config import resolve
from core.practice import load_items
from speech.scoring import score

DEFAULT_VOICES = ("Samantha", "Daniel", "Karen", "Moira")        # en_US, en_GB, en_AU, en_IE

WRONG = {"cat": "dog", "dog": "cat", "apple": "orange", "house": "tent", "car": "bus", "tree": "flower", "fish": "bird",
         "key": "lock", "banana": "grape", "chair": "table", "bicycle": "train", "umbrella": "raincoat",
         "rabbit": "mouse", "glasses": "hat", "carrot": "potato", "hammer": "saw", "kangaroo": "koala",
         "octopus": "squid", "anchor": "boat", "helicopter": "airplane", "microscope": "telescope",
         "giraffe": "zebra", "pretzel": "cookie", "cactus": "flower"}
NEAR = {"cat": "tat", "dog": "bog", "apple": "opple", "house": "mouse", "car": "tar", "tree": "dree", "fish": "vish",
        "key": "tee", "banana": "panana", "chair": "share", "bicycle": "picycle", "umbrella": "ombrella",
        "rabbit": "wabbit", "glasses": "blasses", "carrot": "tarrot", "hammer": "mammer", "kangaroo": "tangaroo",
        "octopus": "actopus", "anchor": "onchor", "helicopter": "melicopter", "microscope": "nicroscope",
        "giraffe": "diraffe", "pretzel": "bretzel", "cactus": "tactus"}
NO_PLURAL = {"glasses", "fish"}        # already plural, or the plural is the same word

# category -> (expected outcome, expected error type)
EXPECTED = {"correct_word": ("correct", None), "alt_word": ("correct", None), "plural": ("correct", None),
            "filler_word": ("correct", None), "wrong_word": ("incorrect", "meaning"),
            "near_miss": ("incorrect", "sound"), "its_a": ("unclear", None),
            "dont_know": ("incorrect", "no_response"), "silence": ("unclear", None)}
SHOULD_BE_INCORRECT = ("wrong_word", "near_miss", "dont_know")


def plural(name: str) -> str:
    return name + ("es" if name.endswith(("s", "x", "ch", "sh")) else "s")


def label(outcome: str, error: str | None) -> str:
    return outcome if error is None else f"{outcome}:{error}"


def synth(voice: str, text: str, out: Path) -> Path:
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["say", "-v", voice, "-o", str(out), "--data-format=LEI16@16000", text], check=True)
    return out


def silence(out: Path, seconds: float = 1.5) -> Path:
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
            w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return out


def clips(voices: list[str], clip_dir: Path):
    """Yield (category, voice, item, said, path) for every labeled clip."""
    for item in load_items().items:
        name = item.preferred_name
        for v in voices:
            def make(cat: str, said: str, _v=v):
                key = hashlib.sha1(said.encode()).hexdigest()[:10]
                return (cat, _v, item, said, synth(_v, said, clip_dir / _v / f"{key}.wav"))
            yield make("correct_word", name)
            for alt in item.accepted_alternatives:
                yield make("alt_word", alt)
            if name not in NO_PLURAL:
                yield make("plural", plural(name))
            yield make("filler_word", f"um, {name}")
            yield make("wrong_word", WRONG[name])
            yield make("near_miss", NEAR[name])
            yield make("its_a", f"it's {'an' if name[0] in 'aeiou' else 'a'} {name}")
            yield make("dont_know", "I don't know")
            yield ("silence", v, item, "", silence(clip_dir / "silence.wav"))


def run(voices: list[str], clip_dir: Path, whisper_model: str) -> list[dict]:
    from speech.whisper_local import WhisperLocalProvider
    provider = WhisperLocalProvider(model=whisper_model, timeout_s=60)
    cache: dict[Path, tuple] = {}
    rows = []
    for cat, voice, item, said, path in clips(voices, clip_dir):
        if path not in cache:                                     # one transcription per distinct recording
            t = provider.transcribe(path.read_bytes(), "audio/wav")
            cache[path] = (t, t.latency_ms)
            fresh = True
        else:
            fresh = False
        t, latency = cache[path]
        s = score(t, item.preferred_name, item.accepted_alternatives)
        exp_o, exp_e = EXPECTED[cat]
        rows.append({
            "category": cat, "voice": voice, "exercise_id": item.exercise_id, "target": item.preferred_name,
            "said": said, "heard": t.text if t.status == "ok" else f"[{t.status}]",
            "expected_outcome": exp_o, "expected_error": exp_e or "", "actual_outcome": s.outcome,
            "actual_error": s.error_type or "", "matched_name": s.matched_name or "",
            "ok": (s.outcome, s.error_type) == (exp_o, exp_e), "reason": s.reason,
            "word_confidence": "" if s.signals.word_confidence is None else s.signals.word_confidence,
            "transcribe_ms": latency if fresh and latency is not None else "", "clip": str(path.relative_to(resolve("."))),
        })
    return rows


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0"


def table(header: list[str], body: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                      *["| " + " | ".join(str(c) for c in r) + " |" for r in body]])


def report(rows: list[dict], voices: list[str], whisper_model: str) -> str:
    n = len(rows)
    exp = lambda r: label(r["expected_outcome"], r["expected_error"] or None)      # noqa: E731
    act = lambda r: label(r["actual_outcome"], r["actual_error"] or None)          # noqa: E731
    exp_labels = ["correct", "incorrect:meaning", "incorrect:sound", "incorrect:no_response", "unclear"]
    act_labels = ["correct", "incorrect:meaning", "incorrect:sound", "incorrect:no_response", "incorrect:unclassified", "unclear"]
    conf = Counter((exp(r), act(r)) for r in rows)
    confusion = table(["expected \\ actual", *act_labels, "total"],
                      [[e, *[conf[(e, a)] or "." for a in act_labels], sum(conf[(e, a)] for a in act_labels)] for e in exp_labels])

    bycat = defaultdict(list)
    for r in rows:
        bycat[r["category"]].append(r)
    cats = table(["category", "should score as", "clips", "scored as expected"],
                 [[c, label(*EXPECTED[c]), len(bycat[c]), pct(sum(r["ok"] for r in bycat[c]), len(bycat[c]))] for c in EXPECTED])
    byvoice = table(["voice", "clips", "scored as expected"],
                    [[v, sum(r["voice"] == v for r in rows), pct(sum(r["ok"] for r in rows if r["voice"] == v), sum(r["voice"] == v for r in rows))]
                     for v in voices])

    inc = [r for r in rows if r["category"] in SHOULD_BE_INCORRECT]
    false_accept = [r for r in inc if r["actual_outcome"] == "correct"]
    as_incorrect = [r for r in inc if r["actual_outcome"] == "incorrect"]
    right_type = [r for r in as_incorrect if r["actual_error"] == r["expected_error"]]
    typ = table(["category", "should be incorrect", "came back incorrect", "with the expected error type", "came back correct", "came back unclear"],
                [[c, len(bycat[c]), pct(sum(r["actual_outcome"] == "incorrect" for r in bycat[c]), len(bycat[c])),
                  pct(sum(r["actual_outcome"] == "incorrect" and r["actual_error"] == r["expected_error"] for r in bycat[c]), len(bycat[c])),
                  pct(sum(r["actual_outcome"] == "correct" for r in bycat[c]), len(bycat[c])),
                  pct(sum(r["actual_outcome"] == "unclear" for r in bycat[c]), len(bycat[c]))] for c in SHOULD_BE_INCORRECT])
    ex_lines = "\n".join(f"- {r['category']}, voice {r['voice']}, picture “{r['target']}”: said “{r['said']}”, heard “{r['heard']}”, "
                         f"scored correct ({r['reason']})" for r in false_accept[:15]) or "- none"

    should_correct = [r for r in rows if r["expected_outcome"] == "correct"]
    false_reject = [r for r in should_correct if r["actual_outcome"] != "correct"]
    fr_lines = "\n".join(f"- {r['category']}, voice {r['voice']}, picture “{r['target']}”: said “{r['said']}”, heard “{r['heard']}”, "
                         f"scored {act(r)} ({r['reason']})" for r in false_reject[:15]) or "- none"
    misses = [r for r in rows if not r["ok"] and r not in false_accept and r not in false_reject]
    other_lines = "\n".join(f"- {r['category']}, {r['voice']}, “{r['target']}”: said “{r['said']}”, heard “{r['heard']}”, "
                            f"expected {exp(r)}, got {act(r)}" for r in misses[:20]) or "- none"

    lat = [float(r["transcribe_ms"]) for r in rows if r["transcribe_ms"] != ""]
    lat_s = sorted(lat)
    latency = (f"{len(lat)} distinct recordings transcribed (repeated phrases such as “I don't know” are transcribed once per voice): "
               f"mean {statistics.mean(lat):.0f} ms, median {statistics.median(lat):.0f} ms, "
               f"95th percentile {lat_s[int(0.95 * (len(lat_s) - 1))]:.0f} ms, max {max(lat):.0f} ms "
               f"(Whisper `{whisper_model}`, CPU, int8). Scoring itself takes well under a millisecond.") if lat else "n/a"

    return f"""# Scorer accuracy on synthetic clips

Generated by `scripts/scorer_report.py` on {date.today().isoformat()}. The numbers below come from one actual run:
{n} labeled clips, {len(voices)} macOS voices ({", ".join(voices)}), every picture of `data/items.json` (24),
speech recognition by Whisper `{whisper_model}` (`WhisperLocalProvider`), scoring by `speech.scoring.score` with its
current thresholds (unchanged). Every clip, with what was said, what was heard and the expected and actual result,
is in `results/scorer_synthetic.csv`.

> **Caveat: these numbers are optimistic and say nothing about patients.** Synthetic speech is far cleaner than
> real speech, and much cleaner than aphasic speech: no background noise, no hesitation or false starts, perfect
> timing, a single steady voice, and a pronunciation the recognizer is good at. The clips of the "near-miss"
> category are spelled non-words read by a text-to-speech voice, not the sound errors a patient produces. A scorer
> that does well here has only shown that its rules do what they say on clean input.

## Headline: how often an answer that should be incorrect came back "correct"

**{pct(len(false_accept), len(inc))}** of the clips that should score as incorrect (a wrong word, a first-sound
near-miss, "I don't know") came back as correct.

{ex_lines}

For the other direction, clips that should be correct (the name, an accepted alternative, a plural, “um, name”)
and did not come back correct: **{pct(len(false_reject), len(should_correct))}**. These make the patient say it
again or count as a failure.

{fr_lines}

## Error type for incorrect answers

Of the {len(inc)} clips that should be incorrect, {pct(len(as_incorrect), len(inc))} came back incorrect, and
{pct(len(right_type), len(as_incorrect))} of those carried the expected error type.

{typ}

## Confusion table (expected, in rows, against actual, in columns)

All {n} clips. A dot is zero.

{confusion}

## Per category

{cats}

## Per voice

{byvoice}

## Other disagreements (neither a false accept nor a false reject)

{other_lines}

## Latency

{latency}

## Notes

- `its_a` clips are expected to be unclear because only single-word answers are scored (D5). `silence` is a
  generated silent WAV; the recognizer provider reports it as empty, which the scorer treats as unclear.
- The error type depends on the recognizer: a near-miss that Whisper turns into a real word is scored as a
  wrong word or as correct. That is a property of the pipeline and is counted here as it came back.
- The scorer's thresholds were not changed for this report. If the data above suggests a change, it is proposed
  in the hand-off report, not applied.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--voices", nargs="+", default=list(DEFAULT_VOICES))
    ap.add_argument("--whisper-model", default="base.en")
    ap.add_argument("--out", default="results/scorer_synthetic")
    ap.add_argument("--clips", default="runs/scorer_clips")
    args = ap.parse_args()
    rows = run(args.voices, resolve(args.clips), args.whisper_model)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out.with_suffix(".md").write_text(report(rows, args.voices, args.whisper_model))
    bad = sum(not r["ok"] for r in rows)
    print(f"{len(rows)} clips, {bad} not scored as expected -> {out.with_suffix('.md')}, {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
