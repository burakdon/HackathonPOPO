"""Try the automatic scorer by hand: score a typed transcript, an audio file, or a Mac-voice answer.

  uv run --extra llm --extra speech python -m scripts.score_answer cat "uuummm cat"
  uv run --extra llm --extra speech python -m scripts.score_answer tree --say "free"
  uv run --extra llm --extra speech python -m scripts.score_answer cat --audio ~/Desktop/my_answer.m4a

The first argument is the expected word, given as an exercise ID ("ex-cat") or its name ("cat");
its accepted alternatives come from data/items.json. Typed text skips speech recognition; --say and
--audio run the real Whisper model first.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from core.practice import load_items
from speech.base import Transcription
from speech.scoring import score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="exercise ID or picture name, e.g. ex-cat or cat")
    ap.add_argument("text", nargs="?", help="a typed transcript (skips speech recognition)")
    ap.add_argument("--audio", help="an audio file to transcribe (m4a, mp3, wav, webm, aiff...)")
    ap.add_argument("--say", help="have the Mac voice say this, then transcribe it")
    ap.add_argument("--voice", default="Samantha", help="Mac voice for --say")
    a = ap.parse_args()

    bank = load_items()
    item = bank.by_id.get(a.target) or next((i for i in bank.items if i.preferred_name == a.target.lower()), None)
    if item is None:
        names = ", ".join(i.preferred_name for i in bank.items)
        raise SystemExit(f"unknown picture {a.target!r}; pictures: {names}")
    if sum(x is not None for x in (a.text, a.audio, a.say)) != 1:
        raise SystemExit("give exactly one of: a typed transcript, --audio FILE, or --say TEXT")

    if a.text is not None:
        t = Transcription(status="ok", text=a.text, provider="typed")
    else:
        from speech.whisper_local import WhisperLocalProvider
        path = a.audio
        if a.say:
            path = str(Path(tempfile.mkdtemp()) / "say.aiff")
            subprocess.run(["say", "-v", a.voice, "-o", path, a.say], check=True)
        t = WhisperLocalProvider().transcribe(Path(path).expanduser().read_bytes(), None)
        print(f"recognizer: {t.status}  heard {t.text!r}  "
              f"words {[(w.word, w.probability) for w in t.words]}  no-speech {t.no_speech_prob}  {t.latency_ms} ms")

    s = score(t, item.preferred_name, item.accepted_alternatives)
    alts = f" (also accepted: {', '.join(item.accepted_alternatives)})" if item.accepted_alternatives else ""
    print(f"picture: {item.preferred_name}{alts}")
    print(f"result:  {s.outcome}" + (f", {s.error_type}" if s.error_type else "") + ("" if s.outcome == "unclear" else "  (counted)"))
    print(f"reason:  {s.reason}")


if __name__ == "__main__":
    main()
