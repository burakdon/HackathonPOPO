"""Generate the hint audio clips (docs/OPEN_DECISIONS.md D7) with the macOS voice, once.

For each of the 24 pictures, five clips under frontend/hints/<exercise_id>/:
  first_sound  first_letter  sentence_frame  semantic   the cue texts of data/items.json
  whole_word                                              the word itself

  say -v Samantha -o tmp.aiff "<text>"  ->  afconvert -f m4af -d aac tmp.aiff <clip>.m4a

Blank markers (___) are spoken as "blank". Existing clips are skipped, so a recording dropped in by hand survives;
--force regenerates everything. Hackathon only: check the license for macOS voices before sharing the clips.
See frontend/hints/README.md for replacing a clip.

Run:  uv run --extra llm --extra speech python -m scripts.make_hint_audio [--force] [--voice Samantha]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from core.config import resolve
from core.practice import load_items

CLIPS = ("first_sound", "first_letter", "sentence_frame", "semantic", "whole_word")


def spoken(text: str) -> str:
    t = text.replace("___", " blank ").replace("…", " ").replace("“", " ").replace("”", " ")
    return re.sub(r"\s+", " ", t).strip()


def clip_text(item, clip: str) -> str:
    return item.preferred_name if clip == "whole_word" else item.cues[clip]


def make_clip(text: str, out: Path, voice: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "clip.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), spoken(text)], check=True)
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(aiff), str(out)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="regenerate clips that already exist")
    ap.add_argument("--voice", default="Samantha")
    ap.add_argument("--out", default="frontend/hints")
    args = ap.parse_args()
    out = resolve(args.out)
    made = skipped = 0
    for item in load_items().items:
        for clip in CLIPS:
            path = out / item.exercise_id / f"{clip}.m4a"
            if path.exists() and not args.force:
                skipped += 1
                continue
            make_clip(clip_text(item, clip), path, args.voice)
            made += 1
    print(f"made {made} clips, skipped {skipped} that already exist, in {out}")


if __name__ == "__main__":
    main()
