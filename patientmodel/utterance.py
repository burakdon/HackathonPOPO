"""How a response is spoken: the framing real patients put around the word they produce.

PSST transcripts show that a naming response is rarely the bare word. Real patients say "uh ...
house", "it's a ... house", "okay that's a ... home", start the word and restart it, or add "oh" or
"or something like that" after it. scripts/mine_utterance_frames.py counts that material (function
words only) before and after the response word in the PSST noun responses, by severity band; this
module samples from those counts so a spoken response is framed the way real responses were. The
response word itself (correct, error or nonword) comes from the engine and is not changed here.

Rates by band (share of responses with a frame): AQ<50 1%, AQ 50-69 7%, AQ 70-84 8%, AQ 85+ 12%
(milder patients talk more); suffixes about 1%. PSST clips are trimmed around the response, so these
are lower bounds. A "<false start>" frame means the first sound(s) of the response, a pause, then the
response (e.g. "HH AE ... HH AW S").
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAMES_PATH = ROOT / "data" / "calibration" / "psst_utterance_frames.json"

# ARPAbet -> rough English respelling, so a speech engine can say a nonword or a false start.
RESPELL = {"AA": "ah", "AE": "a", "AH": "uh", "AO": "aw", "AW": "ow", "AY": "igh", "B": "b", "CH": "ch", "D": "d", "DH": "th",
           "EH": "e", "ER": "er", "EY": "ay", "F": "f", "G": "g", "HH": "h", "IH": "i", "IY": "ee", "JH": "j", "K": "k",
           "L": "l", "M": "m", "N": "n", "NG": "ng", "OW": "oh", "OY": "oy", "P": "p", "R": "r", "S": "s", "SH": "sh",
           "T": "t", "TH": "th", "UH": "oo", "UW": "oo", "V": "v", "W": "w", "Y": "y", "Z": "z", "ZH": "zh"}
FRAME_PRON = {"uh": ["AH"], "um": ["AH", "M"], "eh": ["EY"], "oh": ["OW"], "mm": ["M"], "okay": ["OW", "K", "EY"],
              "it's": ["IH", "T", "S"], "that's": ["DH", "AE", "T", "S"], "this": ["DH", "IH", "S"], "is": ["IH", "Z"],
              "a": ["AH"], "an": ["AE", "N"], "the": ["DH", "AH"], "i": ["AY"], "or": ["AO", "R"], "something": ["S", "AH", "M", "TH", "IH", "NG"],
              "like": ["L", "AY", "K"], "that": ["DH", "AE", "T"], "no": ["N", "OW"], "yeah": ["Y", "AE"], "think": ["TH", "IH", "NG", "K"],
              "there's": ["DH", "EH", "R", "Z"], "here's": ["HH", "IY", "R", "Z"], "well": ["W", "EH", "L"], "so": ["S", "OW"]}


def respell(phonemes) -> str:
    return "".join(RESPELL.get(re.sub(r"\d", "", p), p.lower()) for p in phonemes) if phonemes else ""


def band(aq: float) -> str:
    return "AQ<50" if aq < 50 else "AQ 50-69" if aq < 70 else "AQ 70-84" if aq < 85 else "AQ 85+"


@dataclass
class Segment:
    kind: str                 # frame | false_start | attempt | suffix
    text: str                 # what a speech engine says
    phonemes: list[str]       # what the mouth does
    pause_ms: int             # silence before this segment


class UtteranceShaper:
    def __init__(self, path: Path = FRAMES_PATH) -> None:
        self.frames = json.loads(path.read_text()) if path.exists() else None

    @staticmethod
    def _frame_text(frame: str) -> tuple[str, list[str]]:
        words = frame.split()
        if len(words) > 1 and words[-1] == "uh":      # "it's uh" is the article "a" said with a schwa
            words[-1] = "a"
        phon = [p for w in words for p in FRAME_PRON.get(w, [])]
        return " ".join(words), phon

    def shape(self, category: str, word: str | None, phonemes, aq: float, rng: np.random.Generator) -> list[Segment]:
        phon = [re.sub(r"\d", "", p) for p in (phonemes or [])]
        segs: list[Segment] = []
        if category == "omission" or (word is None and not phon):
            return [Segment("frame", "uh", ["AH"], 400)] if rng.random() < 0.5 else []
        b = band(aq)
        if self.frames:
            n = max(self.frames["n_responses"].get(b, 1), 1)
            pre = self.frames["prefix"].get(b, {}); post = self.frames["suffix"].get(b, {})
            if pre and rng.random() < sum(pre.values()) / n:
                keys, w = list(pre), np.array(list(pre.values()), dtype=float)
                frame = keys[int(rng.choice(len(keys), p=w / w.sum()))]
                if frame == "<false start>":
                    k = 1 if len(phon) < 4 else int(rng.integers(1, 3))
                    segs.append(Segment("false_start", respell(phon[:k]) + "-", phon[:k], 300))
                else:
                    text, fp = self._frame_text(frame)
                    segs.append(Segment("frame", text, fp, 250))
        segs.append(Segment("attempt", word if word else respell(phon), phon, 450 if segs else 0))
        if self.frames:
            post = self.frames["suffix"].get(b, {}); n = max(self.frames["n_responses"].get(b, 1), 1)
            if post and rng.random() < sum(post.values()) / n:
                keys, w = list(post), np.array(list(post.values()), dtype=float)
                text, fp = self._frame_text(keys[int(rng.choice(len(keys), p=w / w.sum()))])
                segs.append(Segment("suffix", text, fp, 500))
        return segs

    @staticmethod
    def to_dicts(segs: list[Segment]) -> list[dict]:
        return [asdict(s) for s in segs]

    @staticmethod
    def text(segs: list[Segment]) -> str:
        out = ""
        for s in segs:
            out += (" … " if s.pause_ms >= 400 and out else (" " if out else "")) + s.text
        return out
