"""Generate the PLACEHOLDER item bank (data/items_placeholder.csv) until a clinically curated bank is supplied.

Frequencies come from the open-source `wordfreq` package (Zipf scale: log10 occurrences per billion
words; Speer 2022), so no number is hand-typed. Length is the number of letters (a proxy; the real
bank should use phonemes or syllables). Tiers are difficulty terciles under the simulator's
difficulty model, so tier 1 = frequent/short, tier 3 = rare/long.

Usage:  uv run python -m scripts.build_placeholder_items
"""
from __future__ import annotations

import csv

import numpy as np
from wordfreq import zipf_frequency

from core.config import load_yaml, resolve

# 60 picturable concrete nouns spanning high to low frequency. PLACEHOLDER, not clinically curated.
WORDS = [
    "cat", "dog", "house", "car", "tree", "book", "ball", "cup", "chair", "bed",
    "door", "hand", "apple", "bird", "fish", "shoe", "hat", "key", "clock", "phone",
    "table", "bread", "egg", "milk", "sun", "moon", "boat", "bus", "train", "horse",
    "cow", "pig", "duck", "spoon", "fork", "knife", "bottle", "window", "flower", "banana",
    "carrot", "umbrella", "guitar", "ladder", "kettle", "scissors", "pencil", "hammer", "candle", "zebra",
    "giraffe", "penguin", "octopus", "cactus", "anchor", "walrus", "accordion", "stethoscope", "harmonica",
    "wheelbarrow",
]


def main() -> None:
    cfg = load_yaml("configs/sim.yaml")["items"]
    freq = np.array([zipf_frequency(w, "en") for w in WORDS])
    length = np.array([len(w) for w in WORDS])
    z = lambda v: (v - v.mean()) / v.std()
    difficulty = cfg["beta_frequency"] * -z(freq) + cfg["beta_length"] * z(length)
    tier = 1 + np.searchsorted(np.quantile(difficulty, [1 / 3, 2 / 3]), difficulty, side="right")

    out = resolve(cfg["path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "image_file", "frequency", "length", "tier"])
        for word, fr, ln, t in sorted(zip(WORDS, freq, length, tier), key=lambda r: (r[3], -r[1])):
            w.writerow([word, f"images/{word}.png", f"{fr:.2f}", int(ln), int(t)])
    print(f"wrote {len(WORDS)} items to {out}; per tier: {np.bincount(tier)[1:].tolist()}")


if __name__ == "__main__":
    main()
