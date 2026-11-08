"""The item bank: IPNP object pictures with measured naming norms and a difficulty index.

Source: International Picture Naming Project, US English norms (Bates et al. 2003; Szekely et al.
2004), `data/raw/ipnp/USpno.xls`, sheet `database`, 520 items. Columns used:
  ename    dominant name                 elex1   % of participants producing the target name
  erttar   mean RT to the target (ms)    lnefreq log frequency of the target name
  esyll    syllables                     enoname % no response
Difficulty is a z-scored composite of slow healthy naming RT, low frequency and length (equal
weights). Name agreement is reported but deliberately left out: low agreement in healthy adults
means the picture has alternative labels, not that the word is hard to retrieve, and a "correct"
synthetic response always returns the dominant name. The composite modulates the simulated
patient's effective lexical weights per item (patient.py); both the mechanism (a multiplicative
scaling of s and p) and its strength are ASSUMPTIONS to be calibrated against item-level patient data.
Items whose dominant name is not a single CMUdict word (33 compounds such as "wateringcan") are kept
in the bank with `pronounceable=False`; the realizer cannot produce errors for them, so evaluations
that need a lexicon skip them and should say so.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
IPNP_XLS = REPO_ROOT / "data" / "raw" / "ipnp" / "USpno.xls"
IPNP_PICS = REPO_ROOT / "data" / "raw" / "ipnp" / "244OrigPNG"


@dataclass(frozen=True)
class Item:
    item_id: str          # IPNP picnum, e.g. obj001
    word: str
    name_agreement: float  # elex1, proportion of healthy adults producing this name
    rt_ms: float
    log_freq: float
    syllables: int
    difficulty: float     # z-scored composite, higher = harder
    image: Path | None
    pronounceable: bool   # dominant name is a single word with a CMUdict pronunciation


def load_item_bank(path: Path = IPNP_XLS, require_image: bool = False) -> list[Item]:
    df = pd.read_excel(path, sheet_name="database")
    df = df.dropna(subset=["ename", "elex1", "erttar", "lnefreq", "esyll"]).copy()
    df["ename"] = df["ename"].astype(str).str.lower().str.strip()
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    # Harder = slower healthy RT, rarer, longer. Equal weights (ASSUMPTION). Name agreement excluded, see above.
    comp = z(np.log(df["erttar"])) - z(df["lnefreq"]) + z(df["esyll"])
    df["difficulty"] = (comp - comp.mean()) / comp.std()
    pics = {p.stem.lower(): p for p in IPNP_PICS.glob("*.png")} if IPNP_PICS.exists() else {}
    import cmudict
    cmu = cmudict.dict()
    items = []
    for _, r in df.iterrows():
        pic = pics.get(f"{r['picnum']}{r['ename']}".lower())
        if require_image and pic is None:
            continue
        items.append(Item(str(r["picnum"]), r["ename"], float(r["elex1"]), float(r["erttar"]), float(r["lnefreq"]),
                          int(r["esyll"]), float(r["difficulty"]), pic, bool(r["ename"].isalpha() and r["ename"] in cmu)))
    return items
