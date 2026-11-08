"""Re-download the public data behind the patient model into data/raw/ (see data/README.md).

Gated sources (PSST, MAPPD) are not fetched here. Run:  uv run python -m scripts.fetch_data
"""
from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (NeuroCue data fetch)"}

IPNP = "https://crl-experiments.ucsd.edu/ipnp/"
FILES = {
    "ipnp/USpno.xls": IPNP + "method/getdata/USpno.xls",
    "ipnp/7lgpno.xls": IPNP + "7lgpno/getdata/7lgpno.xls",
    "ipnp/7lgpnovariables.html": IPNP + "7lgpno/getdata/7lgpnovariables.html",
    "ipnp/obj244OrigPNG.zip": IPNP + "method/getpics/obj244OrigPNG.zip",
    "refs/mirman2010_mappd.pdf": "https://dmirman.github.io/papers/Mirman_etal2010_MAPPD.pdf",
    "refs/dell2004_omission.pdf": "https://project-ccap.github.io/2004Dell_Models_of_errors_of_omission_in_aphasic_naming.pdf",
    "refs/dell2013_simulating.pdf": "https://www.cmu.edu/dietrich/psychology/nozarilab/papers/2013DellEtAl.pdf",
    "refs/castro2025_aphasia_databases.pdf": "https://talkbank.org/aphasia/publications/2025/Castro25.pdf",
}
OSF_NODE = "cwq75"  # Middleton, Rawson & Verkuilen 2019


def get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.relative_to(ROOT)}")
        return
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"  got  {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)")


def main() -> int:
    print("IPNP norms, pictures and reference papers")
    for rel, url in FILES.items():
        try:
            get(url, ROOT / rel)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {rel}: {e}", file=sys.stderr)
    pics = ROOT / "ipnp" / "obj244OrigPNG.zip"
    if pics.exists() and not (ROOT / "ipnp" / "244OrigPNG").exists():
        zipfile.ZipFile(pics).extractall(ROOT / "ipnp")
        print("  unzipped 244OrigPNG/")

    print("Middleton 2019 item-level data (OSF)")
    api = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/"
    with urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=60) as r:
        listing = json.load(r)["data"]
    for f in listing:
        name = f["attributes"]["name"].replace(" ", "_")
        get(f["links"]["download"], ROOT / "middleton2019_osf" / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
