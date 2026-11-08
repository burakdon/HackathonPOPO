"""Merge a returned adjudication file into the adjudication sheet by utterance_id.

Usage: uv run python -m scripts.merge_adjudication <returned.csv>

Copies non-empty `final_category` and `notes` from the returned file onto matching rows of
data/derived/psst_adjudication.csv (the clinician's entry wins over anything already there), then
reports what changed and how the machine and learned codes fared on the newly judged rows. Rerun
scripts.learn_coder and scripts.validate_engine afterwards.
"""
from __future__ import annotations

import collections
import csv
import sys

from scripts.code_psst_responses import ADJ


def main(path: str) -> None:
    returned = {r["utterance_id"]: r for r in csv.DictReader(open(path)) if r.get("final_category", "").strip()}
    sheet = list(csv.DictReader(open(ADJ)))
    fields = list(sheet[0].keys())
    on_sheet = {r["utterance_id"] for r in sheet}
    missing = [u for u in returned if u not in on_sheet]
    changed = 0
    machine = collections.Counter(); learned = collections.Counter()
    for r in sheet:
        new = returned.get(r["utterance_id"])
        if not new:
            continue
        fc = new["final_category"].strip()
        if r.get("final_category", "").strip() != fc or (new.get("notes", "").strip() and new["notes"] != r.get("notes", "")):
            changed += 1
        r["final_category"] = fc
        if new.get("notes", "").strip():
            r["notes"] = new["notes"]
        grp = r.get("sample_group") or "other"
        machine[(grp, r["category"] == fc)] += 1
        if r.get("learned_category"):
            learned[(grp, r["learned_category"] == fc)] += 1
    with open(ADJ, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(sheet)
    print(f"{len(returned)} judged rows in {path}; {changed} sheet rows changed; {len(missing)} ids not on the sheet")
    for grp in sorted({g for g, _ in machine}):
        n = machine[(grp, True)] + machine[(grp, False)]
        line = f"  {grp}: machine code kept on {machine[(grp, True)]}/{n}"
        ln = learned[(grp, True)] + learned[(grp, False)]
        if ln:
            line += f"; learned code right on {learned[(grp, True)]}/{ln}"
        print(line)
    print("next: uv run python -m scripts.learn_coder && uv run python -m scripts.validate_engine")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
