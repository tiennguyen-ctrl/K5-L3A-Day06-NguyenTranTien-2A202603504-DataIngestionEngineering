"""Copy the author's review columns (author_review, author_label) from the spreadsheets they filled in
into the project's audit CSVs. All other columns stay as generated (Excel can alter e.g. 'null' cells).

Usage (from starter/): python scripts/merge_author_review.py <folder with *_reviewed.csv>
"""
import csv
import sys
from pathlib import Path

audit = Path(__file__).resolve().parents[1] / "results" / "audit"
source = Path(sys.argv[1])
KEYS = {"coherence_sample": ("strategy", "page", "chunk_id"),
        "grounding_cases": ("strategy", "question_id"),
        "metadata_records": ("strategy", "question_id", "rank", "chunk_id")}
for name, keys in KEYS.items():
    ours = list(csv.DictReader((audit / f"{name}.csv").open(encoding="utf-8-sig")))
    theirs = list(csv.DictReader((source / f"{name}_reviewed.csv").open(encoding="utf-8-sig")))
    if len(ours) != len(theirs):
        sys.exit(f"{name}: row count differs ({len(ours)} vs {len(theirs)})")
    for mine, other in zip(ours, theirs):
        if [mine[k] for k in keys] != [other[k] for k in keys]:
            sys.exit(f"{name}: row mismatch {[mine[k] for k in keys]} vs {[other[k] for k in keys]}")
        review = other["author_review"].strip()
        if review not in ("agree", "disagree"):
            sys.exit(f"{name}: invalid author_review {review!r}")
        mine["author_review"], mine["author_label"] = review, other["author_label"].strip()
    with (audit / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ours[0]))
        writer.writeheader()
        writer.writerows(ours)
    print(name, "merged", len(ours), "rows")
