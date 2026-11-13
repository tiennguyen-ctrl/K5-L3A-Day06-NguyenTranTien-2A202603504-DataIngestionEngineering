"""Generate the manual-audit sample sheets (run from starter/ after python -m src.run).

Sampling rules (fixed here, independent of retrieval scores):
- coherence: pages 1-10; position = middle character of the cleaned page text; the chunk of
  each strategy (A, B) that covers that position (first one if two overlap in A).
- metadata + grounding: all 6 test questions x strategies A, B, C x top-3 from traces.json.
Also writes results/audit/_reading_dump.txt with full texts, for the human/LLM grader.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chunking import by_paragraph, fixed_size  # noqa: E402
from src.ingestion import load_cached_pages  # noqa: E402

starter = Path(__file__).resolve().parents[1]
out = starter / "results" / "audit"
out.mkdir(parents=True, exist_ok=True)
pages = load_cached_pages(starter / "data" / "ocr", "blld-2019", "bo_luat_lao_dong_2019.pdf")
dump = []


def cover(page, chunks):
    """Chunk covering the page's middle character."""
    mid, search_from = len(page.text) // 2, 0
    for chunk in chunks:
        start = page.text.find(chunk.text[:40], search_from)
        if start < 0:
            start = page.text.find(chunk.text[:20])
        if start <= mid < start + len(chunk.text):
            return chunk, mid
        search_from = max(start, 0)
    raise RuntimeError(f"no chunk covers mid of page {page.page}")


rows = []
for page in pages[:10]:
    for strategy, chunks in (("A", fixed_size(page, 500, 50)), ("B", by_paragraph(page, 1000))):
        chunk, mid = cover(page, chunks)
        rows.append({"strategy": strategy, "page": page.page, "mid_offset": mid, "chunk_id": chunk.chunk_id,
                     "chunk_text": chunk.text, "label_0_2": "", "reason": "", "grader": ""})
with (out / "coherence_sample.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

traces = json.loads((starter / "results" / "traces.json").read_text(encoding="utf-8"))
test = [t for t in traces if t["split"] == "test"]
meta_rows, ground_rows = [], []
for t in test:
    for r in t["top3"]:
        meta_rows.append({"strategy": t["strategy"], "question_id": t["id"], "rank": r["rank"], "chunk_id": r["chunk_id"],
                          "meta_source": r["metadata"].get("source"), "meta_page": r["metadata"].get("page"),
                          "meta_chapter": r["metadata"].get("chapter"), "filter": json.dumps(t["filter"]),
                          "chunk_text": r["text"], "metadata_ok": "", "error_note": "", "grader": ""})
    ground_rows.append({"strategy": t["strategy"], "question_id": t["id"], "question": t["question"],
                        "evidence_quote": t["evidence_quote"], "top3_chunk_ids": ";".join(r["chunk_id"] for r in t["top3"]),
                        "label_0_2": "", "reason": "", "grader": ""})
for name, data in (("metadata_records.csv", meta_rows), ("grounding_cases.csv", ground_rows)):
    with (out / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
print(f"coherence rows={len(rows)} metadata rows={len(meta_rows)} grounding rows={len(ground_rows)}")
