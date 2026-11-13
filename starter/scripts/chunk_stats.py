"""Chunk the cached OCR pages with strategies A and B and print count / length stats (no API calls)."""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chunking import by_paragraph, fixed_size  # noqa: E402
from src.ingestion import load_cached_pages  # noqa: E402

pages = load_cached_pages(Path(__file__).resolve().parents[1] / "data" / "ocr", "blld-2019", "bo_luat_lao_dong_2019.pdf")

for name, fn in (("A fixed 500/50", lambda p: fixed_size(p, 500, 50)),
                 ("B paragraph<=1000", lambda p: by_paragraph(p, 1000))):
    chunks = [c for p in pages for c in fn(p)]
    lengths = [len(c.text) for c in chunks]
    print(f"{name}: chunks={len(chunks)} total_chars={sum(lengths)} min={min(lengths)} "
          f"median={statistics.median(lengths):.0f} mean={statistics.mean(lengths):.0f} max={max(lengths)} "
          f"no_chapter={sum('chapter' not in c.metadata for c in chunks)}")
    sample = next(c for c in chunks if c.page == 40)
    print("  sample", sample.chunk_id, sample.metadata, repr(sample.text[:120]))
