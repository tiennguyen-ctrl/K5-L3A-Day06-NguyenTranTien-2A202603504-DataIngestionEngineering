"""OCR the Labor Code PDF into data/ocr/ and write data/ocr_log.csv (run from starter/)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion import assign_chapters, load_pdf  # noqa: E402

data = Path(__file__).resolve().parents[1] / "data"
pages = load_pdf(data / "bo_luat_lao_dong_2019.pdf", "blld-2019", "bo_luat_lao_dong_2019.pdf",
                 cache_dir=data / "ocr", log_path=data / "ocr_log.csv")
assign_chapters(pages)
summary = [{"page": p.page, "chars": len(p.text), **p.metadata} for p in pages]
(data / "ocr_chapters.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"pages={len(pages)} total_chars={sum(len(p.text) for p in pages)}")
for row in summary:
    if row["chapters"]:
        print(f"page {row['page']}: start={row['chapter_start']} headings={row['chapters']}")
