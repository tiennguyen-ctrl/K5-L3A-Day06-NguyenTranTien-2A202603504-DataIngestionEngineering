"""Write data/manifest.json from the cached OCR output (run from starter/ after run_ocr.py)."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion import load_cached_pages  # noqa: E402

root = Path(__file__).resolve().parents[2]
data = root / "starter" / "data"
pages = [{"page_id": f"blld-2019:p{p.page:03d}", "page": p.page, "chars": len(p.text)}
         for p in load_cached_pages(data / "ocr", "blld-2019", "bo_luat_lao_dong_2019.pdf")]
log = list(csv.DictReader((data / "ocr_log.csv").open(encoding="utf-8")))
flagged = [int(r["page"]) for r in log if r["status"] != "ok"]

manifest = {
    "sources": [{
        "source_id": "blld-2019",
        "title": "Bộ luật Lao động, số 45/2019/QH14",
        "source_path_or_url": "starter/data/bo_luat_lao_dong_2019.pdf",
        "origin": "https://vanban.chinhphu.vn/?pageid=27160&docid=198540 (Cổng thông tin điện tử Chính phủ; bản sao y bản chính, ký số ngày 12.12.2019 theo header trang 1)",
        "license_or_permission": "Văn bản quy phạm pháp luật công khai; theo Điều 15.1(a) Luật Sở hữu trí tuệ không thuộc đối tượng bảo hộ quyền tác giả (căn cứ do tác giả viện dẫn, chưa đối chiếu lại với văn bản luật gốc)",
        "accessed_at": "2026-11-13 (thời gian tạo file tải về: 16:48)",
        "version_or_snapshot": "Quốc hội thông qua 20/11/2019; hiệu lực 01/01/2021 (theo Điều 220 trong văn bản)",
        "physical_pages": len(pages),
        "counted_pages": [p["page"] for p in pages if p["page"] not in flagged],
        "flagged_pages_review": flagged,
        "page_counting_rule": "Mọi trang vật lý có nội dung điều luật được tính (kể cả trang 1 có tiêu đề + Điều 1-3 và trang 83 có điều khoản thi hành); không có trang bìa/mục lục/trang trắng riêng. Trang bị gắn cờ short/empty trong ocr_log.csv được liệt kê ở flagged_pages_review.",
        "text_extraction": "File là bản scan, không có text layer; OCR bằng Tesseract (lang=vie, tessdata_best, 300 DPI, psm 4) qua PyMuPDF + pytesseract",
        "total_chars_after_normalization": sum(p["chars"] for p in pages),
        "pages": pages,
    }]
}
(data / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
print("pages", len(pages), "chars", manifest["sources"][0]["total_chars_after_normalization"], "flagged", flagged)
