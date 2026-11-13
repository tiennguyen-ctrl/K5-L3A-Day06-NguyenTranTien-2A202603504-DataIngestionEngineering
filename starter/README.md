# Bài làm · Data Ingestion Engineering (Bộ luật Lao động 2019)

Pipeline: PDF scan → OCR tiếng Việt → chunk (A cố định / B theo cấu trúc) → embedding OpenAI → Chroma → top-3 + đánh giá.
Corpus: Bộ luật Lao động số 45/2019/QH14, 83 trang (`data/bo_luat_lao_dong_2019.pdf`, nguồn và giấy phép trong `data/manifest.json`).

| Thành phần | Lựa chọn |
|---|---|
| OCR | Tesseract 5.5.0, `lang=vie` (tessdata_best), 300 DPI, psm 4, qua PyMuPDF + pytesseract |
| Embedding | OpenAI `text-embedding-3-small`, 1536 chiều, cùng model cho tài liệu và câu hỏi |
| Vector store | Chroma 1.5.9 `PersistentClient` lưu tại `chroma_db/`, HNSW (xấp xỉ), cosine |
| Cấu hình | A: cố định 500 ký tự, overlap 50 · B: gom đoạn ≤1000 ký tự, mỗi Chương/Mục/Điều mở chunk mới · C: chỉ mục của B + filter `chapter` (chỉ cho câu hỏi có filter) |
| Metadata mỗi chunk | `document_id`, `source`, `page`, `chapter` (số chương La Mã gán theo thứ tự xuất hiện của heading; kiểm chứng với bản PDF) |

> Docker/pgvector của starter gốc **không dùng** (máy không có Docker): `docker-compose.yml`, `init.sql`, `scripts/check_setup.py` giữ nguyên nhưng không thuộc pipeline này.

## Cài đặt (Windows, Python 3.13)

```powershell
cd starter
python -m pip install -r requirements.txt
```

Cần cài thêm **Tesseract 5.x** (ví dụ gói `mingw-w64-ucrt-x86_64-tesseract-ocr` của MSYS2, hoặc bản cài Windows) và tải gói tiếng Việt (12 MB, không commit vào repo):

```powershell
New-Item -ItemType Directory -Force tessdata
Invoke-WebRequest https://github.com/tesseract-ocr/tessdata_best/raw/main/vie.traineddata -OutFile tessdata\vie.traineddata
```

`src/ingestion.py` tự tìm `tesseract` trên PATH (nếu không có sẽ thử `C:\msys64\ucrt64\bin\tesseract.exe`).

## Chạy

Chạy từ thư mục `starter/`. Key OpenAI **không** nằm trong repo: đặt trong terminal hoặc file `.env` (đã nằm trong `.gitignore`).

```powershell
$env:OPENAI_API_KEY="sk-..."; $env:EMBEDDING_MODEL="text-embedding-3-small"
python scripts/run_ocr.py            # OCR 83 trang -> data/ocr/, data/ocr_log.csv, data/ocr_chapters.json
python scripts/build_manifest.py     # data/manifest.json
python scripts/verify_benchmark.py   # kiểm tra evidence và filter của 21 câu trên text OCR
python scripts/chunk_stats.py        # thống kê chunk A/B (không gọi API)
python -m src.run                    # embedding, lưu Chroma, top-3 cho A/B/C -> results/
python -m src.run --reindex          # nạp lại toàn bộ (tốn token); kiểm tra không nhân đôi
python scripts/make_audit_sheets.py  # sinh mẫu chấm thủ công -> results/audit/
python scripts/apply_audit_labels.py # ghi nhãn và số liệu tổng hợp
```

Bước OCR đã có sẵn kết quả trong `data/ocr/` nên có thể bỏ qua; `python -m src.run` chỉ cần `data/ocr/`, `../benchmark/questions.json` và key.

## Kết quả và artifacts

- `../benchmark/questions.json`: 21 câu (15 dev, 6 test), gold trang, evidence, filter; cách tách và rủi ro rò rỉ nằm trong `meta`.
- `results/traces.json`: top-3 đầy đủ (văn bản chunk, score, metadata, evidence) cho 3 chiến lược × 21 câu; `results/summary.json`: Hit@1, Hit@3, tuân thủ filter.
- `results/reindex_check.json`: minh chứng nạp lại không nhân đôi (sau `--reindex`, số bản ghi trong Chroma = số chunk = số ID duy nhất: A 453, B 339).
- `results/ingest_stats.json`: số chunk, phân bố độ dài, token, thời gian, chi phí; `results/run_config.json`: model, tham số, phiên bản, thời điểm chạy.
- `results/audit/`: mẫu chấm thủ công (coherence, metadata, grounding) và `audit_summary.json`. Nhãn do tôi chấm có tham khảo AI (cột `grader`); AI đề xuất bộ nhãn ban đầu, tôi giữ nguyên 92/92 (cột `author_review`).
- `../artifacts/experiment_log.csv`: cấu hình và kết quả A/B/C.

## Ghi chú trung thực

- Thời gian ingest chỉ đo một lần, gồm độ trễ mạng. Chi phí tính theo giá niêm yết 0,02 USD / 1 triệu token của `text-embedding-3-small` (kiểm tra lại giá trước khi trích dẫn).
- OCR đọc sai một số dấu tiếng Việt và chữ số La Mã trong tiêu đề chương; `chapter` được gán theo thứ tự heading nên đúng, nhưng văn bản chunk vẫn mang lỗi OCR (xem `data/ocr_chapters.json`, trường `chapter_ocr_read`).
- Tập test (6 câu) đã được xem kết quả sau lần chạy đầu; tham số chunk không thay đổi sau đó.
