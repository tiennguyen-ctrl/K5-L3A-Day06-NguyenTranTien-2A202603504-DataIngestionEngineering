"""Read and normalize source data (scanned PDF -> OCR -> normalized pages)."""
import csv
import io
import os
import re
import shutil
import unicodedata
from pathlib import Path

from .models import DocumentPage

STARTER_DIR = Path(__file__).resolve().parents[1]
TESSDATA_DIR = STARTER_DIR / "tessdata"
MSYS_TESSERACT = Path(r"C:\msys64\ucrt64\bin\tesseract.exe")
OCR_DPI = 300
OCR_LANG = "vie"
OCR_PSM = "4"
MIN_CHARS = 200  # pages shorter than this are flagged for manual review

# OCR misreads Roman numerals (II->HU, VII->VH, XIII->XHUI ...), so only the heading
# line is matched here and the chapter number is assigned by order of appearance.
CHAPTER_RE = re.compile(r"^\W*Ch[uư][oơ]ng\s+(?!tr[iì]nh)([IVXLCHU]{1,6})\W*$", re.IGNORECASE | re.MULTILINE)


def to_roman(n: int) -> str:
    out = ""
    for value, symbol in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out, n = out + symbol, n - value
    return out


def normalize_text(text: str) -> str:
    """NFC-normalize, drop OCR noise lines and collapse whitespace, keeping line breaks."""
    text = unicodedata.normalize("NFC", text).replace("\x0c", "\n")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if re.fullmatch(r"[\W_]{0,3}", line):  # empty or only 1-3 punctuation marks (OCR specks)
            line = ""
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _drop_page_number(text: str, page: int) -> str:
    """Remove the printed page number (plus OCR specks before it) at the top of the page.

    Looks only at the first 3 non-empty lines; a line counts as the page number when it is
    the number alone, optionally followed by <=2 junk characters (e.g. '52 Ỹ', '67.').
    """
    lines = text.split("\n")
    seen = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        seen += 1
        if re.fullmatch(rf"\W*{page}\W*\w{{0,2}}\W*", line.strip()):
            before = [x for x in lines[:i] if x.strip()]
            if all(len(x.strip()) <= 5 for x in before):
                return "\n".join(lines[i + 1:]).strip()
            return text
        if seen >= 3:
            break
    return text


def clean_page_text(raw: str, page: int) -> str:
    """Normalize OCR text and drop the page-number header; the text indexed and cited."""
    return _drop_page_number(normalize_text(raw), page)


def load_cached_pages(ocr_dir: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Rebuild pages (with chapter metadata) from cached OCR text, without running OCR."""
    pages = []
    for file in sorted(ocr_dir.glob("page_*.txt")):
        n = int(file.stem.split("_")[1])
        pages.append(DocumentPage(document_id, source, n, clean_page_text(file.read_text(encoding="utf-8"), n)))
    assign_chapters(pages)
    return pages


def _ocr_page(page, pytesseract, Image) -> str:
    pix = page.get_pixmap(dpi=OCR_DPI)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image, lang=OCR_LANG, config=f"--psm {OCR_PSM}")


def load_pdf(path: Path, document_id: str, source: str, cache_dir: Path | None = None,
             log_path: Path | None = None) -> list[DocumentPage]:
    """Extract each physical page. Uses the text layer if present, otherwise OCR.

    OCR output is cached per page in cache_dir (page_001.txt, ...) so reruns are
    deterministic and fast. Pages that are empty or very short are logged, not dropped.
    """
    import pymupdf
    import pytesseract
    from PIL import Image

    if not TESSDATA_DIR.joinpath(f"{OCR_LANG}.traineddata").exists():
        raise FileNotFoundError(f"Missing {OCR_LANG}.traineddata in {TESSDATA_DIR}")
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)
    if shutil.which("tesseract") is None:
        if not MSYS_TESSERACT.exists():
            raise FileNotFoundError("tesseract executable not found on PATH")
        pytesseract.pytesseract.tesseract_cmd = str(MSYS_TESSERACT)

    pages: list[DocumentPage] = []
    log_rows = []
    doc = pymupdf.open(path)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for index, page in enumerate(doc, start=1):
        cache_file = cache_dir / f"page_{index:03d}.txt" if cache_dir else None
        method = "cache"
        if cache_file and cache_file.exists():
            raw = cache_file.read_text(encoding="utf-8")
        else:
            raw = page.get_text().strip()
            method = "text_layer"
            # Scans may carry a tiny text layer (e.g. a digital-signature stamp); OCR those too.
            if len(raw) < MIN_CHARS:
                raw = _ocr_page(page, pytesseract, Image)
                method = f"ocr_{OCR_LANG}_{OCR_DPI}dpi_psm{OCR_PSM}"
            if cache_file:
                cache_file.write_text(raw, encoding="utf-8")
        text = clean_page_text(raw, index)
        status = "empty" if not text else "short" if len(text) < MIN_CHARS else "ok"
        log_rows.append({"page": index, "method": method, "chars": len(text), "status": status})
        pages.append(DocumentPage(document_id=document_id, source=source, page=index, text=text))
    if log_path:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["page", "method", "chars", "status"])
            writer.writeheader()
            writer.writerows(log_rows)
    return pages


def assign_chapters(pages: list[DocumentPage]) -> None:
    """Add `chapter_start` (chapter in force at top of page) and `chapters` (headings on page).

    Headings are 'Chương <số La Mã>' lines in the OCR text; the n-th heading in document
    order is chapter n (the OCR numeral is kept as `chapter_ocr_read` for audit). A page
    without a heading inherits the chapter of the previous page.
    """
    current, count = None, 0
    for page in pages:
        found, read, spans = [], [], []
        for match in CHAPTER_RE.finditer(page.text):
            count += 1
            found.append(to_roman(count))
            read.append(match.group(1).upper())
            spans.append((match.start(), to_roman(count)))
        page.metadata["chapter_start"] = current
        page.metadata["chapters"] = found
        page.metadata["chapter_ocr_read"] = read
        page.metadata["chapter_spans"] = spans  # (char offset of heading, chapter)
        if found:
            current = found[-1]


def chapter_at(page: DocumentPage, offset: int) -> str | None:
    """Chapter in force at a character offset of the page text (None before Chương I)."""
    chapter = page.metadata.get("chapter_start")
    for start, name in page.metadata.get("chapter_spans", []):
        if start <= offset:
            chapter = name
    return chapter
