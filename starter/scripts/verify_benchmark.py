"""Check that every benchmark evidence_quote really is on its gold page (OCR text) and that
each filter chapter is a chapter present on / in force at that page. Run from starter/ after run_ocr.py."""
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion import load_cached_pages  # noqa: E402

root = Path(__file__).resolve().parents[2]
pages = load_cached_pages(root / "starter" / "data" / "ocr", "blld-2019", "bo_luat_lao_dong_2019.pdf")
by_page = {p.page: p for p in pages}


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def best_ratio(quote: str, text: str) -> float:
    q, t = flat(quote), flat(text)
    if q in t:
        return 1.0
    matcher = difflib.SequenceMatcher(None, t, q, autojunk=False)
    size = sum(block.size for block in matcher.get_matching_blocks())
    return size / max(len(q), 1)


questions = json.loads((root / "benchmark" / "questions.json").read_text(encoding="utf-8"))["questions"]
bad = 0
for q in questions:
    page = by_page.get(q["gold_page"])
    if page is None:
        print(f"{q['id']}: MISSING PAGE {q['gold_page']}")
        bad += 1
        continue
    ratio = best_ratio(q["evidence_quote"], page.text)
    status = "exact" if ratio == 1.0 else "fuzzy" if ratio >= 0.9 else "NOT FOUND"
    flt = q["filter"]
    chap_note = ""
    if flt:
        chapters = set(page.metadata["chapters"]) | {page.metadata["chapter_start"]}
        ok = flt["chapter"] in chapters
        chap_note = f" | filter {flt['chapter']} {'ok' if ok else 'MISMATCH'} (page chapters={sorted(c for c in chapters if c)})"
        bad += not ok
    bad += status == "NOT FOUND"
    print(f"{q['id']} p{q['gold_page']}: {status} ({ratio:.2f}){chap_note}")
print("problems:", bad)
