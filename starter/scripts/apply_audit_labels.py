"""Write the grader's labels into the three audit CSVs and compute the audit summary.

GRADER = the lab author (with AI consultation; the AI also proposed an initial set of labels, kept unchanged 92/92
in the author_review column). Labels were assigned after reading the full chunk texts, using the rubric below.

Rubric (from GUIDE.md):
- coherence 0/1/2: 0 = cut mid-idea or missing subject so it cannot be understood; 1 = partly understandable,
  missing context or a truncated sentence/item at an edge; 2 = complete sentences/items with identifiable
  article context (a list may continue in the next chunk).
- grounding 0/1/2 (top-3 as a whole): 0 = no evidence of the answer; 1 = evidence present but missing a needed
  condition/context; 2 = enough to answer and trace to source.
- metadata: record is OK when source, page and chapter are present AND correct. Page is checked by finding the
  chunk text on that OCR page; chapter is checked against an independent Điều->Chương table read from the scans.
Run from starter/ after make_audit_sheets.py.
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chunking import fixed_size  # noqa: E402
from src.ingestion import load_cached_pages  # noqa: E402

GRADER = "Tôi (Nguyễn Trần Tiến), có tham khảo AI"
starter = Path(__file__).resolve().parents[1]
audit = starter / "results" / "audit"

COHERENCE = {  # row index in coherence_sample.csv -> (label, reason)
    0: (1, "Tiêu đề Điều 1 bị cắt đầu ('. Phạm vi điều chỉnh'), cuối cắt ở '3. Người'; nội dung Điều 1 vẫn hiểu được."),
    1: (2, "Trọn Điều 2 (4 khoản) kèm tiêu đề."),
    2: (1, "Bắt đầu giữa từ ('ệ lao động tập thể'), kết thúc cắt giữa khoản 8 ('tình trạng hôn nhân,')."),
    3: (2, "Các khoản 6-8 (định nghĩa) và đoạn tiếp trọn vẹn, mỗi khoản tự hiểu được dù thiếu tiêu đề Điều 3."),
    4: (1, "Bắt đầu giữa câu (đuôi Điều 4), trộn hai Điều, kết thúc cắt giữa từ ('tr')."),
    5: (2, "Điều 5 có tiêu đề, các điểm a-c trọn vẹn (danh sách quyền tiếp tục ở chunk sau)."),
    6: (1, "Bắt đầu giữa điểm c Điều 6, trộn khoản 1 và 2, kết thúc cắt ở ''œ) Đào'."),
    7: (1, "Bắt đầu ở điểm d, đ của Điều 6 không có tiêu đề, kết thúc bằng tiêu đề Điều 7 với '1.' dang dở; OCR dính dòng."),
    8: (1, "Bắt đầu giữa khoản 4 Điều 8, kết thúc cắt giữa chữ 'hợp đồn'."),
    9: (2, "Trọn Điều 8 (7 khoản) kèm tiêu đề."),
    10: (1, "Mất 'Điều 13' ở đầu, khoản 1 trọn vẹn nhưng khoản 2 bị cắt ('giao kêt h')."),
    11: (2, "Tiêu đề Chương/Mục + Điều 13 trọn vẹn đến khoản 2."),
    12: (1, "Bắt đầu giữa câu (cuối Điều 16), Điều 17 đầy đủ 2 khoản đầu nhưng khoản 3 bị cắt."),
    13: (2, "Trọn Điều 17 kèm tiêu đề."),
    14: (1, "Bắt đầu giữa câu (đuôi Điều 19.1), kết thúc cắt ở điểm a Điều 20 ('không xác định thời hạ')."),
    15: (2, "Trọn Điều 19 kèm tiêu đề."),
    16: (1, "Bắt đầu giữa danh sách Điều 21 (g-k) mất câu dẫn, kết thúc cắt ở 'bí mật công n'."),
    17: (1, "Mở đầu bằng các điểm h, i, k của Điều 21 mất câu dẫn và tiêu đề; phần khoản 2-4 sau đó trọn vẹn."),
    18: (1, "Bắt đầu bằng 'việc.' (đuôi câu), kết thúc cắt giữa Điều 25 ('Không quá 180 ngày đối'), tách câu dẫn khỏi các mức."),
    19: (2, "Trọn Điều 24 (3 khoản) kèm tiêu đề."),
}
GROUNDING = {  # (strategy, question) -> (label, reason)
    ("A", "t01"): (0, "Top-3 là Điều 121, 122 và 202; không có Điều 118 (quy tắc 'từ 10 người trở lên phải bằng văn bản'). Điều 121 chỉ nói về nội quy của đơn vị dưới 10 người, không trả lời được."),
    ("B", "t01"): (0, "Top-3 là Điều 120-121, Điều 3, Điều 14; không có bằng chứng về ngưỡng 10 người đối với nội quy bằng văn bản."),
    ("C", "t01"): (0, "Câu không có filter nên chạy như B; cùng top-3, không có bằng chứng."),
    ("A", "t02"): (0, "Chunk hạng 1 là Điều 47 (trợ cấp mất việc làm, 1 tháng/năm), dễ gây nhầm; chunk Điều 46 chỉ có khoản 2-3, thiếu khoản 1 'một nửa tháng lương'."),
    ("B", "t02"): (2, "Hạng 2 chứa trọn Điều 46 gồm điều kiện từ đủ 12 tháng, 'một nửa tháng tiền lương' và các ngoại lệ."),
    ("C", "t02"): (2, "Câu không có filter nên chạy như B; cùng top-3."),
    ("A", "t03"): (2, "Hạng 1 có Điều 202.1 (trên 50%, ra quyết định bằng văn bản); hạng 2-3 có Điều 201 cung cấp ngữ cảnh lấy ý kiến."),
    ("B", "t03"): (2, "Hạng 1 chứa trọn Điều 202 khoản 1-2; hạng 2 là Điều 201."),
    ("C", "t03"): (2, "Filter chương XIV cho cùng top-3 với B; đủ điều kiện và nguồn."),
    ("A", "t04"): (2, "Hạng 1 chứa trọn Điều 106 (22 giờ đến 06 giờ sáng hôm sau)."),
    ("B", "t04"): (2, "Hạng 1 là đúng Điều 106, trọn câu trả lời."),
    ("C", "t04"): (2, "Hạng 1 giống B; filter chương VII thay hạng 3 bằng chunk cùng chương (Điều 116)."),
    ("A", "t05"): (2, "Hạng 1 chứa Điều 115.1 với điều kiện 'hưởng nguyên lương và phải thông báo' và 'Kết hôn: nghỉ 03 ngày'."),
    ("B", "t05"): (2, "Hạng 1 chứa trọn Điều 115 (khoản 1-3)."),
    ("C", "t05"): (2, "Câu không có filter nên chạy như B; cùng top-3."),
    ("A", "t06"): (2, "Hạng 1 có Điều 169.2: từ 2021 là 60 tuổi 03 tháng đối với lao động nam (phần 'mỗi năm tăng 03 tháng' bị cắt nhưng không cần cho câu hỏi)."),
    ("B", "t06"): (2, "Hạng 1 chứa trọn Điều 169 khoản 1-2, có cả lộ trình tăng 03 tháng mỗi năm."),
    ("C", "t06"): (2, "Filter chương XII giữ hạng 1 giống B (Điều 169 đầy đủ); hạng 2-3 cùng chương nhưng không liên quan."),
}

# ---- independent chapter check (Điều ranges read from the scanned pages) ----
RANGES = [("I", 1, 8), ("II", 9, 12), ("III", 13, 58), ("IV", 59, 62), ("V", 63, 89), ("VI", 90, 104), ("VII", 105, 116),
          ("VIII", 117, 131), ("IX", 132, 134), ("X", 135, 142), ("XI", 143, 167), ("XII", 168, 169), ("XIII", 170, 178),
          ("XIV", 179, 211), ("XV", 212, 213), ("XVI", 214, 217), ("XVII", 218, 220)]
PAGE_RANGE = {"I": (1, 5), "II": (5, 6), "III": (6, 23), "IV": (23, 25), "V": (25, 35), "VI": (35, 39), "VII": (39, 43),
              "VIII": (43, 48), "IX": (48, 49), "X": (49, 52), "XI": (52, 60), "XII": (60, 61), "XIII": (61, 65),
              "XIV": (65, 78), "XV": (78, 78), "XVI": (78, 80), "XVII": (80, 83)}


def chap_of(n):
    return next((c for c, a, b in RANGES if a <= n <= b), None)


def flat(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def write(name, rows):
    for row in rows:  # columns recording the author's review against the AI's initial proposed label
        row.setdefault("author_review", "")  # agree / disagree
        row.setdefault("author_label", "")  # author's label when disagreeing
    with (audit / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read(name):
    return list(csv.DictReader((audit / name).open(encoding="utf-8-sig")))


pages = {p.page: p for p in load_cached_pages(starter / "data" / "ocr", "blld-2019", "x")}
summary = {"grader": GRADER}

coh = read("coherence_sample.csv")
for i, row in enumerate(coh):
    row["label_0_2"], row["reason"], row["grader"] = COHERENCE[i][0], COHERENCE[i][1], GRADER
write("coherence_sample.csv", coh)
for s in "AB":
    labels = [int(r["label_0_2"]) for r in coh if r["strategy"] == s]
    summary[f"coherence_{s}"] = {"n": len(labels), "counts_0_1_2": [labels.count(k) for k in range(3)],
                                 "mean_of_2": round(sum(labels) / len(labels), 2)}

meta = read("metadata_records.csv")
per = {}
for r in meta:
    p, text = int(r["meta_page"]), r["chunk_text"]
    page_ok = flat(text)[:80] in flat(pages[p].text) and flat(text)[-60:] in flat(pages[p].text)
    arts = [int(x) for x in re.findall(r"(?m)^\s*Điều\s+(\d+)\s*\.", text)]
    expected = chap_of(arts[0]) if arts else None
    lo, hi = PAGE_RANGE.get(r["meta_chapter"], (0, -1))
    complete = r["meta_source"] == "bo_luat_lao_dong_2019.pdf" and r["meta_chapter"] and r["meta_page"]
    ok = bool(complete and page_ok and lo <= p <= hi and (expected is None or expected == r["meta_chapter"]))
    basis = "chương khớp Điều" if arts else "chương chỉ đối chiếu với khoảng trang (chunk không có tiêu đề Điều)"
    r["metadata_ok"] = "yes" if ok else "no"
    r["error_note"] = "" if ok else "sai hoặc thiếu trường"
    r["grader"] = f"{GRADER}; {basis}"
    per.setdefault(r["strategy"], []).append(ok)
write("metadata_records.csv", meta)
for s, flags in per.items():
    summary[f"metadata_{s}"] = f"{sum(flags)}/{len(flags)}"
traces = json.loads((starter / "results" / "traces.json").read_text(encoding="utf-8"))
for s in "ABC":
    fq = [t for t in traces if t["split"] == "test" and t["strategy"] == s and t["filter"]]
    summary[f"filter_compliance_test_{s}"] = f"{sum(bool(t['filter_ok']) for t in fq)}/{len(fq)}"

gr = read("grounding_cases.csv")
for r in gr:
    r["label_0_2"], r["reason"] = GROUNDING[(r["strategy"], r["question_id"])]
    r["grader"] = GRADER
write("grounding_cases.csv", gr)
for s in "ABC":
    labels = [int(r["label_0_2"]) for r in gr if r["strategy"] == s]
    summary[f"grounding_{s}"] = {"n": len(labels), "counts_0_1_2": [labels.count(k) for k in range(3)],
                                 "mean_of_2": round(sum(labels) / len(labels), 2)}

# chunks that start in one chapter but contain the next chapter heading (A only; B always breaks at headings)
straddle = 0
total = 0
for page in pages.values():
    for chunk in fixed_size(page, 500, 50):
        total += 1
        start = page.text.find(chunk.text[:40])
        end = start + len(chunk.text)
        straddle += any(start < off < end for off, _ in page.metadata["chapter_spans"])
summary["A_chunks_straddling_a_chapter_heading"] = f"{straddle}/{total}"
summary["author_review"] = {}
for name in ("coherence_sample", "grounding_cases", "metadata_records"):
    reviewed = [r["author_review"] for r in read(f"{name}.csv")]
    summary["author_review"][name] = {"rows": len(reviewed), "agree": reviewed.count("agree"),
                                      "disagree": reviewed.count("disagree"), "not_reviewed": reviewed.count("")}
(audit / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1))
