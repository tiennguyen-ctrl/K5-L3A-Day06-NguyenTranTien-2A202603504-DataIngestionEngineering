"""Automatic retrieval evaluation (Hit@1, Evidence Hit@3, filter compliance) with full traces.

Human labels (chunk coherence, grounding, metadata audit) are NOT inferred here.
"""
import difflib
import re
from collections import defaultdict
from typing import Any

EVIDENCE_THRESHOLD = 0.9  # tolerance for OCR diacritic errors; exact match scores 1.0


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_coverage(quote: str, text: str) -> float:
    """Fraction of the evidence quote found in the chunk text (1.0 if it is a substring)."""
    q, t = _flat(quote), _flat(text)
    if not q:
        return 0.0
    if q in t:
        return 1.0
    matcher = difflib.SequenceMatcher(None, t, q, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(q)


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """A hit = chunk from the gold source and page whose text contains the full evidence."""
    gold_pages = question.get("gold_pages") or [question["gold_page"]]
    details = []
    for rank, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        coverage = evidence_coverage(question["evidence_quote"], hit["text"])
        is_hit = (meta.get("document_id") == question["gold_source_id"] and meta.get("page") in gold_pages
                  and coverage >= EVIDENCE_THRESHOLD)
        details.append({"rank": rank, "chunk_id": hit["chunk_id"], "score": round(hit["score"], 4),
                        "page": meta.get("page"), "chapter": meta.get("chapter"),
                        "evidence_coverage": round(coverage, 3), "hit": is_hit, "text": hit["text"],
                        "metadata": meta})
    flt = question.get("filter")
    return {
        "id": question["id"], "split": question["split"], "question": question["question"],
        "gold_source_id": question["gold_source_id"], "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"], "filter": flt,
        "hit1": bool(details and details[0]["hit"]),
        "hit3": any(d["hit"] for d in details),
        "filter_ok": (all(d["chapter"] == flt["chapter"] for d in details) if flt else None),
        "top3": details,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by strategy and split, reporting numerator/denominator, not only percentages."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        for split in (trace["split"], "all"):
            groups[(trace["strategy"], split)].append(trace)
    summary = {}
    for (strategy, split), items in sorted(groups.items()):
        with_filter = [t for t in items if t["filter"]]
        summary[f"{strategy}/{split}"] = {
            "n": len(items),
            "hit1": f"{sum(t['hit1'] for t in items)}/{len(items)}",
            "hit3": f"{sum(t['hit3'] for t in items)}/{len(items)}",
            "filter_compliance": (f"{sum(bool(t['filter_ok']) for t in with_filter)}/{len(with_filter)}"
                                  if with_filter else "N/A"),
        }
    return summary
