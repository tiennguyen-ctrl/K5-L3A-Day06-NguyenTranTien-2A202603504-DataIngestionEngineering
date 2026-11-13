"""Day 06 pipeline: OCR text -> chunks (A, B) -> OpenAI embeddings -> Chroma -> top-3 traces.

Run from starter/:  python -m src.run            (reuses an existing index)
                    python -m src.run --reindex  (re-embeds; costs API tokens)
Strategies: A = fixed 500/50, B = paragraph <=1000, C = same index as B + chapter filter
(applied only to questions that carry a filter; others run unfiltered).
"""
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_cached_pages
from .store import ChromaStore

ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "starter"
RESULTS = STARTER / "results"
DOC_ID, SOURCE = "blld-2019", "bo_luat_lao_dong_2019.pdf"
STRATEGY_PARAMS = {"A": {"chunking": "fixed_size", "size": 500, "overlap": 50},
                   "B": {"chunking": "by_paragraph", "max_size": 1000},
                   "C": {"chunking": "by_paragraph (same index as B)", "max_size": 1000, "filter": "chapter, only for questions with a filter"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true", help="re-embed all chunks (API cost)")
    args = parser.parse_args()
    load_dotenv(STARTER / ".env")  # does not override variables already set in the terminal
    RESULTS.mkdir(exist_ok=True)

    pages = load_cached_pages(STARTER / "data" / "ocr", DOC_ID, SOURCE)
    content_pages = [p for p in pages if len(p.text) >= 200]
    if len(content_pages) < 30:
        raise SystemExit(f"Only {len(content_pages)} content pages; need >= 30")
    chunks = {"A": [c for p in pages for c in fixed_size(p, 500, 50)],
              "B": [c for p in pages for c in by_paragraph(p, 1000)]}
    benchmark = json.loads((ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))["questions"]
    if len(benchmark) < 20 or sum(q["split"] == "dev" for q in benchmark) < 15 or sum(q["split"] == "test" for q in benchmark) < 5:
        raise SystemExit("Benchmark needs >= 20 questions with >= 15 dev / >= 5 test")

    embedder = MyEmbedder()
    store = ChromaStore(dimensions=embedder.dimensions)
    store.create_schema()
    stats_path = RESULTS / "ingest_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    for strategy, items in chunks.items():
        lengths = sorted(len(c.text) for c in items)
        reuse = not args.reindex and store.count(strategy) == len(items) and strategy in stats
        if reuse:
            print(f"[{strategy}] index has {len(items)} chunks, reusing (use --reindex to re-embed)")
            continue
        tokens_before, started = embedder.total_tokens, time.perf_counter()
        vectors = embedder.embed_documents([c.text for c in items])
        embed_seconds = time.perf_counter() - started
        store.upsert(strategy, items, vectors)
        tokens = embedder.total_tokens - tokens_before
        stats[strategy] = {
            "chunks": len(items), "total_chars": sum(lengths), "min": lengths[0], "median": lengths[len(lengths) // 2],
            "max": lengths[-1], "embed_and_store_seconds": round(time.perf_counter() - started, 2),
            "embed_api_seconds": round(embed_seconds, 2), "input_tokens": tokens,
            "cost_usd": round(tokens / 1_000_000 * 0.02 if embedder.model_name == "text-embedding-3-small" else 0, 6),
            "indexed_at": datetime.now(timezone.utc).isoformat()}
        print(f"[{strategy}] indexed {len(items)} chunks, {tokens} tokens, {stats[strategy]['embed_and_store_seconds']}s")
    stats_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")

    # idempotency check: re-running upsert must not change the chunk count
    for strategy, items in chunks.items():
        assert store.count(strategy) == len(items), f"{strategy}: index count != chunk count"

    query_vectors = {q["id"]: embedder.embed_query(q["question"]) for q in benchmark}
    traces = []
    for strategy in ("A", "B", "C"):
        collection = "A" if strategy == "A" else "B"
        for q in benchmark:
            filters = q["filter"] if strategy == "C" and q["filter"] else None
            hits = store.search(collection, query_vectors[q["id"]], top_k=3, filters=filters)
            trace = evaluate_query(q, hits)
            trace["strategy"] = strategy
            trace["filter_applied"] = filters
            traces.append(trace)
    summary = summarize(traces)

    (RESULTS / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "run_config.json").write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(), "embedding_model": embedder.model_name,
        "dimensions": embedder.dimensions, "vector_store": f"chromadb {version('chromadb')} PersistentClient (HNSW, cosine)",
        "strategies": STRATEGY_PARAMS, "top_k": 3, "evidence_threshold": 0.9,
        "python": platform.python_version(), "openai": version("openai"), "pymupdf": version("pymupdf"),
        "embedding_input_tokens_this_run": embedder.total_tokens,  # documents indexed in this run + 21 queries
        "ocr": "tesseract 5.5.0, lang=vie (tessdata_best), 300 dpi, psm 4"}, indent=1), encoding="utf-8")
    for key, value in summary.items():
        print(key, value)
    print(f"Traces saved to {RESULTS}")


if __name__ == "__main__":
    main()
