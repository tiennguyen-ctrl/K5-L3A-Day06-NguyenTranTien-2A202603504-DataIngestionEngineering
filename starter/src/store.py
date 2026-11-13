"""Persistent vector store on Chroma (local, on disk, cosine, metadata filter).

Replaces the pgvector skeleton because Docker is not available on this machine; the
interface (schema / upsert / search with filters) is the same. One collection per strategy.
"""
from pathlib import Path
from typing import Any

from .models import Chunk

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "chroma_db"
ALLOWED_FILTERS = {"chapter", "document_id", "page"}


class ChromaStore:
    def __init__(self, path: Path | str = DEFAULT_PATH, dimensions: int = 0) -> None:
        import chromadb

        self.path, self.dimensions = str(path), dimensions
        self.client = chromadb.PersistentClient(path=self.path)

    def _collection(self, strategy: str):
        # hnsw:space=cosine -> distance = 1 - cosine similarity (approximate HNSW search)
        return self.client.get_or_create_collection(f"blld_{strategy.lower()}", metadata={"hnsw:space": "cosine"})

    def create_schema(self) -> None:
        for strategy in ("a", "b"):
            self._collection(strategy)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Write stable IDs (no duplicates on re-run) and delete obsolete chunks of the same documents."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length differ")
        if self.dimensions and any(len(v) != self.dimensions for v in vectors):
            raise ValueError(f"vector dimension != {self.dimensions}")
        collection = self._collection(strategy)
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id in input")
        for document_id in {c.document_id for c in chunks}:
            existing = collection.get(where={"document_id": document_id}, include=[])["ids"]
            obsolete = sorted(set(existing) - set(ids))
            if obsolete:
                collection.delete(ids=obsolete)
        collection.upsert(
            ids=ids, embeddings=vectors, documents=[c.text for c in chunks],
            metadatas=[{"document_id": c.document_id, "source": c.source, "page": c.page, **c.metadata} for c in chunks])

    def count(self, strategy: str) -> int:
        return self._collection(strategy).count()

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Top-k by cosine similarity; `filters` (equality on whitelisted fields) is applied inside Chroma."""
        where = None
        if filters:
            unknown = set(filters) - ALLOWED_FILTERS
            if unknown:
                raise ValueError(f"filter fields not allowed: {sorted(unknown)}")
            items = [{key: value} for key, value in filters.items()]
            where = items[0] if len(items) == 1 else {"$and": items}
        result = self._collection(strategy).query(
            query_embeddings=[query_vector], n_results=top_k, where=where,
            include=["documents", "metadatas", "distances"])
        return [{"chunk_id": cid, "text": text, "score": 1.0 - dist, "metadata": meta}
                for cid, text, meta, dist in zip(result["ids"][0], result["documents"][0],
                                                 result["metadatas"][0], result["distances"][0])]
