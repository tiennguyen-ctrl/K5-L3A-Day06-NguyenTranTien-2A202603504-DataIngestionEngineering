"""OpenAI embedding adapter. Real API only: any failure raises, never falls back to mock vectors."""
import os
import time
from typing import Protocol

DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}
# USD per 1M input tokens (OpenAI list price when written; re-check before reporting cost).
PRICE_PER_M_TOKENS = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13, "text-embedding-ada-002": 0.10}


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """OpenAI embeddings. The same model embeds documents and queries (symmetric, no prefixes)."""

    def __init__(self, model_name: str | None = None, batch_size: int = 64, retries: int = 4) -> None:
        from openai import OpenAI

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or ""
        if self.model_name not in DIMENSIONS:
            raise ValueError(f"Unknown EMBEDDING_MODEL {self.model_name!r}; expected one of {sorted(DIMENSIONS)}")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in this terminal or starter/.env")
        self.dimensions = DIMENSIONS[self.model_name]
        self.client = OpenAI()
        self.batch_size, self.retries = batch_size, retries
        self.total_tokens = 0  # accumulated input tokens, for cost reporting

    def _embed(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self.retries):
            try:
                response = self.client.embeddings.create(model=self.model_name, input=texts)
                self.total_tokens += response.usage.total_tokens
                vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
                if len(vectors) != len(texts) or any(len(v) != self.dimensions for v in vectors):
                    raise RuntimeError("Embedding response has unexpected count or dimensions")
                return vectors
            except Exception as error:  # rate limit / transient network; surface after last retry
                if attempt == self.retries - 1:
                    raise RuntimeError(f"OpenAI embedding failed after {self.retries} attempts: {error}") from error
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed(texts[i:i + self.batch_size]))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question])[0]

    def cost_usd(self) -> float:
        return self.total_tokens / 1_000_000 * PRICE_PER_M_TOKENS[self.model_name]
