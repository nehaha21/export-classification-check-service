"""
FAISS vector store for retrieval.
"""

from dataclasses import dataclass

import faiss
import numpy as np

from src.retrieval.chunking import DocumentChunk


@dataclass
class VectorSearchResult:
    """A chunk returned from vector similarity search."""

    chunk: DocumentChunk
    score: float


class FAISSVectorStore:
    """Store document embeddings and search them with FAISS."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")

        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[DocumentChunk] = []

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks and their embeddings to the index."""

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")

        if not chunks:
            return

        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.ndim != 2 or vectors.shape[1] != self.index.d:
            raise ValueError("Embedding dimensions do not match the index.")

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Return the most similar chunks."""

        if not query_embedding:
            return []

        if top_k <= 0:
            return []

        query = np.asarray([query_embedding], dtype="float32")

        if query.shape[1] != self.index.d:
            raise ValueError("Query embedding dimension does not match the index.")

        limit = min(top_k, len(self.chunks))

        if limit == 0:
            return []

        scores, indices = self.index.search(query, limit)

        results: list[VectorSearchResult] = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append(
                VectorSearchResult(
                    chunk=self.chunks[int(index)],
                    score=float(score),
                )
            )

        return results