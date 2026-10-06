"""
Embedding utilities for the retrieval pipeline.
"""

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Generate vector embeddings for document text."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple document chunks."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a search query."""
        if not query.strip():
            raise ValueError("query cannot be empty.")

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding.tolist()