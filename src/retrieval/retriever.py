"""
Retrieval orchestration for corpus search.
"""

from src.retrieval.chunking import DocumentChunk
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import FAISSVectorStore, VectorSearchResult


class Retriever:
    """Embed documents, build the FAISS index, and search it."""

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = FAISSVectorStore(dimension=384)

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed and add document chunks to the vector index."""

        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        self.vector_store.add(chunks, embeddings)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Search the indexed corpus for relevant chunks."""

        query_embedding = self.embedding_model.embed_query(query)

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

    def search_corpus(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Return corpus search results for the agent tools."""

        return self.search(
            query,
            top_k=top_k,
        )