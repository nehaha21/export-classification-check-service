"""
Document chunking utilities for the retrieval pipeline.
"""

from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """A chunk of source text with its source metadata."""

    chunk_id: str
    text: str
    source: str
    metadata: dict[str, object]


def chunk_document(
    text: str,
    *,
    source: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """
    Split a document into overlapping word-based chunks.

    The overlap helps preserve context across chunk boundaries.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()

    if not words:
        return []

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]

        if not chunk_words:
            break

        chunk_id = f"{source}:chunk-{len(chunks)}"

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                text=" ".join(chunk_words),
                source=source,
                metadata={
                    "chunk_index": len(chunks),
                    "start_word": start,
                    "end_word": start + len(chunk_words),
                },
            )
        )

        if start + chunk_size >= len(words):
            break

    return chunks