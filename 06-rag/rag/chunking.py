"""
Simple text chunking utilities.

Real-world RAG systems usually chunk by tokens, sentences, or semantic
boundaries. To keep this project dependency-light and easy to read, we chunk
by words with a configurable overlap, which is enough to demonstrate the core
idea: documents are too big to embed as a single vector, so we split them into
smaller, retrievable pieces.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 120,
    overlap: int = 20,
) -> List[Chunk]:
    """Split `text` into overlapping word-based chunks.

    Args:
        text: raw document text.
        doc_id: identifier for the source document (e.g. filename).
        chunk_size: number of words per chunk.
        overlap: number of words shared between consecutive chunks, so a
            sentence that spans a chunk boundary is not lost entirely.
    """
    words = text.split()
    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_value = " ".join(chunk_words)
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::chunk{index}",
                text=chunk_text_value,
            )
        )
        index += 1
        if end == len(words):
            break
        start += step

    return chunks
