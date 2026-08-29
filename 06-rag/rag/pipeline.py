"""
RAGPipeline ties the three stages together:

    load documents -> chunk -> index -> retrieve(query) -> generate(answer)

This is deliberately the "hello world" of RAG: no reranking, no query
rewriting, no hybrid search. The goal is to make the core loop legible before
layering on the more advanced techniques covered in 07-agentic-rag.
"""

import os
from typing import List, Tuple

from .chunking import Chunk, chunk_text
from .generator import generate_answer
from .retriever import TfidfRetriever


class RAGPipeline:
    def __init__(self, retriever=None, chunk_size: int = 120, overlap: int = 20):
        self.retriever = retriever or TfidfRetriever()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._chunks: List[Chunk] = []

    def load_directory(self, directory: str) -> None:
        """Read every .txt file in `directory`, chunk it, and build the index."""
        all_chunks: List[Chunk] = []
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".txt"):
                continue
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            doc_id = os.path.splitext(filename)[0]
            all_chunks.extend(
                chunk_text(text, doc_id, chunk_size=self.chunk_size, overlap=self.overlap)
            )

        if not all_chunks:
            raise ValueError(f"No .txt documents found in {directory}")

        self._chunks = all_chunks
        self.retriever.index(all_chunks)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        return self.retriever.retrieve(query, k=k)

    def ask(self, question: str, k: int = 3) -> str:
        retrieved = self.retrieve(question, k=k)
        return generate_answer(question, retrieved)
