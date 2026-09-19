"""
A small, dependency-light document store + retriever.

This project is about the AGENT deciding when/what/how to retrieve — not
about building a production vector database — so the retriever intentionally
uses classic TF-IDF cosine similarity (scikit-learn) instead of an embeddings
API. That keeps the demo runnable with a single Anthropic API key and no
extra vector-DB infrastructure, while still exercising a realistic
"semantic-ish" search interface the agent can call as a tool.

Swap `KnowledgeBase._vectorize` for a real embeddings model (OpenAI,
Anthropic, sentence-transformers, etc.) plus a vector DB (Chroma/FAISS/
Pinecone) later — see /11-vector-databases for that upgrade — without
changing anything in agent.py, since the public `search()` interface stays
the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import DOCS_DIR, TOP_K


@dataclass
class Chunk:
    source: str
    text: str


def _load_documents(docs_dir: Path) -> list[Chunk]:
    """Read every .txt file in docs_dir and split it into paragraph chunks."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        # Split on blank lines -> numbered sections read reasonably as chunks
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        for para in paragraphs:
            # Skip the title/date header lines as their own chunk; fold them
            # into context instead by keeping them short and low-signal.
            if len(para.split()) < 4:
                continue
            chunks.append(Chunk(source=path.name, text=para))
    return chunks


class KnowledgeBase:
    """A tiny searchable knowledge base over the .txt files in data/docs."""

    def __init__(self, docs_dir: Path = DOCS_DIR):
        self.docs_dir = docs_dir
        self.chunks: list[Chunk] = _load_documents(docs_dir)
        if not self.chunks:
            raise RuntimeError(
                f"No documents found in {docs_dir}. Add some .txt files first."
            )
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Return the top_k most relevant chunks for `query`.

        Each result is a dict: {"source": str, "text": str, "score": float}
        so it serializes cleanly into a tool_result for the LLM.
        """
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        ranked_idx = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_idx:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            chunk = self.chunks[idx]
            results.append(
                {"source": chunk.source, "text": chunk.text, "score": round(score, 4)}
            )
        return results


if __name__ == "__main__":
    # Quick manual smoke test: python -m src.retriever
    kb = KnowledgeBase()
    for r in kb.search("how much is the home office stipend"):
        print(f"[{r['score']}] {r['source']}: {r['text'][:80]}...")
