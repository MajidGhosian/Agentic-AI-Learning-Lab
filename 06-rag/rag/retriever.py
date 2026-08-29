"""
Retrieval backends for the RAG pipeline.

Two backends are provided:

- TfidfRetriever: pure scikit-learn, no model download required. Good default
  so the whole project runs offline in seconds. This is "sparse" retrieval —
  it matches on word overlap, weighted by how distinctive each word is.

- DenseRetriever: uses sentence-transformers to embed chunks into a vector
  space and retrieves by cosine similarity. This is what most production RAG
  systems use, because it can match on *meaning* rather than exact words
  (e.g. a query about "undoing an action" can retrieve a chunk about
  "human-in-the-loop confirmation" even with no shared vocabulary).

Both implement the same `.index(chunks)` / `.retrieve(query, k)` interface so
the pipeline can swap between them without changing any other code — this is
the same abstraction real RAG frameworks (LangChain, LlamaIndex) expose as a
"vector store" or "retriever".
"""

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk


class TfidfRetriever:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self._chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        texts = [c.text for c in chunks]
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks:
            raise RuntimeError("Call .index(chunks) before .retrieve(query)")

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices]


class DenseRetriever:
    """Optional embedding-based retriever using sentence-transformers.

    Requires the `sentence-transformers` package (see requirements.txt,
    commented out by default since it pulls in torch). Falls back with a
    clear error if the package isn't installed, rather than failing silently.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "DenseRetriever requires sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._embeddings = None
        self._chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        texts = [c.text for c in chunks]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        if self._embeddings is None or not self._chunks:
            raise RuntimeError("Call .index(chunks) before .retrieve(query)")

        query_vec = self._model.encode([query], normalize_embeddings=True)
        scores = cosine_similarity(query_vec, self._embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices]
