import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunking import chunk_text
from rag.retriever import TfidfRetriever


def test_chunk_text_basic():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, doc_id="doc1", chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert all(c.doc_id == "doc1" for c in chunks)
    # consecutive chunks should overlap
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-5:] == second_words[:5]


def test_chunk_text_empty():
    assert chunk_text("", doc_id="empty") == []


def test_tfidf_retriever_finds_relevant_chunk():
    chunks = chunk_text(
        "The ReAct pattern interleaves reasoning and tool actions. "
        "Guardrails validate model inputs and outputs for safety.",
        doc_id="doc1",
        chunk_size=8,
        overlap=2,
    )
    retriever = TfidfRetriever()
    retriever.index(chunks)
    results = retriever.retrieve("What are guardrails used for?", k=1)
    assert len(results) == 1
    top_chunk, score = results[0]
    assert "guardrail" in top_chunk.text.lower() or "Guardrail" in top_chunk.text
    assert score > 0


if __name__ == "__main__":
    test_chunk_text_basic()
    test_chunk_text_empty()
    test_tfidf_retriever_finds_relevant_chunk()
    print("All tests passed.")
