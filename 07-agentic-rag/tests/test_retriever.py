"""
Lightweight tests for the retriever that don't require an API key.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import KnowledgeBase  # noqa: E402


def test_knowledge_base_loads_documents():
    kb = KnowledgeBase()
    assert len(kb.chunks) > 0


def test_search_returns_relevant_result():
    kb = KnowledgeBase()
    results = kb.search("home office stipend amount")
    assert len(results) > 0
    # The stipend appears in both the remote work and expense docs
    sources = {r["source"] for r in results}
    assert sources & {"remote_work_policy.txt", "expense_reimbursement_policy.txt"}


def test_search_returns_empty_for_nonsense_query():
    kb = KnowledgeBase()
    results = kb.search("xyzzy quantum banana teleportation")
    assert results == [] or all(r["score"] < 0.05 for r in results)


def test_search_respects_top_k():
    kb = KnowledgeBase()
    results = kb.search("vacation days", top_k=2)
    assert len(results) <= 2


if __name__ == "__main__":
    test_knowledge_base_loads_documents()
    test_search_returns_relevant_result()
    test_search_returns_empty_for_nonsense_query()
    test_search_respects_top_k()
    print("All tests passed.")
