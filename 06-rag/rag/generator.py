"""
Generation step: takes the user's question plus retrieved chunks and produces
a grounded answer.

If ANTHROPIC_API_KEY is set in the environment, this calls the real Claude
API so you get an actual generated answer. If it isn't set, it falls back to
a "mock" generator that just assembles the retrieved context — this keeps the
demo runnable with zero setup, and makes it obvious what the model *would*
have received as context.
"""

import os
from typing import List, Tuple

from .chunking import Chunk

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say you "
    "don't know rather than guessing. Cite which source each fact comes from "
    "using the [source] labels given."
)


def _build_context_block(retrieved: List[Tuple[Chunk, float]]) -> str:
    lines = []
    for chunk, score in retrieved:
        lines.append(f"[source: {chunk.doc_id}] (relevance={score:.3f})\n{chunk.text}")
    return "\n\n".join(lines)


def generate_answer(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    context_block = _build_context_block(retrieved)
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return (
            "[MOCK MODE — no ANTHROPIC_API_KEY set, so no LLM call was made]\n\n"
            "Here is the context that would have been sent to Claude:\n\n"
            f"{context_block}\n\n"
            "Set ANTHROPIC_API_KEY to get a real generated answer to:\n"
            f'  "{question}"'
        )

    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "The anthropic package is required for live generation. "
            "Install it with: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=api_key)
    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
