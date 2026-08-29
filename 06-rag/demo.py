"""
Interactive CLI demo for the RAG pipeline.

Usage:
    python demo.py                       # interactive Q&A loop
    python demo.py "What is ReAct?"      # ask a single question and exit
"""

import sys

from rag import RAGPipeline

DOCS_DIR = "sample_docs"


def print_retrieved(retrieved) -> None:
    print("\n--- Retrieved chunks ---")
    for chunk, score in retrieved:
        preview = chunk.text[:100].replace("\n", " ")
        print(f"  [{score:.3f}] {chunk.doc_id} :: {preview}...")
    print("------------------------\n")


def main() -> None:
    pipeline = RAGPipeline()
    pipeline.load_directory(DOCS_DIR)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        retrieved = pipeline.retrieve(question)
        print_retrieved(retrieved)
        answer = pipeline.ask(question)
        print(answer)
        return

    print("RAG demo ready. Loaded docs from:", DOCS_DIR)
    print("Type a question, or 'quit' to exit.\n")
    while True:
        question = input("> ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        retrieved = pipeline.retrieve(question)
        print_retrieved(retrieved)
        answer = pipeline.ask(question)
        print(answer, "\n")


if __name__ == "__main__":
    main()
