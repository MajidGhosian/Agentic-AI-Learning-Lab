"""
Demo CLI for the Agentic RAG project.

Usage:
    python main.py                       # runs a set of sample questions
    python main.py "your question here"  # ask a single question
    python main.py --chat                # interactive REPL

Requires ANTHROPIC_API_KEY to be set in the environment (see .env.example).
"""

import sys

from src.agent import AgenticRAG

SAMPLE_QUESTIONS = [
    # Single-hop: answerable from one document
    "How much is the home office equipment stipend?",
    # Multi-hop: touches two different policy documents
    "If I work remotely from another country for a month, what security "
    "steps and HR approvals do I need?",
    # No-retrieval-needed: general knowledge / chit-chat
    "What's a good icebreaker question for a new remote hire's first team call?",
    # Not in the knowledge base at all
    "What is Acme's parental leave policy?",
]


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_question(agent: AgenticRAG, question: str) -> None:
    print_header(f"Q: {question}")
    result = agent.ask(question, verbose=True)
    print(f"\n(used {result.num_searches} retrieval call(s))")


def main() -> None:
    agent = AgenticRAG()

    args = sys.argv[1:]

    if args and args[0] == "--chat":
        print("Agentic RAG chat. Type 'exit' to quit.\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            run_question(agent, question)
        return

    if args:
        question = " ".join(args)
        run_question(agent, question)
        return

    for question in SAMPLE_QUESTIONS:
        run_question(agent, question)


if __name__ == "__main__":
    main()
