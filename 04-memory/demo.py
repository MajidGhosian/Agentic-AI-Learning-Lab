"""
demo.py
-------
Runs a scripted, offline demonstration of short-term vs long-term memory
with no API key required. Good for a quick sanity check that the concepts
and code work before wiring in a live model in agent_with_memory.py.

Usage:
    python demo.py
"""

from short_term_memory import ConversationBuffer, WindowedConversationBuffer
from long_term_memory import LongTermMemory


def divider(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_short_term() -> None:
    divider("SHORT-TERM MEMORY: lives only for this run")

    buf = ConversationBuffer()
    buf.add("user", "Let's call this project 'Atlas'.")
    buf.add("assistant", "Sure, I'll refer to it as Atlas from now on.")
    buf.add("user", "What's the project called?")

    print("Conversation so far (sent to the model every turn):")
    for m in buf.as_api_messages():
        print(f"  {m['role']:>9}: {m['content']}")

    print("\nSimulating end of session -> buf.clear()")
    buf.clear()
    print(f"Messages remaining: {len(buf)}  (the model has no memory of 'Atlas' anymore)")

    divider("SHORT-TERM MEMORY WITH A WINDOW (context limit simulation)")
    wbuf = WindowedConversationBuffer(max_turns=2)
    for i in range(1, 5):
        wbuf.add("user", f"Message {i} from user")
        wbuf.add("assistant", f"Reply {i} from assistant")
    print(f"After 4 turns with max_turns=2, buffer holds {len(wbuf)} messages:")
    for m in wbuf.as_api_messages():
        print(f"  {m['role']:>9}: {m['content']}")
    print("Notice turns 1 and 2 were silently dropped — same thing happens")
    print("when a real context window fills up.")


def demo_long_term() -> None:
    divider("LONG-TERM MEMORY: survives across separate 'sessions'")

    print("Session A: agent learns and saves facts")
    session_a = LongTermMemory("memory_store.json")
    session_a.remember("The user is working through an Agentic AI learning roadmap.", tags=["project"])
    session_a.remember("The user prefers concise, code-first explanations.", tags=["preference"])
    print(session_a.as_context_string())

    print("\nSession B: brand-new LongTermMemory instance (simulates a restart)")
    session_b = LongTermMemory("memory_store.json")
    print("Facts recalled without ever talking to session A directly:")
    print(session_b.as_context_string())

    print("\nFiltering recall by tag='preference':")
    print(session_b.as_context_string(tag="preference"))


def demo_combined() -> None:
    divider("HOW AN AGENT COMBINES BOTH")
    print(
        "On every turn, a memory-aware agent typically does:\n"
        "  1. Load long-term facts from disk -> inject into the system prompt.\n"
        "  2. Append the new user message to the short-term buffer.\n"
        "  3. Send [system prompt + short-term messages] to the model.\n"
        "  4. If the reply contains something durable, write it to long-term\n"
        "     memory so future sessions (with an empty short-term buffer)\n"
        "     still have access to it.\n"
        "See agent_with_memory.py for a live version of this loop using the\n"
        "Anthropic API."
    )


if __name__ == "__main__":
    demo_short_term()
    demo_long_term()
    demo_combined()
    print("\nDone. Inspect memory_store.json to see what got persisted to disk.")
