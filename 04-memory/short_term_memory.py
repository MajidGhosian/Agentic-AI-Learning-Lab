"""
short_term_memory.py
---------------------
Demonstrates SHORT-TERM (context / working) memory.

Short-term memory is simply the running list of messages passed back to the
model on every call. It only lives as long as the process / session does —
once the object is destroyed (or the window is trimmed), the memory is gone.

Two things are shown here:
  1. A basic ConversationBuffer that just appends every turn.
  2. A WindowedConversationBuffer that caps how many turns are kept, which
     mimics what happens when a context window fills up and older turns
     have to be dropped (a real constraint agents run into).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass
class ConversationBuffer:
    """Naive short-term memory: keeps every message, in order."""

    messages: list[dict] = field(default_factory=list)

    def add(self, role: Role, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def as_api_messages(self) -> list[dict]:
        """Return messages in the shape the Anthropic API expects."""
        return list(self.messages)

    def clear(self) -> None:
        """Simulates the memory vanishing when the session ends."""
        self.messages.clear()

    def __len__(self) -> int:
        return len(self.messages)


@dataclass
class WindowedConversationBuffer(ConversationBuffer):
    """
    Short-term memory bounded by a max number of turns (a 'turn' = one
    user message + one assistant reply). Once the window is exceeded,
    the oldest turn is dropped — approximating a full context window.
    """

    max_turns: int = 5

    def add(self, role: Role, content: str) -> None:
        super().add(role, content)
        self._trim()

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            overflow = len(self.messages) - max_messages
            self.messages = self.messages[overflow:]


if __name__ == "__main__":
    print("=== Basic ConversationBuffer ===")
    buf = ConversationBuffer()
    buf.add("user", "My favorite color is teal.")
    buf.add("assistant", "Got it, teal it is!")
    buf.add("user", "What's my favorite color?")
    for m in buf.as_api_messages():
        print(f"  {m['role']:>9}: {m['content']}")

    print("\n=== WindowedConversationBuffer (max_turns=2) ===")
    wbuf = WindowedConversationBuffer(max_turns=2)
    conversation = [
        ("user", "Turn 1 question"),
        ("assistant", "Turn 1 answer"),
        ("user", "Turn 2 question"),
        ("assistant", "Turn 2 answer"),
        ("user", "Turn 3 question"),
        ("assistant", "Turn 3 answer"),
    ]
    for role, content in conversation:
        wbuf.add(role, content)
    print(f"Kept {len(wbuf)} messages (window drops the oldest turn):")
    for m in wbuf.as_api_messages():
        print(f"  {m['role']:>9}: {m['content']}")

    print("\n=== Memory disappears when session ends ===")
    wbuf.clear()
    print(f"Messages after clear(): {len(wbuf)}")
