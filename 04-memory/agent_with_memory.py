"""
agent_with_memory.py
---------------------
Wires short-term memory (this session's conversation) and long-term memory
(facts persisted across sessions) into an actual agent loop that calls the
Anthropic API.

How the two memory types combine on every turn:
  1. Long-term facts are loaded from disk and injected into the SYSTEM PROMPT.
  2. Short-term messages (this conversation only) are sent as the MESSAGE LIST.
  3. If the model's reply contains a line like `REMEMBER: <fact>`, the agent
     extracts it and writes it to long-term memory so it's available in
     future sessions too.

Requires ANTHROPIC_API_KEY to be set (see .env.example). If no key is
available, run `python demo.py` instead, which exercises the same memory
classes without needing API access.
"""

from __future__ import annotations
import os
import re
from dotenv import load_dotenv
import anthropic

from short_term_memory import ConversationBuffer
from long_term_memory import LongTermMemory

load_dotenv()

MODEL = "claude-sonnet-4-5"
REMEMBER_PATTERN = re.compile(r"REMEMBER:\s*(.+)")

SYSTEM_TEMPLATE = """You are a helpful assistant with persistent memory.

{long_term_context}

If the user shares a durable fact worth remembering across future
conversations (their name, preferences, ongoing projects, etc.), end your
reply with a new line formatted exactly as:
REMEMBER: <the fact, written as a short standalone sentence>
Only include this line when there is genuinely something new and durable to
save. Do not use it for information that only matters in this conversation.
"""


class MemoryAgent:
    def __init__(self, memory_path: str = "memory_store.json"):
        self.client = anthropic.Anthropic()
        self.short_term = ConversationBuffer()
        self.long_term = LongTermMemory(memory_path)

    def _system_prompt(self) -> str:
        return SYSTEM_TEMPLATE.format(
            long_term_context=self.long_term.as_context_string()
        )

    def send(self, user_message: str) -> str:
        self.short_term.add("user", user_message)

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=self._system_prompt(),
            messages=self.short_term.as_api_messages(),
        )
        reply_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        self.short_term.add("assistant", reply_text)
        self._extract_and_save_memories(reply_text)
        return reply_text

    def _extract_and_save_memories(self, reply_text: str) -> None:
        for match in REMEMBER_PATTERN.finditer(reply_text):
            fact = match.group(1).strip()
            self.long_term.remember(fact, tags=["auto"])
            print(f"  [long-term memory saved]: {fact}")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key,")
        print("or run `python demo.py` for an offline demo of the memory classes.")
        raise SystemExit(1)

    agent = MemoryAgent()
    print("Agent ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        reply = agent.send(user_input)
        print(f"Agent: {reply}\n")
