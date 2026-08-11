"""
long_term_memory.py
--------------------
Demonstrates LONG-TERM (persistent) memory.

Unlike short-term memory, this survives across process restarts because it's
written to disk (here, a simple JSON file — swap this for a database or
vector store in a real system without changing the interface). The agent can
save durable facts about the user/task and recall them in a future,
completely separate session.

This is intentionally simple (no embeddings) so the mechanics of "write once,
read in a later session" are easy to see. 06-rag / 11-vector-databases will
cover semantic recall over long-term memory later in the roadmap.
"""

from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MemoryRecord:
    id: str
    text: str
    tags: list[str]
    created_at: float


class LongTermMemory:
    """A minimal persistent key-fact store backed by a JSON file on disk."""

    def __init__(self, path: str | Path = "memory_store.json"):
        self.path = Path(path)
        if not self.path.exists():
            self._write([])

    # -- persistence helpers -------------------------------------------------
    def _read(self) -> list[dict]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, records: list[dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    # -- public API ------------------------------------------------------------
    def remember(self, text: str, tags: list[str] | None = None) -> MemoryRecord:
        """Persist a new fact. This survives after the program exits."""
        record = MemoryRecord(
            id=str(uuid.uuid4())[:8],
            text=text,
            tags=tags or [],
            created_at=time.time(),
        )
        records = self._read()
        records.append(asdict(record))
        self._write(records)
        return record

    def recall(self, tag: str | None = None) -> list[dict]:
        """Load facts back from disk, optionally filtered by tag."""
        records = self._read()
        if tag:
            records = [r for r in records if tag in r["tags"]]
        return records

    def forget(self, record_id: str) -> bool:
        records = self._read()
        remaining = [r for r in records if r["id"] != record_id]
        removed = len(remaining) != len(records)
        self._write(remaining)
        return removed

    def as_context_string(self, tag: str | None = None) -> str:
        """
        Format recalled memories as a block of text you can inject into a
        system prompt — this is how long-term memory gets fed back into an
        LLM call that otherwise has no built-in memory of its own.
        """
        records = self.recall(tag=tag)
        if not records:
            return "No prior memories."
        lines = [f"- {r['text']} (tags: {', '.join(r['tags']) or 'none'})" for r in records]
        return "Known facts about the user:\n" + "\n".join(lines)


if __name__ == "__main__":
    store = LongTermMemory("memory_store.json")

    print("=== Session 1: agent learns something and saves it ===")
    rec = store.remember("The user's name is Majid and they are building an Agentic AI portfolio.", tags=["identity"])
    store.remember("The user prefers Python over JavaScript.", tags=["preference"])
    print(f"Saved record {rec.id}: {rec.text}")

    print("\n=== Session 2 (simulated restart): agent recalls facts from disk ===")
    fresh_store = LongTermMemory("memory_store.json")  # new instance = new 'session'
    print(fresh_store.as_context_string())

    print("\n=== Filtering by tag ===")
    print(fresh_store.as_context_string(tag="preference"))
