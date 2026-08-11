# 04 — Agent Memory

Short-term (context) memory vs. long-term (persistent) memory, and how an
agent combines both on every turn.

## What it is

LLMs are stateless — every API call only "knows" what's in the messages you
send it. "Memory" in an agent is really two different engineering patterns
bolted on top of that statelessness:

- **Short-term / working memory** — the running conversation (or scratchpad)
  passed in the `messages` array on every call. It exists only for the
  lifetime of the session/process and is bounded by the context window. Once
  the window fills up or the process ends, it's gone.
- **Long-term / persistent memory** — facts written to durable storage (a
  file, database, or vector store) that can be recalled in a completely
  separate, later session and re-injected into the prompt (typically the
  system prompt).

## Why it matters

Almost every real agent needs both. Short-term memory alone means the agent
forgets the user the moment the session ends. Long-term memory alone (with
no short-term buffer) means the agent can't hold a coherent multi-turn
conversation. Interviewers and job specs call this out explicitly because
it's one of the first things that breaks when someone moves from a toy
chatbot demo to a "real" agent.

## What I built

- `short_term_memory.py` — a `ConversationBuffer` (keeps every turn) and a
  `WindowedConversationBuffer` (caps turns, dropping the oldest — simulating
  a context window filling up).
- `long_term_memory.py` — a `LongTermMemory` store backed by a JSON file.
  Facts written in one process are readable by a brand-new instance in
  another process, demonstrating persistence across sessions.
- `agent_with_memory.py` — a live agent using the Anthropic API that loads
  long-term facts into the system prompt, keeps a short-term buffer for the
  current conversation, and auto-saves new durable facts the model flags
  with a `REMEMBER:` line.
- `demo.py` — a scripted, **offline** walkthrough of all of the above with
  no API key required.

## Tech stack

- Python 3.10+
- `anthropic` Python SDK (for the live agent only)
- Plain JSON file for persistence (kept intentionally simple — swap for a
  real DB or vector store later; see `06-rag` / `11-vector-databases`)

## How to run it

```bash
cd 04-memory
pip install -r requirements.txt

# No API key needed:
python demo.py

# Live agent (needs an API key):
cp .env.example .env   # then add your ANTHROPIC_API_KEY
python agent_with_memory.py
```

Running `demo.py` (or the live agent) will create `memory_store.json` in
this folder — delete it any time to reset long-term memory.

## What I learned

- Short-term memory is just "the message list you resend" — there's no
  magic, which is also why it's fragile: it's bounded by the context window
  and vanishes with the session.
- Long-term memory needs an explicit **write policy** (what's worth saving
  and when) and an explicit **read/injection policy** (where recalled facts
  get placed in the prompt — here, the system prompt). Getting the write
  policy wrong is how agents end up either forgetting everything or hoarding
  irrelevant noise.
- Recall here is exact/tag-based, not semantic. Real long-term memory at
  scale usually needs embeddings + similarity search to find the *relevant*
  subset of memories instead of dumping everything into the prompt — that's
  the natural next step covered in `06-rag` and `11-vector-databases`.
