# 07 — Agentic RAG

## What it is

"Agentic RAG" contrasts with the fixed pipeline you build in
[`/06-rag`](../06-rag): question → embed → retrieve top-k → stuff into
prompt → answer, every single time, no matter what.

Here, the **agent** — not a hardcoded pipeline — decides:

- **Whether** to retrieve at all (some questions don't need the knowledge
  base; some are answerable from general knowledge or don't need facts).
- **What** to search for (the agent rewrites the user's question into its
  own focused query, in the vocabulary a document would actually use).
- **When to stop or search again** — if the first result doesn't fully
  answer a multi-part question, the agent issues a second, differently
  worded query (multi-hop retrieval) instead of answering from partial
  context.
- **When it doesn't know** — if nothing relevant turns up, the agent says
  so instead of guessing.

This is implemented with Claude's native tool-use API: the knowledge-base
search function is exposed to the model as a tool called
`search_knowledge_base`. The model calls it zero or more times in a loop
until it's ready to produce a final text answer.

## Why it matters

Fixed RAG pipelines retrieve on every query, whether or not it helps, and
retrieve only once, whether or not the first pass was enough. In practice
that means:

- Wasted latency/cost retrieving for questions that didn't need it.
- Wrong or incomplete answers on multi-part questions that span more than
  one document, because the fixed top-k retrieval never saw the second
  document.
- No graceful "I don't know" — a fixed pipeline still stuffs whatever
  mediocre chunks it found into the prompt and asks the LLM to answer
  anyway.

Agentic retrieval is table stakes in production RAG systems (and shows up
constantly in AI engineer job descriptions) because it fixes exactly these
failure modes by letting the model reason about its own information needs.

## What I built

A small CLI app, **Acme Corp's internal policy assistant**, backed by four
fictional HR/security policy documents (`data/docs/`):

- `remote_work_policy.txt`
- `expense_reimbursement_policy.txt`
- `vacation_policy.txt`
- `security_policy.txt`

The agent loop lives in [`src/agent.py`](src/agent.py):

1. The user's question is sent to Claude along with the `search_knowledge_base`
   tool definition and a system prompt telling the model it may search zero,
   one, or several times before answering.
2. If Claude returns a `tool_use` block, the app runs the retriever
   ([`src/retriever.py`](src/retriever.py)) and feeds the results back as a
   `tool_result`.
3. This repeats — Claude can decide to search again with a refined query —
   until Claude responds with plain text instead of a tool call, which is
   treated as the final answer.
4. Every thought / action / observation is printed, so you can watch the
   agent's retrieval decisions in real time (a mini ReAct trace — see
   [`/03-react-agent`](../03-react-agent)).

The retriever itself is intentionally simple: TF-IDF + cosine similarity
over paragraph-level chunks (scikit-learn), not a vector database or
embeddings API. The point of this module is the agent's *decision-making*
about retrieval, not the retrieval quality itself — the vector-DB/embeddings
upgrade is its own module ([`/11-vector-databases`](../11-vector-databases)),
and `KnowledgeBase.search()` is written as a drop-in interface so it can be
swapped later without touching `agent.py`.

### Sample questions used in the demo (`main.py`)

| Question | What it's testing |
|---|---|
| "How much is the home office equipment stipend?" | Single-hop: one search, one document |
| "If I work remotely from another country for a month, what security steps and HR approvals do I need?" | Multi-hop: spans the remote work AND security policy docs |
| "What's a good icebreaker question for a new remote hire's first call?" | No retrieval needed — general knowledge |
| "What is Acme's parental leave policy?" | Not in the knowledge base — agent should say so, not hallucinate |

## Tech stack

- **Python 3.10+**
- **Anthropic API** (`anthropic` SDK) — tool-use / function-calling loop
- **scikit-learn** — TF-IDF vectorizer + cosine similarity for retrieval
- **pytest** — a few offline tests for the retriever that don't need an API key

## How to run

```bash
cd 07-agentic-rag
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY

# run the built-in sample questions
python main.py

# ask a single question
python main.py "How many vacation days do I get in year 4?"

# interactive chat
python main.py --chat

# run the offline retriever tests (no API key needed)
python -m pytest tests/ -v
```

## What I learned

- **Tool-calling is the mechanism, agentic behavior is the prompt +
  loop design.** The Anthropic API doesn't have an "agentic RAG mode" —
  it's a plain tool-use loop. What makes it *agentic* is (a) giving the
  model an explicit choice not to call the tool, and (b) letting the loop
  run for multiple turns so the model can decide, after seeing a result,
  whether it needs to search again.
- **Multi-hop questions expose the weakness of fixed top-k retrieval
  immediately.** The "remote work abroad" question needs both the Remote
  Work and Security policies; a single fixed retrieval call tuned for one
  document type would likely miss half the answer. Watching the agent
  independently decide to fire a second, differently-worded query for the
  second half of the question made the value of agentic retrieval concrete
  rather than theoretical.
- **The system prompt does a lot of the "agentic" work.** Explicitly telling
  the model it's allowed to answer without searching, and that it should
  search again if results are incomplete, mattered more than any code
  change — without that instruction the model defaulted to "search once,
  answer" almost like a fixed pipeline anyway.
- **A cheap retriever is fine for this exercise.** TF-IDF has no idea that
  "PTO" means "vacation," but because the agent can *see* a weak result and
  choose to rephrase its query, some of that weakness gets compensated for
  by the agent's own reasoning — which wouldn't happen in a fixed pipeline.
- **Grounding still needs to be enforced in the prompt, not assumed.**
  Early testing without the "never state a number you didn't get from a
  search result" instruction produced confident-sounding but made-up
  thresholds. The instruction fixed it, which was a good reminder that
  tool access alone doesn't stop hallucination — the model has to be told
  it isn't allowed to fall back on its own guess.

## Possible extensions

- Add a `search_knowledge_base` result-quality check so the agent can
  explicitly reflect "was this good enough?" before deciding to search
  again (ties into [`/13-reflection-self-critique`](../13-reflection-self-critique)).
- Swap the TF-IDF retriever for real embeddings + a vector DB
  ([`/11-vector-databases`](../11-vector-databases)).
- Add a second tool (e.g. a calculator, or a "check current date" tool) so
  the agent is choosing between multiple tools, not just deciding
  retrieve-or-not.
- Log every trace to a file for later inspection
  ([`/15-observability-evaluation`](../15-observability-evaluation)).
