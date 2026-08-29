# 06 · RAG (Retrieval-Augmented Generation)

## What it is
RAG grounds an LLM's answers in an external set of documents instead of
relying purely on what the model memorized during training. Instead of
sending an entire knowledge base to the model on every request (which won't
fit in a context window and is expensive), a RAG pipeline:

1. **Chunks** documents into small, retrievable pieces.
2. **Indexes** those chunks so they can be searched by similarity.
3. **Retrieves** the top-k most relevant chunks for a given question.
4. **Generates** an answer by giving the model the question plus only the
   retrieved chunks as context.

## Why it matters
It's one of the most commonly asked-about patterns in AI engineer interviews
and job postings, because it's the cheapest way to make an LLM answer
questions about private or fast-changing data (internal docs, a codebase,
today's news) without fine-tuning. It also directly sets up **07-agentic-rag**,
where the agent decides *when* and *what* to retrieve rather than always
retrieving on a fixed schedule.

## What I built
A minimal, dependency-light RAG pipeline over a small corpus of `.txt` notes
(reused from other modules in this repo, so the corpus is genuinely about
agentic AI concepts):

```
06-rag/
├── README.md
├── requirements.txt
├── demo.py                 # CLI: ask questions from the terminal
├── sample_docs/            # the corpus (3 short .txt files)
├── rag/
│   ├── chunking.py          # word-based chunking with overlap
│   ├── retriever.py         # TfidfRetriever (default) + DenseRetriever (optional)
│   ├── generator.py         # calls Claude, or falls back to a mock if no API key
│   └── pipeline.py          # RAGPipeline: load -> chunk -> index -> retrieve -> generate
└── tests/
    └── test_rag.py
```

**Retrieval** defaults to TF-IDF + cosine similarity (`TfidfRetriever`) —
pure scikit-learn, no model download, runs offline in under a second. There's
also a `DenseRetriever` that uses `sentence-transformers` embeddings for
semantic (meaning-based, not just keyword) matching — same `.index()` /
`.retrieve()` interface, so swapping one for the other is a one-line change
in `pipeline.py`. This mirrors how LangChain/LlamaIndex abstract "vector
store" implementations behind a shared interface.

**Generation** calls the real Claude API (`claude-sonnet-5`) if
`ANTHROPIC_API_KEY` is set in the environment, using a system prompt that
instructs the model to answer only from the provided context and say "I
don't know" otherwise (basic hallucination mitigation). If no key is set, it
runs in a mock mode that prints exactly the context block that *would* have
been sent — useful for inspecting retrieval quality independent of
generation, and means the whole demo runs with zero setup.

### Try it
```bash
pip install -r requirements.txt
python demo.py "What is the ReAct pattern?"

# or interactively:
python demo.py
```

To get real generated answers instead of mock mode:
```bash
export ANTHROPIC_API_KEY=your-key-here
python demo.py "How do guardrails relate to human-in-the-loop confirmation?"
```

## Tech stack
- Python
- scikit-learn (TF-IDF + cosine similarity)
- anthropic SDK (generation)
- sentence-transformers (optional, for dense/embedding-based retrieval)

## What I learned
- **Chunking strategy matters more than it looks.** Word-count chunking is
  simple but can split a sentence in half right at a chunk boundary — the
  `overlap` parameter exists specifically to reduce how often that loses
  information the retriever needed.
- **TF-IDF retrieval only matches on shared vocabulary.** It found the
  guardrails chunk fine for a query using the word "guardrails," but a
  query phrased entirely differently (e.g. "how do you stop an agent from
  doing something irreversible without permission") would score much lower
  under TF-IDF than under a dense/embedding retriever, which can match on
  meaning rather than exact words. That's the main practical argument for
  `DenseRetriever` in production.
- **Grounding only works if the prompt enforces it.** Without the system
  prompt explicitly telling the model to answer only from context and admit
  uncertainty, an LLM will happily blend in outside knowledge, which quietly
  defeats the point of RAG.
- **Retrieval quality is easy to inspect, generation quality isn't.** Because
  `pipeline.retrieve()` and `pipeline.ask()` are separate methods, you can
  debug bad answers by first checking whether the *right chunks* were even
  retrieved, before assuming the LLM reasoned badly over correct context.

## Status
🟨 In progress — next: try `DenseRetriever` on a harder query set where TF-IDF
visibly fails, to make the sparse-vs-dense trade-off concrete.
