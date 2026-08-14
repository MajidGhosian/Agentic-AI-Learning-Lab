# 🤖 Agentic AI Learning Lab

A hands-on journey through the core concepts of **Agentic AI** — the skills and terms that show up again and again in job descriptions for AI Engineer, LLM Engineer, and Applied AI roles. Each concept below gets its own small, self-contained project so I can *build* my understanding instead of just reading about it.

> 📌 **Goal:** One focused mini-project per concept → clear README → link it back here.
> Each project lives in its own folder (or repo) and is linked from the table once it's done.

---

## 🧭 How this repo is organized

```
agentic-ai-learning-lab/
├── README.md                  ← you are here (the index)
├── 01-prompt-engineering/
├── 02-function-calling/
├── 03-react-agent/
├── 04-memory/
├── 05-planning-task-decomposition/
├── 06-rag/
├── 07-agentic-rag/
├── 08-multi-agent-systems/
├── 09-agent-frameworks/
├── 10-mcp-tool-integration/
├── 11-vector-databases/
├── 12-human-in-the-loop/
├── 13-reflection-self-critique/
├── 14-guardrails-safety/
├── 15-observability-evaluation/
└── 16-workflow-orchestration/
```

Each sub-folder will contain its own `README.md` with:
- **What it is** (short explanation)
- **Why it matters** (why employers care)
- **What I built**
- **Tech stack**
- **What I learned**

---

## 📚 Concept Roadmap & Projects

### Foundations

| # | Concept | What it covers | Status | Project |
|---|---------|----------------|--------|---------|
| 01 | **Prompt Engineering** | Zero/few-shot prompting, system prompts, chain-of-thought | 🟨 In progress | [`/01-prompt-engineering`](./01-prompt-engineering) |
| 02 | **Function / Tool Calling** | Letting an LLM call external functions/APIs with structured outputs | 🟨 In progress | [`/02-function-calling`](./02-function-calling) |
| 03 | **ReAct Pattern** | Reasoning + Acting loop (thought → action → observation) | 🟨 In progress | [`/03-react-agent`](./03-react-agent) |
| 04 | **Agent Memory** | Short-term (context) vs long-term (persistent) memory | 🟨 In progress | [`/04-memory`](./04-memory) |
| 05 | **Planning & Task Decomposition** | Breaking a goal into sub-tasks, task queues | 🟨 In progress | [`/05-planning-task-decomposition`](./05-planning-task-decomposition) |

### Knowledge & Retrieval

| # | Concept | What it covers | Status | Project |
|---|---------|----------------|--------|---------|
| 06 | **RAG (Retrieval-Augmented Generation)** | Grounding LLM answers in external documents | ⬜ Not started | [`/06-rag`](./06-rag) |
| 07 | **Agentic RAG** | Agent decides *when/what/how* to retrieve, not just a fixed pipeline | ⬜ Not started | [`/07-agentic-rag`](./07-agentic-rag) |
| 11 | **Vector Databases & Embeddings** | Semantic search, chunking strategies, similarity metrics | ⬜ Not started | [`/11-vector-databases`](./11-vector-databases) |

### Multi-Agent & Orchestration

| # | Concept | What it covers | Status | Project |
|---|---------|----------------|--------|---------|
| 08 | **Multi-Agent Systems** | Specialized agents collaborating (planner, researcher, critic, etc.) | ⬜ Not started | [`/08-multi-agent-systems`](./08-multi-agent-systems) |
| 09 | **Agent Frameworks** | Hands-on with LangGraph / CrewAI / AutoGen (pick one or compare) | ⬜ Not started | [`/09-agent-frameworks`](./09-agent-frameworks) |
| 10 | **MCP / Tool Integration** | Model Context Protocol — standardized way to plug tools/data into agents | ⬜ Not started | [`/10-mcp-tool-integration`](./10-mcp-tool-integration) |
| 16 | **Workflow Orchestration** | State machines / DAGs for multi-step agent workflows | ⬜ Not started | [`/16-workflow-orchestration`](./16-workflow-orchestration) |

### Reliability & Production Readiness

| # | Concept | What it covers | Status | Project |
|---|---------|----------------|--------|---------|
| 12 | **Human-in-the-Loop** | Approval steps, escalation, interrupting agents for confirmation | ⬜ Not started | [`/12-human-in-the-loop`](./12-human-in-the-loop) |
| 13 | **Reflection & Self-Critique** | Agent reviewing/improving its own output before finishing | ⬜ Not started | [`/13-reflection-self-critique`](./13-reflection-self-critique) |
| 14 | **Guardrails & Safety** | Input/output validation, prompt-injection defense, content filters | ⬜ Not started | [`/14-guardrails-safety`](./14-guardrails-safety) |
| 15 | **Observability & Evaluation** | Tracing agent runs, logging, eval metrics, benchmarking | ⬜ Not started | [`/15-observability-evaluation`](./15-observability-evaluation) |

**Status legend:** ⬜ Not started · 🟨 In progress · ✅ Done

---

## 🛠️ Suggested Tech Stack (flexible — will evolve as I learn)

- **Language:** Python
- **LLM Access:** Anthropic API / OpenAI API (swap freely between projects)
- **Agent Frameworks:** LangChain, LangGraph, CrewAI, AutoGen (tried across different projects for comparison)
- **Vector DB:** Chroma / FAISS / Pinecone
- **Tracing/Eval:** LangSmith / simple custom logging
- **Deployment (optional, later):** FastAPI + Docker

---

## 🎯 Why these specific concepts

These are pulled from patterns that repeatedly show up in **agentic AI / AI engineer job postings**: tool use, RAG, multi-agent orchestration, memory, planning, guardrails, and observability. The idea is that after finishing this roadmap, I'll have a small working example for nearly every buzzword that shows up in that kind of posting — and can point to real code instead of just saying "I know about agents."

---

## 🔗 Connect

Feel free to explore the individual project folders above — each one is meant to be small enough to read in a few minutes but complete enough to actually demonstrate the concept in action.
