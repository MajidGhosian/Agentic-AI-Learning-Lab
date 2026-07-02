🤖 Agentic AI Learning Lab

A hands-on journey through the core concepts of Agentic AI — the skills and terms that show up again and again in job descriptions for AI Engineer, LLM Engineer, and Applied AI roles. Each concept below gets its own small, self-contained project so I can build my understanding instead of just reading about it.


📌 Goal: One focused mini-project per concept → clear README → link it back here.
Each project lives in its own folder (or repo) and is linked from the table once it's done.




🧭 How this repo is organized

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

Each sub-folder will contain its own README.md with:


What it is (short explanation)
Why it matters (why employers care)
What I built
Tech stack
What I learned



📚 Concept Roadmap & Projects

Foundations

#ConceptWhat it coversStatusProject01Prompt EngineeringZero/few-shot prompting, system prompts, chain-of-thought⬜ Not started/01-prompt-engineering02Function / Tool CallingLetting an LLM call external functions/APIs with structured outputs⬜ Not started/02-function-calling03ReAct PatternReasoning + Acting loop (thought → action → observation)⬜ Not started/03-react-agent04Agent MemoryShort-term (context) vs long-term (persistent) memory⬜ Not started/04-memory05Planning & Task DecompositionBreaking a goal into sub-tasks, task queues⬜ Not started/05-planning-task-decomposition

Knowledge & Retrieval

#ConceptWhat it coversStatusProject06RAG (Retrieval-Augmented Generation)Grounding LLM answers in external documents⬜ Not started/06-rag07Agentic RAGAgent decides when/what/how to retrieve, not just a fixed pipeline⬜ Not started/07-agentic-rag11Vector Databases & EmbeddingsSemantic search, chunking strategies, similarity metrics⬜ Not started/11-vector-databases

Multi-Agent & Orchestration

#ConceptWhat it coversStatusProject08Multi-Agent SystemsSpecialized agents collaborating (planner, researcher, critic, etc.)⬜ Not started/08-multi-agent-systems09Agent FrameworksHands-on with LangGraph / CrewAI / AutoGen (pick one or compare)⬜ Not started/09-agent-frameworks10MCP / Tool IntegrationModel Context Protocol — standardized way to plug tools/data into agents⬜ Not started/10-mcp-tool-integration16Workflow OrchestrationState machines / DAGs for multi-step agent workflows⬜ Not started/16-workflow-orchestration

Reliability & Production Readiness

#ConceptWhat it coversStatusProject12Human-in-the-LoopApproval steps, escalation, interrupting agents for confirmation⬜ Not started/12-human-in-the-loop13Reflection & Self-CritiqueAgent reviewing/improving its own output before finishing⬜ Not started/13-reflection-self-critique14Guardrails & SafetyInput/output validation, prompt-injection defense, content filters⬜ Not started/14-guardrails-safety15Observability & EvaluationTracing agent runs, logging, eval metrics, benchmarking⬜ Not started/15-observability-evaluation

Status legend: ⬜ Not started · 🟨 In progress · ✅ Done


🛠️ Suggested Tech Stack (flexible — will evolve as I learn)


Language: Python
LLM Access: Anthropic API / OpenAI API (swap freely between projects)
Agent Frameworks: LangChain, LangGraph, CrewAI, AutoGen (tried across different projects for comparison)
Vector DB: Chroma / FAISS / Pinecone
Tracing/Eval: LangSmith / simple custom logging
Deployment (optional, later): FastAPI + Docker



🎯 Why these specific concepts

These are pulled from patterns that repeatedly show up in agentic AI / AI engineer job postings: tool use, RAG, multi-agent orchestration, memory, planning, guardrails, and observability. The idea is that after finishing this roadmap, I'll have a small working example for nearly every buzzword that shows up in that kind of posting — and can point to real code instead of just saying "I know about agents."


📈 Progress Log

DateUpdate(add entries as projects are completed)


🔗 Connect

Feel free to explore the individual project folders above — each one is meant to be small enough to read in a few minutes but complete enough to actually demonstrate the concept in action.
