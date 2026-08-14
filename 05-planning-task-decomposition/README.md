# 05 · Planning & Task Decomposition

Part of the [Agentic AI Learning Lab](../README.md).

## What it is

Breaking a high-level, fuzzy **goal** ("launch a portfolio website") into a
list of smaller, concrete **sub-tasks**, wiring up the **dependencies**
between them, and then working through a **task queue** that only runs a
task once everything it depends on has finished. This is the planning layer
that sits underneath most "autonomous agent" demos — it's what turns a
single user prompt into a multi-step execution plan instead of one giant
uncontrolled LLM call.

## Why it matters

Real-world goals are rarely a single action. Employers describing "agentic"
or "AI engineer" roles consistently expect familiarity with:

- Decomposing ambiguous objectives into ordered, actionable steps
- Representing dependencies between steps (task B can't start before task A)
- Queue/state management for multi-step execution
- Retry and failure-handling logic when a step fails
- Swapping between a rule-based/deterministic planner and an LLM-driven
  planner depending on cost, latency, and reliability needs

This mini-project builds a minimal but complete version of all of the above.

## What I built

A small, dependency-free Python package with three pieces:

| File | Purpose |
|---|---|
| `planner.py` | `Task`, `TaskQueue`, `RuleBasedDecomposer`, `LLMDecomposer`, and `Planner` — the reusable core. |
| `demo.py` | A runnable CLI demo that decomposes a goal, executes the resulting queue, and prints the log. |
| `example_output.txt` | Captured output from a sample run (see below). |

**Two decomposers, one interface:**
- `RuleBasedDecomposer` — deterministic, offline, no API key needed. Useful
  as a fallback and for testing the queue/execution mechanics in isolation.
- `LLMDecomposer` — calls the real Anthropic Messages API with a system
  prompt that forces structured JSON output (title / description /
  dependency index per sub-task), so an actual model does the semantic
  planning.

**The queue (`TaskQueue`)** tracks each task's status (`pending` → `ready`
→ `in_progress` → `done` / `failed` / `skipped`) and exposes
`next_runnable()`, which only returns a task once all of its dependencies
are `done` (or `skipped`).

**The executor (`Planner.run`)** pulls the next runnable task, executes it
via a pluggable `execute_fn`, retries on failure up to `max_attempts`, and
cascades a `skipped` status to anything permanently blocked by a failed
dependency — so the pipeline always terminates instead of hanging.

### Try it

```bash
# Offline mode (no API key required)
python demo.py "Launch a personal portfolio website"

# Real LLM-driven decomposition (needs ANTHROPIC_API_KEY in your environment)
export ANTHROPIC_API_KEY=sk-...
python demo.py "Plan a 3-day trip to Lisbon"
```

`demo.py` also wires in a deliberately flaky executor (the "Research" step
fails once) so you can see the retry logic fire in the log.

## Tech stack

- Python 3.10+ (standard library only — `dataclasses`, `enum`, `uuid`,
  `urllib.request`)
- Optional: Anthropic Messages API (`https://api.anthropic.com/v1/messages`)
  for real semantic decomposition — no SDK required, just a raw HTTPS call

## What I learned

- The hard part of "planning" isn't generating a list of steps — it's
  representing **dependencies** correctly and making sure the executor
  can't get stuck (deadlock) when a dependency fails.
- Separating the **decomposer** (turns a goal into tasks) from the
  **executor** (runs tasks) makes it trivial to swap a cheap/offline
  planner in for testing and a real LLM planner in for production, without
  touching the queue logic at all.
- Forcing an LLM to respond in strict JSON (and stripping markdown fences
  defensively) is a small but recurring bit of engineering every
  agent-with-structured-output project ends up needing.
- Retry + skip-on-blocked-dependency logic is a minimal but realistic taste
  of the failure-handling agent frameworks (LangGraph, CrewAI, etc.) build
  much more elaborate versions of.

## Possible extensions

- Parallel execution of independent tasks (anything with no unmet deps can
  run concurrently, not just sequentially)
- Re-planning: if a task fails permanently, ask the LLM to propose an
  alternate sub-plan instead of just skipping
- Persist the queue to disk/DB so a long-running plan survives a restart
