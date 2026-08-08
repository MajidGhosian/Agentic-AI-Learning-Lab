# 03 — ReAct Pattern

## What it is

**ReAct** (Reasoning + Acting) is a prompting pattern introduced by
[Yao et al., 2022](https://arxiv.org/abs/2210.03629) in which a language
model interleaves three kinds of steps in a loop, instead of jumping
straight to an answer:

```
Thought:      the model reasons about what to do next
Action:       the model picks a tool + input to call
Observation:  the result of that tool call is fed back into the model
```

This repeats — Thought → Action → Observation → Thought → ... — until the
model has enough information to produce a `Final Answer`. It's the
foundation that most modern "agent" frameworks (LangGraph, CrewAI,
AutoGen, etc.) build their agent loops on top of.

## Why it matters

Two failure modes ReAct addresses:

- **Reasoning-only** (e.g. plain chain-of-thought) models can reason
  fluently but have no way to check their reasoning against the real
  world — they'll happily "calculate" a wrong number with total
  confidence.
- **Acting-only** models call tools but don't explain *why*, which makes
  their behavior hard to debug, steer, or trust.

ReAct fixes both by forcing an explicit, inspectable trace of *why* an
action was taken and *what* actually happened as a result. This is the
core building block behind agentic AI systems, and it shows up
constantly in AI/LLM engineer job descriptions.

## What I built

A minimal, dependency-light ReAct agent implemented **without** relying
on a provider's native tool-calling API — the Thought/Action/Action
Input/Observation loop is implemented explicitly by prompting the model
for that exact text format, parsing it with a small regex-based parser,
and executing matching Python functions. This keeps every part of the
loop visible instead of hidden behind an abstraction.

```
03-react-agent/
├── react_agent.py   ← the ReAct loop (ReActAgent class)
├── tools.py         ← calculator, word_length, search (mock KB)
├── demo.ipynb        ← runnable walkthrough with two example questions
├── requirements.txt
└── README.md
```

**Available tools:**

| Tool | Description |
|---|---|
| `calculator` | evaluates a restricted arithmetic expression (safe AST eval, no `eval()`) |
| `word_length` | counts letters in a word |
| `search` | looks up a fact in a small mock knowledge base (stand-in for a real search API) |

**Example run:**

```
$ python react_agent.py "What is 12% of 850, plus the number of letters in the word 'agentic'?"

--- Step 1 ---
Thought: I need to calculate 12% of 850 first.
Action: calculator[850 * 0.12]
Observation: 102.0

--- Step 2 ---
Thought: Now I need the number of letters in 'agentic'.
Action: word_length[agentic]
Observation: 7

--- Step 3 ---
Thought: 102 + 7 = 109. I have everything I need.
Final Answer: 109
```

## Tech stack

- Python 3.11
- [Anthropic API](https://docs.claude.com) (`claude-sonnet-4-6`)
- No agent framework — the loop is hand-rolled to make the pattern explicit

## How to run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
python react_agent.py "your question here"

# or open demo.ipynb for a walkthrough with two chained examples
```

## What I learned

- The hardest part of a text-based ReAct loop isn't the reasoning — it's
  **reliably parsing the model's free-text output** into a structured
  step. A stricter format (or falling back to native tool-calling) removes
  a lot of this fragility in production systems.
- Feeding the observation back as a new turn (rather than editing the
  same message) keeps the running transcript honest and lets the model
  "see" its own prior reasoning on the next call.
- A `max_steps` safety cap is essential — without one, a confused agent
  can loop indefinitely re-trying the same failing action.
- This hand-rolled version makes it obvious *why* frameworks like
  LangGraph model agents as explicit state graphs: once you add retries,
  parallel tool calls, and branching, a plain `for` loop stops being
  enough.

## Status

✅ Done — see `/03-react-agent` linked from the [main roadmap](../README.md).
