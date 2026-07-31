# 01 · Prompt Engineering — Support Ticket Triage

## What it is
Prompt engineering is the practice of designing the instructions given to an LLM — wording, structure, examples, and constraints — to reliably get the output you actually want.

## Why it matters
Every agentic system starts here. Before an LLM can call tools, retrieve documents, or coordinate with other agents, it needs a well-designed prompt just to behave consistently on a single task. Weak prompting shows up as: inconsistent formatting, missed edge cases, and unreliable structured output — all things that break downstream automation.

## What I built
A Jupyter notebook (`prompt_lab.ipynb`) that runs the **same task** — triaging a customer support ticket into urgency, category, and a one-line summary — through **five different prompting strategies**:

1. Zero-shot (bare prompt, no instructions)
2. Zero-shot + system prompt (role, categories, constraints defined)
3. Few-shot (worked examples included)
4. Chain-of-thought (explicit step-by-step reasoning before the answer)
5. Structured output (forced JSON schema, parsed immediately)

All five strategies are run against the same 5 sample tickets, results are saved to `prompt_lab_results.json`, and the notebook ends with a side-by-side comparison plus a written evaluation of what worked and what didn't.

## Tech stack
- Python
- Jupyter Notebook
- Anthropic API (`anthropic` Python SDK)

## How to run it
1. `pip install anthropic jupyter`
2. Set your API key: `export ANTHROPIC_API_KEY="your-key-here"`
3. `jupyter notebook prompt_lab.ipynb`
4. Run all cells top to bottom, then fill in the **Evaluation** section at the bottom with your own observations.

## What I learned
_(Fill this in after running the notebook — 2-4 sentences on which prompting strategy performed best/worst and why, plus anything surprising.)_
