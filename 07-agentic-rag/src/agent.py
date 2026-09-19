"""
AgenticRAG: the agent decides WHEN, WHAT, and HOW OFTEN to retrieve.

This is the core difference from "plain" RAG (see /06-rag):

  Plain RAG        : question -> always retrieve top_k -> stuff into prompt -> answer
                      (retrieval happens on a fixed schedule, once, no matter what)

  Agentic RAG (here): question -> LLM decides:
                        - "Do I even need to search? Maybe I can answer directly."
                        - "What should I search for?" (it writes its own query,
                          which may differ a lot from the user's original wording)
                        - "Was that result good enough, or do I need another,
                          differently-worded search?" (multi-hop retrieval)
                        - "Am I ready to answer now?"

We implement this with Anthropic's tool-use API: the knowledge base search
function is exposed to the model as a tool. The model calls it zero or more
times, in a loop, until it produces a final text answer instead of a tool
call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import anthropic

from .config import MAX_AGENT_TURNS, MODEL_NAME
from .retriever import KnowledgeBase

SYSTEM_PROMPT = """\
You are Acme Corp's internal policy assistant.

You have access to a `search_knowledge_base` tool that searches Acme's HR \
and security policy documents. You do NOT have these policies memorized — \
anything you say about specific numbers, thresholds, or procedures MUST come \
from a search result, never from assumption.

Think like an agent, not a fixed pipeline:
- If a question is general chit-chat or doesn't require company-specific \
facts, you may answer directly without searching.
- If a question requires a policy fact, call `search_knowledge_base` with a \
concise, well-formed query (it does not need to reuse the user's exact \
wording — rephrase it into the terms a policy document would use).
- If the first search result doesn't fully answer the question, or the \
question has multiple parts touching different policies (e.g. remote work \
AND expenses), call the tool again with a different, more targeted query. \
You may search up to a few times before answering.
- If the knowledge base has nothing relevant after searching, say so plainly \
instead of guessing.
- Once you have enough information, answer the user directly in plain text \
(do not call the tool again), and briefly cite which policy document(s) you \
used.
"""

SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "Search Acme Corp's internal policy documents (remote work, "
        "expenses, vacation/PTO, security) and return the most relevant "
        "passages with their source document."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A focused search query in the terms a policy document "
                    "would use, e.g. 'home office equipment stipend amount'."
                ),
            }
        },
        "required": ["query"],
    },
}


@dataclass
class TraceStep:
    """One step of the agent's reasoning, for printing/inspection."""

    kind: str  # "thought" | "action" | "observation" | "answer"
    content: str


@dataclass
class AgentResult:
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    num_searches: int = 0


class AgenticRAG:
    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self.kb = knowledge_base or KnowledgeBase()
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def _run_tool(self, tool_name: str, tool_input: dict) -> str:
        if tool_name != "search_knowledge_base":
            return json.dumps({"error": f"Unknown tool '{tool_name}'"})
        query = tool_input.get("query", "")
        results = self.kb.search(query)
        if not results:
            return json.dumps(
                {"query": query, "results": [], "note": "No relevant passages found."}
            )
        return json.dumps({"query": query, "results": results})

    def ask(self, question: str, verbose: bool = True) -> AgentResult:
        trace: list[TraceStep] = []
        num_searches = 0

        messages = [{"role": "user", "content": question}]

        for turn in range(MAX_AGENT_TURNS):
            response = self.client.messages.create(
                model=MODEL_NAME,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=[SEARCH_TOOL],
                messages=messages,
            )

            # Record any reasoning text the model produced alongside tool calls
            text_parts = [b.text for b in response.content if b.type == "text"]
            if text_parts:
                thought = "\n".join(text_parts).strip()
                if thought:
                    trace.append(TraceStep("thought", thought))
                    if verbose:
                        print(f"\n💭 Thought: {thought}")

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason != "tool_use" or not tool_use_blocks:
                # The model chose to answer directly instead of calling the tool.
                final_answer = "\n".join(text_parts).strip() or "(no answer produced)"
                trace.append(TraceStep("answer", final_answer))
                if verbose:
                    print(f"\n✅ Answer: {final_answer}")
                return AgentResult(answer=final_answer, trace=trace, num_searches=num_searches)

            # Append the assistant turn (including tool_use blocks) to history
            messages.append({"role": "assistant", "content": response.content})

            # Execute each requested tool call and feed results back
            tool_results = []
            for block in tool_use_blocks:
                num_searches += 1
                action_desc = f"search_knowledge_base(query={block.input.get('query')!r})"
                trace.append(TraceStep("action", action_desc))
                if verbose:
                    print(f"🔎 Action: {action_desc}")

                observation = self._run_tool(block.name, block.input)
                trace.append(TraceStep("observation", observation))
                if verbose:
                    print(f"📄 Observation: {observation}")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": observation,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        # Safety valve: forced final answer if MAX_AGENT_TURNS was hit
        forced_note = (
            "(Reached the maximum number of reasoning steps before reaching "
            "a confident final answer.)"
        )
        trace.append(TraceStep("answer", forced_note))
        return AgentResult(answer=forced_note, trace=trace, num_searches=num_searches)
