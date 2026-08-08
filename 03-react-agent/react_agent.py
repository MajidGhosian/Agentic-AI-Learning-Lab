"""
react_agent.py
----------------
A minimal, dependency-light implementation of the ReAct pattern
(Reasoning + Acting), as described in:

    Yao et al., 2022 — "ReAct: Synergizing Reasoning and Acting in
    Language Models" (https://arxiv.org/abs/2210.03629)

The core idea:
    The LLM alternates between three phases in a loop:

        Thought:  the model reasons about what to do next
        Action:   the model picks a tool + input to call
        Observation: the result of the tool call is fed back in

    This repeats until the model emits a `Final Answer:`.

This implementation does NOT use native "tool calling" / function-calling
APIs on purpose — it implements the classic *text-based* ReAct loop by
prompting the model to produce a structured Thought/Action/Action Input
transcript, then parsing that text ourselves and executing the matching
Python tool. That makes the pattern fully visible and easy to reason
about, rather than hidden behind a provider's tool-use abstraction.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python react_agent.py "What is 12% of 850, plus the length of the word 'agentic'?"
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable

from anthropic import Anthropic

from tools import TOOL_REGISTRY, TOOLS_DESCRIPTION


MODEL = "claude-sonnet-4-6"
MAX_STEPS = 6  # safety cap so a looping agent can't run forever


SYSTEM_PROMPT = f"""You are an assistant that solves problems using the ReAct pattern:
alternating between Thought, Action, and Observation steps.

You have access to the following tools:

{TOOLS_DESCRIPTION}

Respond using EXACTLY this format, one step at a time. Do not skip ahead
and do not answer directly without using this format:

Thought: <your reasoning about what to do next>
Action: <one of: {", ".join(TOOL_REGISTRY.keys())}>
Action Input: <the input to the tool>

After you receive an "Observation:", continue the loop with another
Thought/Action/Action Input, or, once you have enough information,
finish with:

Thought: <final reasoning>
Final Answer: <the final answer to the user's question>

Rules:
- Only ever output ONE Thought/Action/Action Input block per turn, then stop
  and wait for the Observation.
- Never fabricate an Observation yourself — it will be provided to you.
- If no tool is needed, reason briefly and go straight to Final Answer.
"""


@dataclass
class Step:
    thought: str
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None
    final_answer: str | None = None


@dataclass
class ReActAgent:
    client: Anthropic = field(default_factory=lambda: Anthropic())
    model: str = MODEL
    verbose: bool = True
    max_steps: int = MAX_STEPS

    def _log(self, text: str) -> None:
        if self.verbose:
            print(text)

    def run(self, question: str) -> str:
        transcript: list[dict] = [{"role": "user", "content": question}]
        history: list[Step] = []

        for step_num in range(1, self.max_steps + 1):
            response_text = self._call_model(transcript)
            step = self._parse_step(response_text)

            self._log(f"\n--- Step {step_num} ---")
            self._log(f"Thought: {step.thought}")

            if step.final_answer is not None:
                self._log(f"Final Answer: {step.final_answer}")
                return step.final_answer

            if step.action and step.action_input is not None:
                self._log(f"Action: {step.action}[{step.action_input}]")
                observation = self._execute_tool(step.action, step.action_input)
                step.observation = observation
                self._log(f"Observation: {observation}")

                # Feed the assistant's own text back in, followed by the
                # observation, so the model sees the full running transcript.
                transcript.append({"role": "assistant", "content": response_text})
                transcript.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )
            else:
                # Model produced something we couldn't parse — nudge it.
                transcript.append({"role": "assistant", "content": response_text})
                transcript.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last response didn't match the required "
                            "Thought/Action/Action Input or Final Answer format. "
                            "Please respond again using exactly that format."
                        ),
                    }
                )

            history.append(step)

        return (
            "Agent stopped after reaching max_steps without a Final Answer. "
            "Last observation: "
            f"{history[-1].observation if history else 'N/A'}"
        )

    def _call_model(self, transcript: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=transcript,
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    @staticmethod
    def _parse_step(text: str) -> Step:
        thought_match = re.search(r"Thought:\s*(.*)", text)
        thought = thought_match.group(1).strip().splitlines()[0] if thought_match else ""

        final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
        if final_match:
            return Step(thought=thought, final_answer=final_match.group(1).strip())

        action_match = re.search(r"Action:\s*(.*)", text)
        input_match = re.search(r"Action Input:\s*(.*)", text)

        action = action_match.group(1).strip() if action_match else None
        action_input = input_match.group(1).strip() if input_match else None

        return Step(thought=thought, action=action, action_input=action_input)

    @staticmethod
    def _execute_tool(action: str, action_input: str) -> str:
        tool: Callable[[str], str] | None = TOOL_REGISTRY.get(action)
        if tool is None:
            return f"Error: unknown tool '{action}'. Available: {list(TOOL_REGISTRY.keys())}"
        try:
            return tool(action_input)
        except Exception as exc:  # noqa: BLE001 - surface any tool error as an observation
            return f"Error while running tool '{action}': {exc}"


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY before running this script.")
        sys.exit(1)

    question = (
        " ".join(sys.argv[1:])
        or "What is 12% of 850, plus the number of letters in the word 'agentic'?"
    )

    agent = ReActAgent()
    print(f"Question: {question}")
    answer = agent.run(question)
    print(f"\n=== FINAL ANSWER ===\n{answer}")


if __name__ == "__main__":
    main()
