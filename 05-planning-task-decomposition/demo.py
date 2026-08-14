"""
demo.py
-------
Run this to see planning + task decomposition in action:

    python demo.py
    python demo.py "Plan a 3-day trip to Lisbon"

If ANTHROPIC_API_KEY is set in your environment, the goal is decomposed by
Claude (real semantic planning). Otherwise it falls back to the offline
RuleBasedDecomposer so the demo always runs.
"""

import os
import sys

from planner import LLMDecomposer, Planner, RuleBasedDecomposer, Task


def choose_decomposer():
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[info] ANTHROPIC_API_KEY found -> using LLMDecomposer (real Claude call)\n")
        return LLMDecomposer()
    print("[info] No ANTHROPIC_API_KEY found -> using offline RuleBasedDecomposer\n")
    return RuleBasedDecomposer()


def flaky_execute(task: Task) -> str:
    """Example custom executor: pretend one specific step fails once, then
    succeeds on retry, to show off the retry logic in Planner.run()."""
    if "Research" in task.title and task.attempts == 1:
        raise RuntimeError("simulated transient error")
    return f"Completed '{task.title}'"


def main():
    goal = " ".join(sys.argv[1:]) or "Build and launch a personal portfolio website"
    print(f"GOAL: {goal}\n{'=' * 60}\n")

    planner = Planner(decomposer=choose_decomposer(), execute_fn=flaky_execute)
    queue = planner.plan(goal)

    print("Decomposed into sub-tasks:")
    print(queue.summary())
    print()

    planner.run()

    print("Execution log:")
    for line in planner.log:
        print(" ", line)

    print("\nFinal status:")
    print(queue.summary())


if __name__ == "__main__":
    main()
