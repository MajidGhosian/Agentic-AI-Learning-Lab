"""
planner.py
----------
Core building blocks for the "Planning & Task Decomposition" mini-project.

Concept being demonstrated
===========================
Given a high-level GOAL (e.g. "Launch a personal portfolio website"), an
agent rarely can (or should) act on it directly. Instead, it should:

  1. DECOMPOSE the goal into smaller, concrete sub-tasks.
  2. Track DEPENDENCIES between those sub-tasks (some must happen before
     others).
  3. Maintain a TASK QUEUE that always knows what is safe to run next.
  4. EXECUTE tasks (here: simulated, but pluggable with real tool calls),
     updating status as it goes.
  5. Handle FAILURES by retrying, skipping, or re-planning.

This file has zero required external dependencies. If an ANTHROPIC_API_KEY
environment variable is present, `LLMDecomposer` will call the real Claude
API to break the goal down. Otherwise, `RuleBasedDecomposer` produces a
deterministic decomposition so the whole pipeline still runs end-to-end
without any credentials -- useful for demos, tests, and CI.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    title: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)   # list of task ids
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 2

    def __repr__(self) -> str:
        return f"<Task {self.id} [{self.status.value}] {self.title!r}>"


# ---------------------------------------------------------------------------
# Task queue with dependency resolution
# ---------------------------------------------------------------------------

class TaskQueue:
    """A dependency-aware queue: a task only becomes RUNNABLE once every
    task it depends on has reached DONE (or SKIPPED)."""

    def __init__(self, tasks: list[Task] | None = None):
        self.tasks: dict[str, Task] = {t.id: t for t in (tasks or [])}

    def add(self, task: Task) -> None:
        self.tasks[task.id] = task

    def by_id(self, task_id: str) -> Task:
        return self.tasks[task_id]

    def _dependencies_satisfied(self, task: Task) -> bool:
        for dep_id in task.depends_on:
            dep = self.tasks.get(dep_id)
            if dep is None:
                continue
            if dep.status not in (TaskStatus.DONE, TaskStatus.SKIPPED):
                return False
        return True

    def next_runnable(self) -> Optional[Task]:
        """Return the next task that is PENDING and whose dependencies are
        all satisfied, or None if nothing can run right now."""
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and self._dependencies_satisfied(task):
                return task
        return None

    def all_finished(self) -> bool:
        return all(t.status in (TaskStatus.DONE, TaskStatus.SKIPPED, TaskStatus.FAILED)
                    for t in self.tasks.values())

    def summary(self) -> str:
        lines = []
        for t in self.tasks.values():
            deps = f" (after {', '.join(t.depends_on)})" if t.depends_on else ""
            lines.append(f"  [{t.status.value:^11}] {t.id}  {t.title}{deps}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decomposers: turn a goal (string) into a list of Task objects
# ---------------------------------------------------------------------------

class RuleBasedDecomposer:
    """A deterministic, offline fallback decomposer. It doesn't understand
    the goal semantically -- it just demonstrates the *mechanics* of
    decomposition + dependency wiring so the pipeline runs without an API
    key. Swap in LLMDecomposer for real semantic planning."""

    TEMPLATE = [
        ("Clarify the goal", "Restate the goal in concrete, measurable terms."),
        ("Research / gather requirements", "Collect the information needed to act."),
        ("Draft a plan", "Break the work into an ordered list of steps."),
        ("Execute core work", "Do the main task(s) implied by the goal."),
        ("Review & verify", "Check the output against the original goal."),
        ("Deliver / wrap up", "Package and report the final result."),
    ]

    def decompose(self, goal: str) -> list[Task]:
        tasks = []
        previous_id = None
        for title, desc in self.TEMPLATE:
            t = Task(
                title=f"{title}: {goal}" if title == "Clarify the goal" else title,
                description=desc,
                depends_on=[previous_id] if previous_id else [],
            )
            tasks.append(t)
            previous_id = t.id
        return tasks


class LLMDecomposer:
    """Uses the real Anthropic API to semantically decompose a goal into an
    ordered list of sub-tasks with dependencies. Requires ANTHROPIC_API_KEY.
    """

    SYSTEM_PROMPT = (
        "You are a task-planning assistant. Given a high-level goal, break it "
        "into 4-8 concrete, actionable sub-tasks. Respond ONLY with JSON: a "
        "list of objects with keys 'title', 'description', and "
        "'depends_on_index' (the 0-based index of the sub-task it depends on, "
        "or null if it has no dependency). No prose, no markdown fences."
    )

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def decompose(self, goal: str) -> list[Task]:
        import urllib.request

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; use RuleBasedDecomposer instead.")

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": self.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"Goal: {goal}"}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        text = "".join(block.get("text", "") for block in data.get("content", []))
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        raw_tasks = json.loads(text)

        tasks: list[Task] = []
        for item in raw_tasks:
            tasks.append(Task(title=item["title"], description=item.get("description", "")))
        for item, task in zip(raw_tasks, tasks):
            dep_idx = item.get("depends_on_index")
            if dep_idx is not None:
                task.depends_on = [tasks[dep_idx].id]
        return tasks


# ---------------------------------------------------------------------------
# Executor: walks the queue, running each ready task
# ---------------------------------------------------------------------------

ExecuteFn = Callable[[Task], str]


def default_execute(task: Task) -> str:
    """Placeholder 'execution'. In a real agent this is where you'd call a
    tool, hit an API, run code, etc. Here we just simulate work."""
    time.sleep(0.05)
    return f"Completed '{task.title}'"


class Planner:
    """Orchestrates decomposition + queued execution of a goal."""

    def __init__(self, decomposer, execute_fn: ExecuteFn = default_execute):
        self.decomposer = decomposer
        self.execute_fn = execute_fn
        self.queue: Optional[TaskQueue] = None
        self.log: list[str] = []

    def plan(self, goal: str) -> TaskQueue:
        tasks = self.decomposer.decompose(goal)
        self.queue = TaskQueue(tasks)
        self.log.append(f"Decomposed goal into {len(tasks)} sub-tasks.")
        return self.queue

    def run(self) -> TaskQueue:
        assert self.queue is not None, "Call plan(goal) before run()."
        while not self.queue.all_finished():
            task = self.queue.next_runnable()
            if task is None:
                # Nothing runnable but not all finished -> deadlock/failure upstream
                stuck = [t for t in self.queue.tasks.values() if t.status == TaskStatus.PENDING]
                for t in stuck:
                    t.status = TaskStatus.SKIPPED
                    self.log.append(f"Skipping {t.id} ({t.title}): blocked by a failed dependency.")
                continue

            task.status = TaskStatus.IN_PROGRESS
            task.attempts += 1
            self.log.append(f"Running {task.id}: {task.title}")
            try:
                task.result = self.execute_fn(task)
                task.status = TaskStatus.DONE
                self.log.append(f"  -> done: {task.result}")
            except Exception as exc:  # noqa: BLE001
                if task.attempts < task.max_attempts:
                    task.status = TaskStatus.PENDING
                    self.log.append(f"  -> failed ({exc}); will retry")
                else:
                    task.status = TaskStatus.FAILED
                    self.log.append(f"  -> failed permanently ({exc})")
        return self.queue
