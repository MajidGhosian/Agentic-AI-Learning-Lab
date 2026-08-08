"""
tools.py
--------
A tiny registry of "tools" the ReAct agent can choose to call.

Each tool is a plain Python function: str -> str.
Keeping tools this simple keeps the focus of the project on the
ReAct *loop* itself, not on tool infrastructure.
"""

from __future__ import annotations

import ast
import operator


# --- Tool implementations -------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a restricted arithmetic AST (no builtins, no names)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'."""
    expression = expression.replace("%", "/100")
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree.body)
    return str(result)


def word_length(word: str) -> str:
    """Return the number of letters in a word (ignoring surrounding quotes/spaces)."""
    cleaned = word.strip().strip("'\"")
    return str(len(cleaned))


# A tiny mock "knowledge base" so the agent has something to look up
# without needing a real search API or internet access.
_MOCK_KNOWLEDGE_BASE = {
    "react pattern": (
        "ReAct (Reasoning + Acting) is a prompting pattern from a 2022 paper "
        "by Yao et al. where a language model interleaves reasoning traces "
        "(Thoughts) with actions (tool calls) and observations, improving "
        "both interpretability and task performance versus reasoning-only "
        "or acting-only prompting."
    ),
    "anthropic": (
        "Anthropic is an AI safety company that develops the Claude family "
        "of large language models."
    ),
    "capital of france": "The capital of France is Paris.",
}


def search(query: str) -> str:
    """Look up a fact in a small mock knowledge base (stand-in for a real search tool)."""
    key = query.strip().strip("'\"").lower()
    for topic, fact in _MOCK_KNOWLEDGE_BASE.items():
        if topic in key or key in topic:
            return fact
    return (
        f"No result found for '{query}' in the mock knowledge base. "
        "(In a production agent this would call a real search API.)"
    )


# --- Registry ---------------------------------------------------------------

TOOL_REGISTRY = {
    "calculator": calculator,
    "word_length": word_length,
    "search": search,
}

TOOLS_DESCRIPTION = """\
- calculator: evaluate a math expression, e.g. "12 * (3 + 4)" or "12% of 850" -> "12/100*850"
- word_length: count the letters in a word, e.g. "agentic" -> 7
- search: look up a fact in a small mock knowledge base, e.g. "capital of France\""""
