"""
Central configuration for the Agentic RAG demo.

Nothing here is secret — the actual API key is read from the environment
(ANTHROPIC_API_KEY) by the Anthropic SDK itself, never hardcoded.
"""

import os
from pathlib import Path

# Root of the project (07-agentic-rag/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the knowledge base text files live
DOCS_DIR = PROJECT_ROOT / "data" / "docs"

# Which Claude model to use for the agent's reasoning + tool-calling loop.
# Override with the MODEL_NAME env var if you want to swap models without
# touching code. Update this default whenever a newer model is available.
MODEL_NAME = os.environ.get("MODEL_NAME", "claude-sonnet-4-5")

# Max reasoning/tool-call turns before we force the agent to answer, so a
# confused agent can't loop forever calling the retriever.
MAX_AGENT_TURNS = int(os.environ.get("MAX_AGENT_TURNS", "6"))

# How many chunks the retriever returns per call
TOP_K = int(os.environ.get("TOP_K", "3"))
