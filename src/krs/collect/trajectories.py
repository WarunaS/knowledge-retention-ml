"""SWE-agent / OpenHands trajectory loader. Agent runs with tool calls (modality=agentic)."""

from __future__ import annotations


def load(dataset: str, limit: int | None = None):
    """Yield Conversation records; tool calls become Turn(role=tool, tool_name=...)."""
    raise NotImplementedError
