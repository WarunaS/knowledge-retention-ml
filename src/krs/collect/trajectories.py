"""Agent trajectory loaders: SWE-agent and OpenHands runs on real repositories (modality=agentic).

- SWE-agent: `nebius/SWE-agent-trajectories`, CC BY 4.0, 80,036 runs.
  Roles are system / user / ai. The first user message is the issue; later user messages are
  tool observations, so they become TOOL turns named after the command that produced them.
- OpenHands: `nebius/SWE-rebench-openhands-trajectories`, CC BY 4.0.
  OpenAI-style messages: system / user / assistant (with tool_calls) / tool.

The repository is the project id. Completed-run filtering is Stage 3; exit status and
whether the run resolved the issue are kept in metadata for that.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from typing import Any

from krs.collect._hf import iter_rows
from krs.collect._util import build_turns
from krs.schema import Conversation, Modality, Role, SourceDataset

LICENCE = "CC-BY-4.0"
HF_IDS = {
    "swe_agent": "nebius/SWE-agent-trajectories",
    "openhands": "nebius/SWE-rebench-openhands-trajectories",
}
TOOL_ARGS_MAX_CHARS = 2000  # tool-call arguments are trimmed here; long outputs are Stage 3

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def repo_from_instance_id(instance_id: str) -> str | None:
    """'django__django-11099' -> 'django/django'."""
    if "__" not in instance_id:
        return None
    owner, rest = instance_id.split("__", 1)
    name = re.sub(r"-\d+$", "", rest)
    return f"{owner}/{name}"


def _last_command(text: str) -> str | None:
    """First word of the last fenced block in a SWE-agent ai message (the command it ran)."""
    blocks = _FENCE.findall(text or "")
    if not blocks:
        return None
    words = blocks[-1].strip().split()
    return words[0] if words else None


def swe_agent_to_conversation(row: dict[str, Any]) -> Conversation:
    items: list[tuple[Role, str, str | None]] = []
    seen_issue = False
    last_cmd: str | None = None
    for step in row.get("trajectory") or []:
        role, text = step.get("role"), step.get("text") or ""
        if role == "system":
            items.append((Role.SYSTEM, step.get("system_prompt") or text, None))
        elif role == "ai":
            items.append((Role.ASSISTANT, text, None))
            last_cmd = _last_command(text)
        elif role == "user" and not seen_issue:
            items.append((Role.USER, text, None))
            seen_issue = True
        elif role == "user":
            items.append((Role.TOOL, text, last_cmd))
    raw = json.dumps([row.get("instance_id"), row.get("model_name"), row.get("trajectory")], default=str)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return Conversation(
        conversation_id=f"swe_agent:{row.get('instance_id')}:{digest}",
        source_dataset=SourceDataset.SWE_AGENT,
        licence=LICENCE,
        modality=Modality.AGENTIC,
        turns=build_turns(items),
        project_id=repo_from_instance_id(row.get("instance_id") or ""),
        metadata={
            "instance_id": row.get("instance_id"),
            "model": row.get("model_name"),
            "exit_status": row.get("exit_status"),
            "resolved": bool(row.get("target")),
        },
    )


def _render_tool_calls(calls: list[dict[str, Any]] | None) -> tuple[str, str | None]:
    if not calls:
        return "", None
    lines, first = [], None
    for c in calls:
        fn = c.get("function") or {}
        name = fn.get("name")
        first = first or name
        lines.append(f"[tool call] {name}: {(fn.get('arguments') or '')[:TOOL_ARGS_MAX_CHARS]}")
    return "\n".join(lines), first


def openhands_to_conversation(row: dict[str, Any]) -> Conversation:
    messages = row.get("trajectory") or []
    if isinstance(messages, str):
        messages = json.loads(messages)
    roles = {"system": Role.SYSTEM, "user": Role.USER, "assistant": Role.ASSISTANT, "tool": Role.TOOL}
    items: list[tuple[Role, str, str | None]] = []
    for m in messages:
        role = roles.get(m.get("role"))
        if role is None:
            continue
        content = m.get("content") or ""
        tool_name = m.get("name") if role == Role.TOOL else None
        if role == Role.ASSISTANT:
            calls_text, tool_name = _render_tool_calls(m.get("tool_calls"))
            content = "\n".join(p for p in (content, calls_text) if p)
        items.append((role, content, tool_name))
    return Conversation(
        conversation_id=f"openhands:{row.get('trajectory_id')}",
        source_dataset=SourceDataset.OPENHANDS,
        licence=LICENCE,
        modality=Modality.AGENTIC,
        turns=build_turns(items),
        project_id=row.get("repo") or repo_from_instance_id(row.get("instance_id") or ""),
        metadata={
            "instance_id": row.get("instance_id"),
            "exit_status": row.get("exit_status"),
            "resolved": bool(row.get("resolved")),
        },
    )


_MAPPERS = {"swe_agent": swe_agent_to_conversation, "openhands": openhands_to_conversation}


def load(
    dataset: str,
    limit: int | None = None,
    *,
    revision: str | None = None,
) -> Iterator[Conversation]:
    """Read Conversations for `dataset` ('swe_agent' or 'openhands') from Hugging Face, in file order."""
    if dataset not in HF_IDS:
        raise ValueError(f"unknown trajectory dataset {dataset!r}; expected one of {list(HF_IDS)}")
    to_conversation = _MAPPERS[dataset]
    for row in iter_rows(HF_IDS[dataset], revision, limit):
        yield to_conversation(row)
