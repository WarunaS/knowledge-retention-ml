"""Consented own-log loader. TRANSFER TEST AND PERSON-LEVEL DEMO ONLY - never used for training.

Supported tool: Claude Code (session files at ~/.claude/projects/<project-folder>/<session>.jsonl).
Only members listed in the consent file, with a consent date, are read. The consent file holds
pseudonyms, never real names, and lives in the gitignored `own_logs/` folder
(see configs/consent.example.yaml).

Raw own logs can contain secrets and personal data. Run this locally, keep the output local,
and redact (Stage 3) before anything else touches it. Never upload raw own logs to Drive or a
cloud LLM.

Mapping of a Claude Code session:
- user text -> USER; tool results -> TOOL (named after the tool that was called)
- assistant text -> ASSISTANT; tool calls -> ASSISTANT "[tool call] name: input"
- thinking blocks, images, sub-agent (sidechain) and injected meta records are skipped
- <system-reminder> blocks are stripped from user text
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from krs.collect._util import build_turns, to_utc
from krs.schema import Conversation, Modality, PersonRole, Role, SourceDataset

LICENCE = "consented-internal"
TOOL_TEXT_MAX_CHARS = 2000
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


@dataclass(frozen=True)
class Member:
    author_id: str  # pseudonym such as "p01"
    role: PersonRole
    consented_on: date
    log_dir: Path
    tool: str = "claude_code"
    include_projects: tuple[str, ...] = ()  # project folder names; empty = all
    project_names: dict[str, str] = field(default_factory=dict)  # folder name -> project id


def load_consent(path: str | Path) -> list[Member]:
    """Read the consent file. Members without a consent date are refused."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    members = []
    for m in data.get("members") or []:
        if not m.get("consented_on"):
            raise ValueError(f"member {m.get('author_id')!r} has no consented_on date")
        consented = m["consented_on"]
        if not isinstance(consented, date):
            consented = date.fromisoformat(str(consented))
        members.append(
            Member(
                author_id=str(m["author_id"]),
                role=PersonRole(m["role"]),
                consented_on=consented,
                log_dir=Path(str(m.get("log_dir", "~/.claude/projects"))).expanduser(),
                tool=m.get("tool", "claude_code"),
                include_projects=tuple(m.get("include_projects") or ()),
                project_names=dict(m.get("project_names") or {}),
            )
        )
    return members


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def session_to_conversation(path: str | Path, member: Member, project_id: str | None = None) -> Conversation | None:
    """Map one Claude Code session file to a Conversation, or None if it has no exchange."""
    path = Path(path)
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    tool_names: dict[str, str] = {}
    items: list[tuple[Role, str, str | None]] = []
    first_ts, cwd, session_id = None, None, None

    def add(role: Role, text: str, tool: str | None = None) -> None:
        text = text.strip()
        if not text:
            return
        # streamed assistant blocks and split user text arrive as separate records: merge them
        if items and tool is None and items[-1][0] == role and items[-1][2] is None:
            items[-1] = (role, items[-1][1] + "\n" + text, None)
        else:
            items.append((role, text, tool))

    for r in records:
        if r.get("type") not in ("user", "assistant") or r.get("isSidechain") or r.get("isMeta"):
            continue
        first_ts = first_ts or r.get("timestamp")
        cwd = cwd or r.get("cwd")
        session_id = session_id or r.get("sessionId")
        content = (r.get("message") or {}).get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        for b in blocks:
            kind = b.get("type")
            if r["type"] == "user" and kind == "text":
                add(Role.USER, _REMINDER.sub("", b.get("text", "")))
            elif r["type"] == "user" and kind == "tool_result":
                name = tool_names.get(b.get("tool_use_id", ""))
                add(Role.TOOL, _text_of(b.get("content"))[:TOOL_TEXT_MAX_CHARS], name or "unknown")
            elif r["type"] == "assistant" and kind == "text":
                add(Role.ASSISTANT, b.get("text", ""))
            elif r["type"] == "assistant" and kind == "tool_use":
                tool_names[b.get("id", "")] = b.get("name", "")
                args = json.dumps(b.get("input"), ensure_ascii=False)[:TOOL_TEXT_MAX_CHARS]
                add(Role.ASSISTANT, f"[tool call] {b.get('name')}: {args}", b.get("name"))

    roles = {i[0] for i in items}
    if Role.USER not in roles or Role.ASSISTANT not in roles:
        return None
    return Conversation(
        conversation_id=f"own_logs:{member.author_id}:{session_id or path.stem}",
        source_dataset=SourceDataset.OWN_LOGS,
        licence=LICENCE,
        modality=Modality.AGENTIC,
        turns=build_turns(items),
        author_id=member.author_id,
        role=member.role,
        project_id=project_id or (Path(cwd).name if cwd else None),
        occurred_at=to_utc(first_ts),
        metadata={"tool": member.tool},
    )


def load(consent_path: str | Path) -> Iterator[Conversation]:
    """Yield Conversations for every consented member's sessions."""
    for member in load_consent(consent_path):
        if member.tool != "claude_code":
            raise NotImplementedError(f"tool {member.tool!r} is not supported yet (only claude_code)")
        for folder in sorted(p for p in member.log_dir.iterdir() if p.is_dir()):
            if member.include_projects and folder.name not in member.include_projects:
                continue
            for session in sorted(folder.glob("*.jsonl")):
                conv = session_to_conversation(session, member, member.project_names.get(folder.name))
                if conv is not None:
                    yield conv
