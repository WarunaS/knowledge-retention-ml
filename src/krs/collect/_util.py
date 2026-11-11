"""Small helpers shared by the loaders."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from krs.schema import Role, Turn


def to_utc(value: Any) -> datetime | None:
    """Accept a datetime or an ISO-8601 string; return an aware UTC datetime, or None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def build_turns(items: list[tuple[Role, str, str | None]]) -> list[Turn]:
    """Number turns 0..n-1 in order. Each item is (role, content, tool_name)."""
    return [
        Turn(index=i, role=role, content=content or "", tool_name=tool)
        for i, (role, content, tool) in enumerate(items)
    ]
