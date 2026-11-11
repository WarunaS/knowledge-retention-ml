"""DevGPT loader (Xiao et al., 2024).

ChatGPT conversations shared by developers in GitHub issues, pull requests, commits, code files,
discussions and Hacker News. Licence CC BY 4.0 via Zenodo (record 10086809, one ~680 MB zip).

File layout: {"Sources": [{Type, RepoName, RepoLanguage, ..., "ChatgptSharing": [
    {URL, Status, DateOfConversation, Model, Title, "Conversations": [{Prompt, Answer, ListOfCode}]}]}]}

Each shared link becomes one Conversation. The repository (`RepoName`) is the project id, which
gives the project view on public data. GitHub usernames (`Author`, `MentionedAuthor`) are never
copied and no author is assigned. The same link can be mentioned in several places; the collect
pipeline keeps the first occurrence of each conversation id.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from krs.collect._util import build_turns
from krs.schema import Conversation, Modality, Role, SourceDataset

ZENODO_URL = "https://zenodo.org/records/10086809/files/DevGPT.zip?download=1"
LICENCE = "CC-BY-4.0"
DEFAULT_SNAPSHOT = "snapshot_20231012"  # latest snapshot


def download(dest_dir: str | Path, snapshot: str = DEFAULT_SNAPSHOT, url: str = ZENODO_URL) -> Path:
    """Download the Zenodo zip once and extract one snapshot's JSON files. Returns the snapshot dir."""
    dest = Path(dest_dir)
    out = dest / snapshot
    if out.is_dir() and any(out.glob("*.json")):
        return out
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "DevGPT.zip"
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".part")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        tmp.rename(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if f"{snapshot}/" in m and m.endswith(".json")]
        if not members:
            raise FileNotFoundError(f"No JSON files for {snapshot} inside {zip_path}")
        out.mkdir(parents=True, exist_ok=True)
        for m in members:
            with z.open(m) as src, open(out / Path(m).name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return out


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def to_conversations(source: dict[str, Any]) -> Iterator[Conversation]:
    """Map one DevGPT source (an issue, PR, commit, ...) to its shared conversations."""
    for sharing in source.get("ChatgptSharing") or []:
        exchanges = sharing.get("Conversations") or []
        if not exchanges:  # link was dead (Status != 200) in every snapshot: no content
            continue
        items: list[tuple[Role, str, str | None]] = []
        for ex in exchanges:
            items.append((Role.USER, ex.get("Prompt") or "", None))
            items.append((Role.ASSISTANT, ex.get("Answer") or "", None))
        share_id = (sharing.get("URL") or "").rstrip("/").rsplit("/", 1)[-1]
        mention = sharing.get("Mention") or {}
        yield Conversation(
            conversation_id=f"devgpt:{share_id}",
            source_dataset=SourceDataset.DEVGPT,
            licence=LICENCE,
            modality=Modality.CONVERSATIONAL,
            turns=build_turns(items),
            project_id=source.get("RepoName") or None,  # None for Hacker News
            occurred_at=_parse_date(sharing.get("DateOfConversation")),
            metadata={
                "source_type": source.get("Type"),
                "repo_language": source.get("RepoLanguage"),
                "model": sharing.get("Model"),
                "title": sharing.get("Title"),
                "share_url": sharing.get("URL"),
                "mentioned_property": mention.get("MentionedProperty"),
            },
        )


def load(snapshot_dir: str | Path) -> Iterator[Conversation]:
    """Yield Conversations from every *.json file of one extracted snapshot, in name order."""
    for path in sorted(Path(snapshot_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        sources = data.get("Sources", []) if isinstance(data, dict) else data
        for source in sources:
            yield from to_conversations(source)
