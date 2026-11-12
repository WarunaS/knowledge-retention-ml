"""Stage 2 tests. Fixture rows copy the real dataset schemas (checked 11 Nov 2026), with short text."""

import json
from datetime import datetime, timezone

import pytest

from krs.collect import devgpt, manifest, own_logs, trajectories, wildchat
from krs.collect.pipeline import unique
from krs.schema import Modality, PersonRole, Role, SourceDataset

# --------------------------------------------------------------------------- WildChat

WILDCHAT_ROW = {
    "conversation_hash": "c9ec5b440fbdd2a269333dd241f32f64",
    "model": "gpt-4-0314",
    "timestamp": datetime(2023, 4, 9, 0, 2, 53, tzinfo=timezone.utc),
    "conversation": [
        {"content": "How do I add an index in Postgres?", "role": "user", "language": "English",
         "hashed_ip": "22fd87ba", "country": "United States", "state": "Texas",
         "header": {"user-agent": "Mozilla/5.0"}, "toxic": False, "redacted": False},
        {"content": "Use CREATE INDEX ...", "role": "assistant", "language": "English",
         "hashed_ip": None, "country": None, "state": None, "header": None, "toxic": False},
    ],
    "turn": 1,
    "language": "English",
    "toxic": False,
    "redacted": False,
    "state": "Texas",
    "country": "United States",
    "hashed_ip": "22fd87ba",
    "header": {"user-agent": "Mozilla/5.0"},
}


def test_wildchat_maps_turns_and_language():
    c = wildchat.to_conversation(WILDCHAT_ROW)
    assert c.conversation_id == "wildchat:c9ec5b440fbdd2a269333dd241f32f64"
    assert [t.role for t in c.turns] == [Role.USER, Role.ASSISTANT]
    assert c.language == "en" and c.toxic is False
    assert c.modality == Modality.CONVERSATIONAL and c.licence == "ODC-BY-1.0"
    assert c.occurred_at == datetime(2023, 4, 9, 0, 2, 53, tzinfo=timezone.utc)


def test_wildchat_never_copies_identifying_fields():
    c = wildchat.to_conversation(WILDCHAT_ROW)
    dumped = c.model_dump_json()
    for leak in ("22fd87ba", "Texas", "United States", "Mozilla"):
        assert leak not in dumped
    assert c.author_id is None


def test_wildchat_language_codes():
    assert wildchat.normalise_language("English") == "en"
    assert wildchat.normalise_language("Russian") == "russian"
    assert wildchat.normalise_language(None) is None


# --------------------------------------------------------------------------- DevGPT

DEVGPT_SOURCE = {
    "Type": "commit",
    "URL": "https://github.com/grnpin/textbox/commit/fa335cf",
    "Author": "grnpin",
    "RepoName": "grnpin/textbox",
    "RepoLanguage": "HTML",
    "ChatgptSharing": [
        {
            "URL": "https://chat.openai.com/share/4bad57dd-9636-4b63-9e73-37351f51dbc9",
            "Mention": {"MentionedProperty": "message", "MentionedAuthor": "grnpin", "MentionedText": "see chat"},
            "Status": 200,
            "DateOfConversation": "September 20, 2023",
            "Title": "Adjust Button for Keyboard",
            "Model": "Default (GPT-3.5)",
            "Conversations": [
                {"Prompt": "Move the button above the keyboard", "Answer": "Use visualViewport ...", "ListOfCode": []},
                {"Prompt": "Why not position fixed?", "Answer": "Because iOS ...", "ListOfCode": []},
            ],
        },
        {"URL": "https://chat.openai.com/share/dead-link", "Status": 404, "Conversations": []},
    ],
}


def test_devgpt_maps_sharing_with_project():
    [c] = list(devgpt.to_conversations(DEVGPT_SOURCE))  # dead link yields nothing
    assert c.conversation_id == "devgpt:4bad57dd-9636-4b63-9e73-37351f51dbc9"
    assert c.project_id == "grnpin/textbox"
    assert [t.role for t in c.turns] == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]
    assert c.occurred_at == datetime(2023, 9, 20, tzinfo=timezone.utc)
    assert c.metadata["repo_language"] == "HTML"


def test_devgpt_never_copies_usernames():
    [c] = list(devgpt.to_conversations(DEVGPT_SOURCE))
    assert c.author_id is None
    assert "grnpin" not in json.dumps(c.metadata).replace("grnpin/textbox", "")


def test_devgpt_load_reads_snapshot_files(tmp_path):
    hn = {"Type": "hacker news", "ChatgptSharing": [{**DEVGPT_SOURCE["ChatgptSharing"][0], "URL": "https://chat.openai.com/share/hn1"}]}
    (tmp_path / "a_commit_sharings.json").write_text(json.dumps({"Sources": [DEVGPT_SOURCE]}))
    (tmp_path / "b_hn_sharings.json").write_text(json.dumps({"Sources": [hn]}))
    convs = list(devgpt.load(tmp_path))
    assert [c.conversation_id for c in convs] == ["devgpt:4bad57dd-9636-4b63-9e73-37351f51dbc9", "devgpt:hn1"]
    assert convs[1].project_id is None  # Hacker News has no repository


# --------------------------------------------------------------------------- trajectories

SWE_AGENT_ROW = {
    "instance_id": "AnalogJ__lexicon-336",
    "model_name": "swe-agent-llama-70b",
    "target": False,
    "exit_status": "submitted (exit_context)",
    "trajectory": [
        {"role": "system", "system_prompt": "SETTING: You are an autonomous programmer", "text": None, "mask": False},
        {"role": "user", "text": "ISSUE: Memset provider TypeError", "mask": False},
        {"role": "ai", "text": "Let's reproduce it.\n```\npython reproduce.py\n```", "mask": True},
        {"role": "user", "text": "Traceback (most recent call last): ...", "mask": False},
        {"role": "ai", "text": "Open the file.\n```\nopen lexicon/providers/memset.py\n```", "mask": True},
        {"role": "user", "text": "[File: memset.py (120 lines)]", "mask": False},
    ],
}


def test_swe_agent_roles_and_tool_names():
    c = trajectories.swe_agent_to_conversation(SWE_AGENT_ROW)
    assert [t.role for t in c.turns] == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT, Role.TOOL]
    assert [t.tool_name for t in c.turns if t.role == Role.TOOL] == ["python", "open"]
    assert c.project_id == "AnalogJ/lexicon"
    assert c.modality == Modality.AGENTIC and c.source_dataset == SourceDataset.SWE_AGENT
    assert c.metadata["resolved"] is False


def test_swe_agent_id_is_deterministic():
    a = trajectories.swe_agent_to_conversation(SWE_AGENT_ROW)
    b = trajectories.swe_agent_to_conversation(dict(SWE_AGENT_ROW))
    assert a.conversation_id == b.conversation_id


def test_repo_from_instance_id():
    assert trajectories.repo_from_instance_id("django__django-11099") == "django/django"
    assert trajectories.repo_from_instance_id("PlasmaFAIR__sdf-xarray-24") == "PlasmaFAIR/sdf-xarray"
    assert trajectories.repo_from_instance_id("nounderscores") is None


OPENHANDS_ROW = {
    "trajectory_id": "chatcmpl-27dd0152",
    "instance_id": "PlasmaFAIR__sdf-xarray-24",
    "repo": "PlasmaFAIR/sdf-xarray",
    "exit_status": "submit",
    "resolved": 0,
    "trajectory": [
        {"role": "system", "content": "You are OpenHands agent", "name": None, "tool_calls": None},
        {"role": "user", "content": "Fix the dataset loader bug", "name": None, "tool_calls": None},
        {"role": "assistant", "content": "I'll look at the repo.", "name": None,
         "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "execute_bash", "arguments": "{\"command\": \"ls\"}"}}]},
        {"role": "tool", "content": "README.md src tests", "name": "execute_bash", "tool_call_id": "t1", "tool_calls": None},
    ],
}


def test_openhands_maps_tool_calls():
    c = trajectories.openhands_to_conversation(OPENHANDS_ROW)
    assert c.conversation_id == "openhands:chatcmpl-27dd0152"
    assert [t.role for t in c.turns] == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.TOOL]
    assert "[tool call] execute_bash" in c.turns[2].content and c.turns[2].tool_name == "execute_bash"
    assert c.turns[3].tool_name == "execute_bash"
    assert c.project_id == "PlasmaFAIR/sdf-xarray"


def test_unknown_trajectory_dataset():
    with pytest.raises(ValueError):
        next(trajectories.load("nope"))


# --------------------------------------------------------------------------- own logs


def _write_session(folder, name="s1.jsonl"):
    recs = [
        {"type": "user", "sessionId": "abc", "cwd": "/work/orders", "timestamp": "2026-03-12T10:00:00Z",
         "message": {"role": "user", "content": "Why Postgres?<system-reminder>REMINDER-TEXT</system-reminder>"}},
        {"type": "assistant", "sessionId": "abc", "message": {"id": "m1", "role": "assistant",
         "content": [{"type": "thinking", "thinking": "PRIVATE-THINKING"}]}},
        {"type": "assistant", "sessionId": "abc", "message": {"id": "m1", "role": "assistant",
         "content": [{"type": "text", "text": "Because orders need transactions."}]}},
        {"type": "assistant", "sessionId": "abc", "message": {"id": "m1", "role": "assistant",
         "content": [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"command": "psql -c 'select 1'"}}]}},
        {"type": "user", "sessionId": "abc", "message": {"role": "user",
         "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "1 row"}]}},
        {"type": "user", "isMeta": True, "sessionId": "abc", "message": {"role": "user", "content": "INJECTED-META-RECORD"}},
        {"type": "user", "isSidechain": True, "sessionId": "abc", "message": {"role": "user", "content": "SUBAGENT-RECORD"}},
        {"type": "file-history-snapshot", "snapshot": {}},
        "not json",
    ]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in recs))


def _consent(tmp_path, log_dir, **extra):
    member = {"author_id": "p01", "role": "tech_lead", "consented_on": "2026-11-20", "log_dir": str(log_dir), **extra}
    p = tmp_path / "consent.yaml"
    import yaml
    p.write_text(yaml.safe_dump({"members": [member]}))
    return p


def test_own_logs_session_mapping(tmp_path):
    logs = tmp_path / "projects"
    _write_session(logs / "-work-orders")
    [c] = list(own_logs.load(_consent(tmp_path, logs)))
    assert c.conversation_id == "own_logs:p01:abc"
    assert c.author_id == "p01" and c.role == PersonRole.TECH_LEAD and c.project_id == "orders"
    assert [t.role for t in c.turns] == [Role.USER, Role.ASSISTANT, Role.ASSISTANT, Role.TOOL]
    assert c.turns[0].content == "Why Postgres?"  # reminder stripped
    assert c.turns[2].tool_name == "Bash" and c.turns[3].tool_name == "Bash"
    text = c.model_dump_json()
    for skipped in ("PRIVATE-THINKING", "INJECTED-META-RECORD", "SUBAGENT-RECORD", "REMINDER-TEXT"):
        assert skipped not in text


def test_own_logs_project_name_mapping(tmp_path):
    logs = tmp_path / "projects"
    _write_session(logs / "-work-orders")
    [c] = list(own_logs.load(_consent(tmp_path, logs, project_names={"-work-orders": "orders-service"})))
    assert c.project_id == "orders-service"


def test_own_logs_refuses_member_without_consent(tmp_path):
    import yaml
    p = tmp_path / "consent.yaml"
    p.write_text(yaml.safe_dump({"members": [{"author_id": "p02", "role": "qa_engineer"}]}))
    with pytest.raises(ValueError):
        own_logs.load_consent(p)


def test_own_logs_include_projects(tmp_path):
    logs = tmp_path / "projects"
    _write_session(logs / "-work-orders")
    _write_session(logs / "-personal-stuff")
    convs = list(own_logs.load(_consent(tmp_path, logs, include_projects=["-work-orders"])))
    assert len(convs) == 1


# --------------------------------------------------------------------------- manifest + pipeline


def test_manifest_record_and_replace(tmp_path):
    path = tmp_path / "manifest.json"
    manifest.record("wildchat", source="allenai/WildChat-1M", licence="ODC-BY-1.0", n_rows=10, version="abc", path=path)
    manifest.record("wildchat", source="allenai/WildChat-1M", licence="ODC-BY-1.0", n_rows=20, version="abc", path=path)
    manifest.record("devgpt", source="zenodo", licence="CC-BY-4.0", n_rows=5, path=path)
    data = manifest.read(path)
    assert list(data) == ["devgpt", "wildchat"]
    assert data["wildchat"]["n_rows"] == 20 and "collected_at" in data["wildchat"]


def test_unique_skips_repeated_ids():
    a = wildchat.to_conversation(WILDCHAT_ROW)
    counter = {}
    out = list(unique([a, a, a], counter))
    assert len(out) == 1 and counter["duplicate_ids"] == 2


# --------------------------------------------------------------------------- parquet reader sampling


def test_iter_rows_rows_per_group_spreads_sample(tmp_path, monkeypatch):
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    from krs.collect import _hf

    # one file, 4 row groups of 100 rows, sorted by repository like the SWE-agent files
    repos = [f"repo{i // 100}" for i in range(400)]
    table = pa.table({"repo": repos, "n": list(range(400))})
    path = tmp_path / "train.parquet"
    pq.write_table(table, path, row_group_size=100)
    monkeypatch.setattr(_hf, "_fs", lambda: fsspec.filesystem("file"))
    monkeypatch.setattr(_hf, "parquet_files", lambda repo, rev=None: [str(path)])

    in_order = list(_hf.iter_rows("x", limit=20))
    assert {r["repo"] for r in in_order} == {"repo0"}
    spread = list(_hf.iter_rows("x", limit=20, random_row_groups=True, rows_per_group=5, seed=1))
    assert len(spread) == 20 and len({r["repo"] for r in spread}) == 4
    again = list(_hf.iter_rows("x", limit=20, random_row_groups=True, rows_per_group=5, seed=1))
    assert [r["n"] for r in spread] == [r["n"] for r in again]  # reproducible
    assert len({r["n"] for r in spread}) == 20  # no row twice
