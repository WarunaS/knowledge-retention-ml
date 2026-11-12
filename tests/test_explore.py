"""Stage 4 tests: tables, plots, domain hints, knowledge share. BERTopic test is marked slow."""

import json

import pytest

from krs import explore
from krs.io import write_jsonl
from krs.label.taxonomy import KnowledgeType
from krs.schema import (
    Conversation,
    Label,
    LabelSource,
    Modality,
    Role,
    Segment,
    SourceDataset,
    Turn,
)

U, A, T = Role.USER, Role.ASSISTANT, Role.TOOL


def _conv(cid, source, modality, turns, project=None, relevance=("programming",)):
    return Conversation(
        conversation_id=cid, source_dataset=source, licence="t", modality=modality, project_id=project,
        turns=[Turn(index=i, role=r, content=c, tool_name=tn) for i, (r, c, tn) in enumerate(turns)],
        metadata={"relevance": list(relevance)},
    )


def _seg(sid, cid, source, modality, text, ctx="", project=None):
    return Segment(segment_id=sid, conversation_id=cid, source_dataset=source, modality=modality,
                   turn_span=(0, 1), text=text, context_text=ctx, project_id=project)


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    W, S = SourceDataset.WILDCHAT, SourceDataset.SWE_AGENT
    C, G = Modality.CONVERSATIONAL, Modality.AGENTIC
    write_jsonl(tmp_path / "filtered" / "wildchat.jsonl", [
        _conv("w1", W, C, [(U, "pandas question", None), (A, "use groupby", None)] * 2),
        _conv("w2", W, C, [(U, "user story please", None), (A, "As a user...", None)] * 2, relevance=("requirements",)),
    ])
    write_jsonl(tmp_path / "filtered" / "swe_agent.jsonl", [
        _conv("s1", S, G, [(U, "fix", None), (A, "look", None), (T, "out", "open"), (A, "edit", None),
                           (T, "ok", "edit")], project="a/b"),
    ])
    write_jsonl(tmp_path / "segments" / "wildchat.jsonl", [
        _seg("w1:0", "w1", W, C, "USER: pandas dataframe groupby\nASSISTANT: use groupby"),
        _seg("w1:2", "w1", W, C, "USER: more\nASSISTANT: x" + " [... 10 characters removed ...]", ctx="USER: a"),
        _seg("w2:0", "w2", W, C, "USER: write a user story with acceptance criteria"),
    ])
    write_jsonl(tmp_path / "segments" / "swe_agent.jsonl", [
        _seg("s1:0", "s1", S, G, "USER: fix the docker build in CI pipeline", project="a/b"),
    ])
    return tmp_path


def test_tables(data):
    conv = explore.conversation_table()
    seg = explore.segment_table()
    assert len(conv) == 3 and len(seg) == 4
    s1 = conv.set_index("conversation_id").loc["s1"]
    assert s1["n_tool"] == 2 and s1["tools"] == ["edit", "open"] and s1["project_id"] == "a/b"
    assert seg["truncated"].sum() == 1 and seg["has_context"].sum() == 1


def test_summary_tables(data):
    t = explore.summary_tables(explore.conversation_table(), explore.segment_table())
    ov = t["overview"]
    assert ov.loc["wildchat", "conversations"] == 2 and ov.loc["wildchat", "segments"] == 3
    assert ov.loc["total", "segments"] == 4
    assert t["relevance"].loc["wildchat", "requirements"] == 1
    assert set(t["top_tools"]["tools"]) == {"edit", "open"}


def test_plots(data, tmp_path):
    made = explore.plot_distributions(explore.conversation_table(), explore.segment_table(), tmp_path / "eda")
    assert len(made) == 4 and all(p.exists() and p.stat().st_size > 1000 for p in made)


def test_domain_hints():
    assert explore.domain_hints("Fix the Dockerfile in the GitHub Actions pipeline")[0] == "devops_cicd"
    assert explore.domain_hints("Write a user story with acceptance criteria")[0] == "business_requirements"
    assert explore.domain_hints("banana bread recipe") == ["other"]
    assert explore.domain_hints("refactor this recursion into a loop and return a list") == ["general_programming"]
    assert explore.domain_hints("pytest fixture with a loop and a list")[0] == "qa_testing"  # specific wins
    assert "web_frontend" not in explore.domain_hints("random reactive candor")  # word boundaries


def test_sample_is_seeded_and_capped(data):
    a = explore.sample_segment_texts(n=2, seed=1)
    b = explore.sample_segment_texts(n=2, seed=1)
    assert a == b and len(a) == 2 and len(explore.sample_segment_texts(n=100)) == 4


def test_knowledge_share(tmp_path):
    labels = [Label(segment_id=f"s{i}", knowledge_types=[KnowledgeType.DECISION], source=LabelSource.LLM)
              for i in range(6)]
    labels += [Label(segment_id=f"n{i}", knowledge_types=[KnowledgeType.NONE], source=LabelSource.LLM)
               for i in range(4)]
    write_jsonl(tmp_path / "labels.jsonl", labels)
    ks = explore.knowledge_share(tmp_path / "labels.jsonl")
    assert ks["labelled"] == 10 and ks["share"] == pytest.approx(0.6) and ks["ci95_low"] < 0.6 < ks["ci95_high"]
    assert ks["knowledge_types"] == {"decision": 6, "none": 4}


def test_run_explore_without_topics(data, tmp_path):
    cfg = tmp_path / "explore.yaml"
    cfg.write_text("topics:\n  enabled: false\n")
    summary = explore.run_explore(cfg, out_dir=tmp_path / "eda")
    assert summary["conversations"] == 3 and summary["segments"] == 4 and summary["projects"] == 1
    assert (tmp_path / "eda" / "overview.csv").exists()
    assert json.loads((tmp_path / "eda" / "summary.json").read_text())["segments"] == 4


@pytest.mark.slow
def test_topics_runs_on_small_sample():
    words = ["docker build pipeline deploy", "pandas dataframe groupby plot", "react component css layout",
             "user story acceptance criteria stakeholder"]
    sample = [(f"s{i}", "wildchat", f"{words[i % 4]} example number {i}") for i in range(200)]
    table, assign = explore.topics(sample, min_topic_size=10)
    assert len(assign) == 200 and {"topic", "top_words", "domain_hint"} <= set(table.columns)
