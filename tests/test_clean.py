"""Stage 3 tests: filters, normalise, dedup, segment, redact, precision check, pipeline."""

import csv
import json

import pytest

from krs.clean import dedup, filters, normalise, precision_check, redact, segment
from krs.clean.pipeline import run_cleaning
from krs.io import read_jsonl, write_jsonl
from krs.schema import Conversation, Modality, PersonRole, Role, Segment, SourceDataset, Turn


def conv(turns, cid="c1", source=SourceDataset.WILDCHAT, modality=Modality.CONVERSATIONAL, **kw):
    return Conversation(
        conversation_id=cid,
        source_dataset=source,
        licence="test",
        modality=modality,
        turns=[Turn(index=i, role=r, content=c, tool_name=tn) for i, (r, c, tn) in enumerate(turns)],
        **kw,
    )


U, A, T, S = Role.USER, Role.ASSISTANT, Role.TOOL, Role.SYSTEM
CODE_CHAT = [
    (U, "My Python function throws a KeyError when I read the JSON config", None),
    (A, "Use dict.get:\n```python\nvalue = cfg.get('key')\n```", None),
    (U, "Why not catch the exception?", None),
    (A, "Catching hides the bug; get() makes the default explicit.", None),
]
BA_CHAT = [
    (U, "Write a user story for the refund process", None),
    (A, "As a customer I want a refund... Acceptance criteria: ...", None),
    (U, "Add the business rule for partial refunds", None),
    (A, "Business rule: partial refunds need manager sign-off.", None),
]
RECIPE_CHAT = [
    (U, "Give me a recipe for banana bread please, something easy for the weekend", None),
    (A, "Mash three bananas, mix with flour, sugar and an egg, bake for an hour.", None),
    (U, "Can I use honey instead of sugar in this banana bread?", None),
    (A, "Yes, use a little less honey than sugar and lower the oven slightly.", None),
]

# --------------------------------------------------------------------------- filters


def test_relevance_categories():
    assert filters.relevance(conv(CODE_CHAT)) == ["programming"]
    assert filters.relevance(conv(BA_CHAT)) == ["requirements"]
    assert filters.relevance(conv(RECIPE_CHAT)) == []


def test_keywords_need_word_boundaries():
    # "java" inside "javanese", "api" inside "rapid" must not count
    c = conv([(U, "Javanese food is rapid to cook", None), (A, "Indeed, very rapid", None)])
    assert filters.relevance(c) == []


def test_language_detection_and_unknown_kept():
    fr = conv([(U, "Bonjour, je voudrais comprendre comment fonctionne la photosynthèse des plantes vertes", None),
               (A, "x", None)])
    assert filters.is_english(fr) is False
    only_code = conv([(U, "```\nx = 1\n```", None), (A, "ok", None)])
    assert filters.detect_language(only_code) is None and filters.is_english(only_code)
    assert filters.is_english(conv(CODE_CHAT, language="en"))
    assert not filters.is_english(conv(CODE_CHAT, language="russian"))


def test_min_exchanges_exempts_agentic():
    short = CODE_CHAT[:2]
    assert not filters.has_min_exchanges(conv(short))
    assert filters.has_min_exchanges(conv(short, modality=Modality.AGENTIC))


CFG = {"language": "en", "drop_toxic": True, "min_exchanges": 2, "min_exchanges_apply_to": ["wildchat"],
       "exempt_agentic_from_min_exchanges": True,
       "relevance": {"keep": ["programming", "requirements"], "apply_to": ["wildchat"], "min_hits": 2}}


def test_drop_reasons():
    assert filters.drop_reason(conv(CODE_CHAT, language="en"), CFG, "wildchat") is None
    assert filters.drop_reason(conv(BA_CHAT, language="en"), CFG, "wildchat") is None
    assert filters.drop_reason(conv(RECIPE_CHAT, language="en"), CFG, "wildchat") == "relevance"
    assert filters.drop_reason(conv(RECIPE_CHAT, language="en"), CFG, "devgpt") is None  # not applied
    assert filters.drop_reason(conv(CODE_CHAT, language="en", toxic=True), CFG, "wildchat") == "toxic"
    assert filters.drop_reason(conv(CODE_CHAT, language="russian"), CFG, "wildchat") == "language"
    assert filters.drop_reason(conv(CODE_CHAT[:2], language="en"), CFG, "wildchat") == "min_exchanges"
    assert filters.drop_reason(conv(CODE_CHAT[:2], language="en"), CFG, "devgpt") is None  # one-question share kept
    c = conv(CODE_CHAT, language="en")
    filters.drop_reason(c, CFG, "wildchat")
    assert c.metadata["relevance"] == ["programming"]


# --------------------------------------------------------------------------- normalise


def test_normalise_text_keeps_indentation():
    text = "a\r\nb\u200b\x07  \n\n\n\n    indented()"
    assert normalise.normalise_text(text) == "a\nb\n\n    indented()"


def test_normalise_drops_system_shortens_tool_and_renumbers():
    c = conv([(S, "sys", None), (U, "run it", None), (A, "ok", None), (T, "x" * 5000, "bash")],
             modality=Modality.AGENTIC)
    n = normalise.normalise(c, tool_output_max_chars=300)
    assert [t.role for t in n.turns] == [U, A, T]
    assert [t.index for t in n.turns] == [0, 1, 2]
    assert "characters removed" in n.turns[2].content and len(n.turns[2].content) < 400


# --------------------------------------------------------------------------- dedup


def test_near_duplicates_removed_distinct_kept():
    a = conv(CODE_CHAT, cid="a")
    b = conv(CODE_CHAT[:3] + [(A, "Catching hides the bug; get() makes the default explicit!", None)], cid="b")
    c = conv(BA_CHAT, cid="c")
    kept = [x.conversation_id for x in dedup.deduplicate([a, b, c])]
    assert kept == ["a", "c"]


def test_agent_runs_on_same_issue_not_merged():
    issue = (U, "ISSUE: " + "the parser fails on empty input " * 100, None)
    run1 = conv([issue, (A, "Open parser.py and add a guard for empty strings before tokenising", None),
                 (T, "file opened", "open"), (A, "Edit line 40 to return an empty list early", None)],
                cid="r1", modality=Modality.AGENTIC)
    run2 = conv([issue, (A, "Search the tests folder for failing cases with grep first", None),
                 (T, "3 matches", "grep"), (A, "Run pytest to reproduce the failure and read the trace", None)],
                cid="r2", modality=Modality.AGENTIC)
    assert len(list(dedup.deduplicate([run1, run2]))) == 2


# --------------------------------------------------------------------------- segment


def test_chat_segments_with_context():
    segs = list(segment.segment(conv(CODE_CHAT, project_id="p", author_id="a1", role=PersonRole.QA_ENGINEER)))
    assert [s.segment_id for s in segs] == ["c1:0", "c1:2"]
    assert segs[0].context_text == "" and segs[0].turn_span == (0, 1)
    assert segs[1].context_text.startswith("USER: My Python") and segs[1].turn_span == (0, 3)
    assert segs[1].text.startswith("USER: Why not") and "ASSISTANT:" in segs[1].text
    assert segs[1].project_id == "p" and segs[1].role == PersonRole.QA_ENGINEER


def test_agent_steps_become_segments():
    c = conv([(U, "fix bug", None), (A, "look", None), (T, "out1", "ls"), (A, "edit", None), (T, "out2", "edit")],
             modality=Modality.AGENTIC)
    segs = list(segment.segment(c))
    assert [s.segment_id for s in segs] == ["c1:0", "c1:2"]
    assert segs[1].text.startswith("TOOL (ls): out1")  # observation -> action
    # trailing tool output with no assistant answer is not a segment


def test_segment_cap_and_length():
    turns = [x for i in range(100) for x in ((U, f"q{i}", None), (A, "a" * 5000, None))]
    segs = list(segment.segment(conv(turns), max_tokens=100, max_segments=10))
    assert len(segs) == 10 and segs[0].segment_id == "c1:0" and segs[-1].segment_id == "c1:198"
    assert all(len(s.text) < 500 for s in segs)


# --------------------------------------------------------------------------- redact


def test_redact_secrets_and_personal_data():
    text = (
        "email me at jane.doe@corp.com, key sk-ant-abcdefghijklmnopqrstuvwx and ghp_" + "a" * 36 + "\n"
        "password = hunter2secret\nAuthorization: Bearer abcdefghijklmnop1234\n"
        "postgres://admin:s3cret@db.internal:5432/app from 10.1.2.3, call +44 20 7946 0958\n"
        "file /Users/janedoe/work/app.py, reviewed by Kasun"
    )
    out, counts = redact.redact_text(text, ["Kasun"])
    for leak in ("jane.doe@corp.com", "sk-ant-", "ghp_", "hunter2", "abcdefghijklmnop1234", "s3cret",
                 "10.1.2.3", "7946", "janedoe", "Kasun"):
        assert leak not in out, leak
    assert "/Users/[USER]/work/app.py" in out and "[NAME]" in out
    assert counts["email"] == 1 and counts["extra_term"] == 1


def test_redact_leaves_normal_code_alone():
    code = "def add(a, b):\n    return a + b  # version 3.11\n"
    assert redact.redact_text(code)[0] == code


def test_redact_conversation_marks_metadata():
    c = conv([(U, "my key is sk-proj-abcdefghijklmnopqrstuvwxyz", None), (A, "ok", None)])
    out, counts = redact.redact_conversation(c)
    assert "[SECRET]" in out.turns[0].content and out.metadata["redacted"] is True and counts["api_key"] == 1


# --------------------------------------------------------------------------- precision check


def test_precision_sample_and_score(tmp_path):
    path = tmp_path / "wildchat.jsonl"
    write_jsonl(path, [conv(CODE_CHAT, cid=f"c{i}") for i in range(30)])
    out = tmp_path / "check.csv"
    assert precision_check.export_sample(path, out, n=10) == 10
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10 and rows[0]["is_relevant"] == ""
    for i, r in enumerate(rows):
        r["is_relevant"] = "y" if i < 8 else "n"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p = precision_check.precision(out)
    assert p["checked"] == 10 and p["precision"] == pytest.approx(0.8)
    assert p["ci95_low"] < 0.8 < p["ci95_high"]


# --------------------------------------------------------------------------- pipeline


def test_run_cleaning_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    raw = tmp_path / "raw"
    write_jsonl(raw / "wildchat.jsonl", [
        conv(CODE_CHAT, cid="w1", language="en"),
        conv(CODE_CHAT, cid="w2", language="en"),  # exact duplicate
        conv(RECIPE_CHAT, cid="w3", language="en"),  # not relevant
        conv(BA_CHAT, cid="w4", language="en"),
        conv(CODE_CHAT, cid="w5", language="russian"),
    ])
    write_jsonl(raw / "own_logs.jsonl", [
        conv([(U, "deploy with token ghp_" + "b" * 36, None), (A, "done", None)], cid="o1",
             source=SourceDataset.OWN_LOGS, modality=Modality.AGENTIC, author_id="p01"),
    ])
    report = run_cleaning("clean")
    assert report["wildchat"] == {"input": 5, "kept": 2, "dropped_near_duplicate": 1, "dropped_relevance": 1,
                                  "dropped_language": 1, "segments": 4}
    assert report["own_logs"]["redactions"] == 1 and report["own_logs"]["kept"] == 1
    own = list(read_jsonl(tmp_path / "filtered" / "own_logs.jsonl", Conversation))
    assert "ghp_" not in own[0].model_dump_json()
    segs = list(read_jsonl(tmp_path / "segments" / "wildchat.jsonl", Segment))
    assert {s.conversation_id for s in segs} == {"w1", "w4"}
    assert json.loads((tmp_path / "filtered" / "cleaning_report.json").read_text())["wildchat"]["kept"] == 2
