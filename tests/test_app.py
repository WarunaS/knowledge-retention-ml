"""Application plane tests: extraction + faithfulness, store, retrieval, handover, agent, REST API, MCP.
Embedder and Claude are faked: nothing is downloaded or sent. Privacy rules are tested explicitly."""

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import numpy as np
import pytest

from krs.app import agent, extraction, handover, mcp_server, retrieval
from krs.app.common import Embedder, cloud_allowed
from krs.app.storage import KnowledgeStore
from krs.io import read_jsonl, write_jsonl
from krs.label.taxonomy import Domain, KnowledgeType
from krs.schema import KnowledgeCard, Modality, PersonRole, Prediction, Segment, SourceDataset

CFG = {"cloud_sources": ["wildchat", "devgpt", "swe_agent", "openhands"],
       "extraction": {"backend": "claude", "workers": 2, "faithfulness_threshold": 0.3},
       "answer": {"backend": "claude"}, "retrieval": {"k": 3, "expand": True}}


class BowModel:
    """Bag-of-words hashing 'embedding': texts sharing words are close."""

    def encode(self, texts, **kw):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]{3,}", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        return out


def emb():
    return Embedder(model=BowModel())


class FakeClaude:
    """Records every request. Extraction: returns a card, or a hallucinated one for segments saying 'invent'.
    Answering: cites the first card id it sees, plus one made-up id."""

    def __init__(self):
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **params):
        content = params["messages"][0]["content"]
        self.calls.append(content)
        if "output_config" in params and "format" in params["output_config"]:
            seg = content.split("<segment>")[1]
            first_sentence = re.split(r"(?<=[.!?])\s+|\n", seg.split("ASSISTANT:")[1].strip())[0]
            queue = "queue" in seg
            body = {"title": "Queue for order emails" if queue else "From the segment",
                    "summary": "Use a queue for order emails because the mail service times out." if queue else first_sentence,
                    "decision": "Chose a queue over inline calls because the mail service times out." if queue else "",
                    "tools_involved": ["rabbitmq"] if queue else [], "evidence": first_sentence}
            if "invent" in seg:
                body["evidence"] = "This sentence was never said."
            text = json.dumps(body)
        else:
            ids = re.findall(r"\[(k[0-9a-f]{12})\]", content)
            text = f"Use the queue [{ids[0]}]. Also see [k000000000000]." if ids else "Nothing found."
        return NS(content=[NS(type="text", text=text)], stop_reason="end_turn", model="claude-opus-5-5")


def seg(sid, conv, text, source=SourceDataset.WILDCHAT, **kw):
    return Segment(segment_id=sid, conversation_id=conv, source_dataset=source, modality=Modality.CONVERSATIONAL,
                   turn_span=(0, 1), text=text, **kw)


def pred(sid, kts, doms, bearing=True):
    return Prediction(segment_id=sid, model_version="b2-v1", run_id="B2",
                      knowledge_type_probs={k: (0.9 if k.value in kts else 0.1) for k in KnowledgeType},
                      domain_probs={d: (0.9 if d.value in doms else 0.1) for d in Domain}, knowledge_bearing=bearing,
                      knowledge_types=[KnowledgeType(k) for k in kts], domains=[Domain(d) for d in doms])


QUEUE = "USER: Should order emails be sent inline?\nASSISTANT: Use a queue rather than inline calls because the mail service times out under load. Then retry failed messages with `rabbitmq`."
DEPLOY = "USER: How do we deploy?\nTOOL (bash): build ok\nASSISTANT: First build the docker image with the release script. Then push it to the registry and restart the service."
OWN = "USER: Our staging database copies production nightly.\nASSISTANT: Then never test destructive migrations on staging because it holds production data copies."


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path / "data"))
    t = lambda d: datetime(2026, 3, d, tzinfo=timezone.utc)
    segs = [
        seg("wildchat:a:0", "wildchat:a", QUEUE, project_id="orders", role=PersonRole.TECH_LEAD, author_id="p01", occurred_at=t(12)),
        seg("wildchat:a:2", "wildchat:a", DEPLOY, project_id="orders", role=PersonRole.TECH_LEAD, author_id="p01", occurred_at=t(14)),
        seg("wildchat:b:0", "wildchat:b", "USER: hi\nASSISTANT: invent something about caches here please now."),
        seg("own_logs:p02:s1:0", "own_logs:p02:s1", OWN, SourceDataset.OWN_LOGS, project_id="orders",
            role=PersonRole.QA_ENGINEER, author_id="p02", occurred_at=t(10)),
        seg("wildchat:c:0", "wildchat:c", "USER: thanks\nASSISTANT: You are welcome, glad it helped you today."),
    ]
    write_jsonl(tmp_path / "data" / "segments" / "all.jsonl", segs)
    preds = [pred("wildchat:a:0", ["decision"], ["architecture_design"]),
             pred("wildchat:a:2", ["procedural"], ["devops_cicd"]),
             pred("wildchat:b:0", ["domain"], ["backend_apis"]),
             pred("own_logs:p02:s1:0", ["contextual"], ["qa_testing"]),
             pred("wildchat:c:0", ["none"], ["other"], bearing=False)]
    write_jsonl(tmp_path / "preds.jsonl", preds)
    return tmp_path


# --------------------------------------------------------------------------- extraction


def test_privacy_rule():
    assert cloud_allowed("wildchat:x", CFG) and not cloud_allowed("own_logs:p01:s", CFG)
    assert not cloud_allowed("own_logs:p01:s", {"cloud_sources": ["own_logs"]})  # never, even if misconfigured


def test_faithfulness_rules():
    s = seg("x:0", "x", QUEUE)
    ok, score, _ = extraction.faithfulness({"evidence": "use a queue rather than   inline calls",
                                            "summary": "Use a queue rather than inline calls."}, s)
    assert ok and score > 0.8
    ok, _, reason = extraction.faithfulness({"evidence": "never said", "summary": "x"}, s)
    assert not ok and "verbatim" in reason


def test_local_card_is_extractive():
    f = extraction.local_card_fields(seg("x:0", "x", QUEUE), [KnowledgeType.DECISION])
    assert "rather than" in f["decision"] and f["tools_involved"] == ["rabbitmq"]
    assert f["evidence"] in QUEUE and f["title"].startswith("Should order emails")
    assert extraction.local_card_fields(seg("y:0", "y", DEPLOY), [KnowledgeType.PROCEDURAL])["tools_involved"] == ["bash"]
    agent_step = "TOOL (bash): <uploaded_files>\n/workspace/repo\nASSISTANT: The parser fails on empty input, so add a guard first."
    assert extraction.local_card_fields(seg("z:0", "z", agent_step), [])["title"].startswith("The parser fails")


def test_decision_without_statement_keeps_other_knowledge():
    s = seg("d:0", "d", DEPLOY)
    fields = extraction.local_card_fields(s, [KnowledgeType.PROCEDURAL])
    card = extraction.build_card(s, pred("d:0", ["decision", "procedural"], ["devops_cicd"]), fields, 0.9, "local")
    assert card.knowledge_types == [KnowledgeType.PROCEDURAL] and card.decision is None
    with pytest.raises(ValueError, match="only a decision"):
        extraction.build_card(s, pred("d:0", ["decision"], ["devops_cicd"]), fields, 0.9, "local")


def test_extract_cards_backends_rejections_resume(world):
    client = FakeClaude()
    out = world / "cards.jsonl"
    rep = extraction.extract_cards(world / "preds.jsonl", out, CFG, client=client)
    assert rep == {"cards": 3, "rejected": 1, "local": 1, "claude": 2, "skipped_already_done": 0}
    assert not any("staging" in c for c in client.calls)  # own-log text never reached the cloud model
    cards = {c.source_segment_ids[0]: c for c in read_jsonl(out, KnowledgeCard)}
    assert cards["wildchat:a:0"].decision and cards["wildchat:a:0"].role == PersonRole.TECH_LEAD
    assert cards["own_logs:p02:s1:0"].model_version.endswith("+local") and cards["own_logs:p02:s1:0"].author_id == "p02"
    rejected = [json.loads(line) for line in (world / "cards_rejected.jsonl").read_text().splitlines()]
    assert rejected[0]["segment_id"] == "wildchat:b:0" and "verbatim" in rejected[0]["reason"]
    again = extraction.extract_cards(world / "preds.jsonl", out, CFG, client=FakeClaude())
    assert again["skipped_already_done"] == 3 and again["cards"] == 0


# --------------------------------------------------------------------------- store, retrieval, handover


@pytest.fixture
def store(world):
    extraction.extract_cards(world / "preds.jsonl", world / "cards.jsonl", CFG, client=FakeClaude())
    s = KnowledgeStore(world / "store", embedder=emb())
    s.add(read_jsonl(world / "cards.jsonl", KnowledgeCard))
    return s


def test_store_graph_and_reopen(store, world):
    assert store.count() == 3 and len(store.ids) == 3
    a0, a2 = extraction.card_id("wildchat:a:0"), extraction.card_id("wildchat:a:2")
    assert store.graph.has_edge(a0, a2) and store.graph.edges[a0, a2]["rel"] == "follows"
    assert a2 in store.neighbours(a0)
    reopened = KnowledgeStore(world / "store", embedder=emb())
    assert reopened.count() == 3 and reopened.ids == store.ids and (world / "store" / "graph.json").exists()


def test_delete_person_everywhere(store):
    own = extraction.card_id("own_logs:p02:s1:0")
    assert store.delete_person("p02") == 1
    assert store.get(own) is None and own not in store.ids and own not in store.graph and store.count() == 2


def test_search_ranking_filters_and_expansion(store):
    hits = retrieval.search(store, "why a queue for order emails mail service", k=1)
    assert hits[0].card.source_segment_ids == ["wildchat:a:0"] and hits[0].via == "semantic"
    assert retrieval.search(store, "anything", filters={"role": "qa_engineer"})[0].card.author_id == "p02"
    assert {h.card.card_id for h in retrieval.search(store, "x", k=5, filters={"knowledge_types": ["procedural"]})} == {
        extraction.card_id("wildchat:a:2")}
    assert retrieval.search(store, "x", filters={"since": "2026-03-11", "until": "2026-03-13"})[0].card.source_segment_ids == ["wildchat:a:0"]
    assert retrieval.search(store, "x", filters={"project_id": "nope"}) == []
    expanded = retrieval.search(store, "why a queue for order emails mail service", k=1, expand=True)
    assert any(h.via.startswith("graph:") for h in expanded)


def test_handover_pack(store):
    md = handover.build_pack(store, project_id="orders")
    assert md.startswith("# Handover pack - project `orders`") and "## Decision log (1)" in md
    assert "Chose a queue" in md and "## Procedures (1)" in md and "Source: card `" in md
    assert md.index("## Decision log") < md.index("## Procedures") < md.index("## Context and constraints")
    assert "staging" not in handover.build_pack(store, author_id="p01")
    from krs.app.common import content_words
    assert "calls" in content_words("inline calls.") and "app.py" in content_words("see app.py")
    with pytest.raises(ValueError):
        handover.build_pack(store)


# --------------------------------------------------------------------------- agent, REST API, MCP


def test_ask_claude_with_citation_check(store):
    client = FakeClaude()
    r = agent.ask(store, "why a queue for order emails", {"project_id": "orders", "role": "tech_lead"}, cfg=CFG, client=client)
    assert r["backend"] == "claude" and r["citations"] and r["unknown_citations"] == ["k000000000000"]
    assert "staging" not in client.calls[0]


def test_ask_falls_back_to_local_for_own_logs(store):
    client = FakeClaude()
    r = agent.ask(store, "staging database migrations", cfg=CFG, client=client)
    assert r["backend"] == "local" and "own logs" in r["note"] and client.calls == []
    assert agent.ask(store, "x", {"project_id": "nope"}, cfg=CFG, client=client)["backend"] == "none"


def test_rest_api(store):
    from fastapi.testclient import TestClient

    api = TestClient(agent.create_app(lambda: store, client=FakeClaude()))
    assert api.get("/health").json()["cards"] == 3
    hits = api.post("/search", json={"query": "queue order emails", "k": 1}).json()
    assert hits[0]["card"]["project_id"] == "orders"
    assert api.post("/ask", json={"question": "deploy steps", "filters": {"role": "tech_lead"}}).json()["citations"]
    assert api.get("/cards/knope").status_code == 404
    assert api.get("/handover").status_code == 400
    assert api.get("/handover", params={"project_id": "orders"}).text.startswith("# Handover pack")


def test_mcp_tools(store):
    server = mcp_server.build_server(lambda: store, client=FakeClaude())
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"ask_predecessor", "search_knowledge", "handover_pack"}
    res = asyncio.run(server.call_tool("search_knowledge", {"query": "queue order emails", "k": 1}))
    assert not res.is_error and "queue" in res.content[0].text.lower()
    res = asyncio.run(server.call_tool("handover_pack", {"project_id": "orders"}))
    assert "Decision log" in res.content[0].text


def test_ask_without_key_or_with_failing_claude_falls_back(store, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    r = agent.ask(store, "why a queue for order emails", {"project_id": "orders", "role": "tech_lead"}, cfg=CFG)
    assert r["backend"] == "local" and "No Anthropic API key" in r["note"]

    class Broken:
        beta = NS(messages=NS(create=lambda **kw: (_ for _ in ()).throw(ConnectionError("down"))))

    r = agent.ask(store, "why a queue for order emails", {"project_id": "orders", "role": "tech_lead"}, cfg=CFG, client=Broken())
    assert r["backend"] == "local" and "ConnectionError" in r["note"] and r["answer"].startswith("Most relevant")


def test_extraction_without_key_fails_clearly(world, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        extraction.extract_cards(world / "preds.jsonl", world / "cards.jsonl", CFG)


def test_expansion_is_capped_and_skips_hub_tools(tmp_path):
    from krs.app.extraction import card_id

    s = KnowledgeStore(tmp_path / "hub", embedder=emb())
    cards = []
    for i in range(30):  # 30 cards all using 'bash': a hub tool that links everything
        cards.append(KnowledgeCard(card_id=card_id(f"h{i}"), source_segment_ids=[f"h{i}"], conversation_id=f"wildchat:h{i}",
                                   domains=[Domain.QA_TESTING], knowledge_types=[KnowledgeType.PROCEDURAL],
                                   title=f"step {i}", summary=f"run tests number {i}", tools_involved=["bash"],
                                   evidence="x", confidence=0.9, model_version="t"))
    s.add(cards)
    assert s.neighbours(cards[0].card_id) == set()  # hub tool skipped
    hits = retrieval.search(s, "run tests", k=3, expand=True)
    assert len(hits) <= 6
