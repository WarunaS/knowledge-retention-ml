"""Knowledge map and dashboard tests. The dashboard runs headless (Streamlit AppTest) on a tiny store."""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from krs.app import knowledge_map as KM
from krs.app.common import Embedder
from krs.app.storage import KnowledgeStore
from krs.label.taxonomy import Domain, KnowledgeType
from krs.schema import KnowledgeCard, PersonRole

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "krs" / "app" / "dashboard.py"


class BowModel:
    def encode(self, texts, **kw):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]{3,}", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return out


def card(i, author, domain, kt="procedural", project="orders", month=3):
    return KnowledgeCard(card_id=f"k{i:012d}", source_segment_ids=[f"s{i}:0"], conversation_id=f"wildchat:c{i}",
                         domains=[Domain(domain)], knowledge_types=[KnowledgeType(kt)], title=f"Card {i} about {domain}",
                         summary=f"How to handle {domain} task number {i}.", evidence="x", confidence=0.9, model_version="t",
                         decision="Chose option A over B because it scales." if kt == "decision" else None,
                         author_id=author, role=PersonRole.SOFTWARE_ENGINEER, project_id=project,
                         occurred_at=datetime(2026, month, 1, tzinfo=timezone.utc))


CARDS = ([card(i, "p01", "security") for i in range(6)]                                   # one person holds security
         + [card(10 + i, f"p0{2 + i % 3}", "backend_apis", "decision") for i in range(6)]  # shared across 3 people
         + [card(20, "p02", "qa_testing", month=10)])


def test_coverage_counts():
    cov = KM.coverage(CARDS)
    assert cov.loc["security", "procedural"] == 6 and cov.loc["backend_apis", "decision"] == 6
    assert "none" not in cov.columns and cov["total"].sum() == 13


def test_single_person_dependency_never_names_people():
    dep = KM.single_person_dependency(CARDS, by="domain", min_cards=5, threshold=0.8)
    sec = dep[dep["domain"] == "security"].iloc[0]
    assert sec["at_risk"] and sec["largest_share"] == 1.0 and sec["contributors"] == 1
    assert not dep[dep["domain"] == "backend_apis"].iloc[0]["at_risk"]
    text = dep.to_csv()
    for pseudonym in ("p01", "p02", "p03", "p04"):
        assert pseudonym not in text  # contributors are never named
    assert list(dep.columns) == ["domain", "cards", "contributors", "largest_share", "at_risk"]


def test_staleness():
    st = KM.staleness(CARDS, by="domain", months=4, now=datetime(2026, 11, 1, tzinfo=timezone.utc))
    stale = dict(zip(st["domain"], st["stale"], strict=True))
    assert stale["security"] and not stale["qa_testing"]


def test_model_results_tolerates_missing(tmp_path):
    assert KM.model_results(tmp_path / "res", tmp_path / "models") == {"confusions": []}


@pytest.fixture
def filled_store(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    store = KnowledgeStore(tmp_path / "store", embedder=Embedder(model=BowModel()))
    store.add(CARDS)
    return store


def _app(monkeypatch, store):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import krs.app.storage as storage_mod

    st.cache_resource.clear()  # the dashboard caches its store; each test starts fresh
    # the dashboard does `from krs.app.storage import KnowledgeStore` and calls it: hand it the test store
    monkeypatch.setattr(storage_mod, "KnowledgeStore", lambda *a, **k: store)
    return AppTest.from_file(str(DASHBOARD), default_timeout=60)


def test_dashboard_runs_and_answers_locally(filled_store, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = _app(monkeypatch, filled_store).run()
    assert not at.exception
    assert "13 knowledge cards" in at.caption[0].value
    at.text_input[0].input("how to handle security task").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Most relevant knowledge" in m.value for m in at.markdown)  # local backend, no API call
    at.selectbox(key="pack_project").select("orders").run()
    assert not at.exception and any(m.value.startswith("# Handover pack") for m in at.markdown)


def test_dashboard_empty_store(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    empty = KnowledgeStore(tmp_path / "store", embedder=Embedder(model=BowModel()))
    at = _app(monkeypatch, empty).run()
    assert not at.exception and any("empty" in i.value for i in at.info)
