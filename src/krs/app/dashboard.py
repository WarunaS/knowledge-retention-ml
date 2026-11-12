"""Streamlit demo (optional): one screen for the whole system.

    streamlit run src/krs/app/dashboard.py        # http://localhost:8501

Tabs: Ask (ask-your-predecessor), Handover pack, Knowledge map, Model results.
Read-only: it never writes to the store, the results or the registry. People are never ranked.
Answers default to the local backend (free, offline); Claude is used only if chosen and allowed.
"""

from __future__ import annotations

import os

import streamlit as st

from krs.app.agent import ask
from krs.app.common import app_config
from krs.app.handover import build_pack
from krs.app.knowledge_map import coverage, model_results, single_person_dependency, staleness
from krs.app.storage import KnowledgeStore

st.set_page_config(page_title="Knowledge Retention System", layout="wide")


@st.cache_resource
def get_store() -> KnowledgeStore:
    return KnowledgeStore()


store = get_store()
cards = store.all()
projects = sorted({c.project_id for c in cards if c.project_id})
roles = sorted({c.role.value for c in cards if c.role})

st.title("Knowledge Retention System")
st.caption(f"{len(cards)} knowledge cards · describes knowledge, never people")
if not cards:
    st.info("The knowledge store is empty. Build it in notebook 15 (extract cards, then store.add).")

ask_tab, pack_tab, map_tab, results_tab = st.tabs(["Ask", "Handover pack", "Knowledge map", "Model results"])

with ask_tab:
    question = st.text_input("Ask your predecessor", placeholder="Why did we choose PostgreSQL for orders?")
    c1, c2, c3 = st.columns(3)
    project = c1.selectbox("Project", ["(any)", *projects])
    role = c2.selectbox("Role", ["(any)", *roles])
    backend = c3.radio("Answer with", ["local (free, offline)", "Claude (needs API key)"], horizontal=False)
    if st.button("Ask", type="primary") and question:
        cfg = app_config()
        use_claude = backend.startswith("Claude") and bool(os.environ.get("ANTHROPIC_API_KEY"))
        if backend.startswith("Claude") and not use_claude:
            st.warning("No ANTHROPIC_API_KEY set - answering locally.")
        cfg = {**cfg, "answer": {**(cfg.get("answer") or {}), "backend": "claude" if use_claude else "local"}}
        filters = {k: v for k, v in {"project_id": project, "role": role}.items() if v != "(any)"}
        r = ask(store, question, filters, cfg=cfg)
        st.markdown(r["answer"])
        if r["note"]:
            st.info(r["note"])
        if r["unknown_citations"]:
            st.warning(f"Citations not among the retrieved cards: {', '.join(r['unknown_citations'])}")
        for c in r["cards"]:
            card = store.get(c["card_id"])
            with st.expander(f"{card.title}  ·  score {c['score']:.2f} ({c['via']})"):
                st.write(card.summary)
                if card.decision:
                    st.write(f"**Decision:** {card.decision}")
                st.caption(f"Evidence: “{card.evidence}”")
                st.caption(f"Card {card.card_id} · conversation {card.conversation_id} · {', '.join(card.source_segment_ids)}")

with pack_tab:
    c1, c2 = st.columns(2)
    p_project = c1.selectbox("Project", ["(none)", *projects], key="pack_project")
    p_person = c2.text_input("Person (pseudonym, optional)", key="pack_person")
    if p_project != "(none)" or p_person:
        md = build_pack(store, author_id=p_person or None, project_id=None if p_project == "(none)" else p_project)
        st.download_button("Download pack (Markdown)", md, file_name="handover_pack.md")
        st.markdown(md)
    else:
        st.write("Choose a project or enter a person to build a handover pack.")

with map_tab:
    st.subheader("Coverage: cards per domain and knowledge type")
    cov = coverage(cards)
    st.bar_chart(cov.drop(columns="total"))
    st.dataframe(cov)
    by = st.radio("Group areas by", ["domain", "project"], horizontal=True)
    st.subheader("Areas that depend on one person")
    st.caption("Share of an area's cards held by its largest contributor. Contributors are not named or ranked.")
    st.dataframe(single_person_dependency(cards, by=by))
    st.subheader("Areas with no recent knowledge")
    st.dataframe(staleness(cards, by=by))

with results_tab:
    res = model_results()
    if "results_table" not in res and "model_card" not in res:
        st.write("No model results yet - run notebooks 08, 09 and 12.")
    if "results_table" in res:
        st.subheader("Gold-set results (95% CIs)")
        st.dataframe(res["results_table"])
    if res.get("comparisons"):
        st.subheader("Hypotheses")
        st.json(res["comparisons"], expanded=False)
    if "learning_curve" in res:
        st.image(str(res["learning_curve"]), caption="Learning curve (A2)")
    for img in res.get("confusions", []):
        st.image(str(img), caption=img.stem)
    if "transfer" in res:
        st.subheader("Transfer test (X1)")
        st.json(res["transfer"])
    if "retrieval" in res:
        st.subheader("Retrieval quality")
        st.json(res["retrieval"], expanded=False)
    if "model_card" in res:
        st.subheader(f"Model card - {res['model_version']}")
        st.markdown(res["model_card"])
