"""App evaluation tests: retrieval metrics and modes, question and card-check sheets, successor task."""

import csv
import hashlib
import re

import numpy as np
import pytest

from krs.app.common import Embedder
from krs.app.storage import KnowledgeStore
from krs.evaluate import retrieval_eval as RE
from krs.evaluate import successor as S
from krs.label.taxonomy import Domain, KnowledgeType
from krs.schema import KnowledgeCard, PersonRole


class BowModel:
    def encode(self, texts, **kw):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]{3,}", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        return out


def card(cid, title, summary, project, role=PersonRole.SOFTWARE_ENGINEER):
    return KnowledgeCard(card_id=cid, source_segment_ids=[f"{cid}:0"], conversation_id=f"wildchat:{cid}",
                         domains=[Domain.DEVOPS_CICD], knowledge_types=[KnowledgeType.PROCEDURAL], title=title,
                         summary=summary, evidence=summary, confidence=0.9, model_version="t", project_id=project, role=role)


@pytest.fixture
def store(tmp_path):
    s = KnowledgeStore(tmp_path / "store", embedder=Embedder(model=BowModel()))
    s.add([
        card("ka00000000001", "Deploy the service", "Deploy the service with docker compose.", "alpha"),
        # same procedure in another project, worded closer to the question: plain search prefers it
        card("kb00000000002", "How we deploy the service", "How we deploy the service with docker compose quickly.", "beta"),
        card("kc00000000003", "Rotate API keys", "Rotate keys in the console then update CI secrets.", "alpha", PersonRole.TECH_LEAD),
    ])
    return s


def test_metric_functions():
    assert RE.rank_of_first_relevant(["a", "b", "c"], {"c"}) == 3
    assert RE.rank_of_first_relevant(["a"], {"z"}) is None
    assert RE.recall_at(["a", "b", "c"], {"b", "z"}, 2) == 0.5


def test_filters_help_when_projects_collide(store, tmp_path):
    queries = [RE.Query("q1", "how do we deploy the service quickly", ["ka00000000001"], {"project_id": "alpha"}),
               RE.Query("q2", "rotate api keys", ["kc00000000003"], {"role": "tech_lead"})]
    rep = RE.evaluate_retrieval(store, queries, ks=(1, 3), n_boot=200, out=tmp_path / "ev")
    m = rep["modes"]
    assert m["filtered"]["mrr"] == 1.0 and m["semantic"]["mrr"] == pytest.approx(0.75)
    assert m["filtered"]["recall@1"] == 1.0 and m["semantic"]["recall@1"] == 0.5
    assert rep["filters_help"]["mrr_diff"] == pytest.approx(0.25) and "filtered+graph" in m
    assert (tmp_path / "ev" / "retrieval.json").exists()
    with pytest.raises(ValueError, match="no usable questions"):
        RE.evaluate_retrieval(store, [])


def test_query_sheet_round_trip(store, tmp_path):
    sheet = tmp_path / "queries.csv"
    assert RE.export_query_sheet(store, n=3, seed=1, out_csv=sheet) == 3
    with open(sheet, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["question"] == "" and rows[0]["relevant_card_ids"].startswith("k") and "project_id" in rows[0]["filters_json"]
    rows[0]["question"] = "how do we ship it"
    rows[1]["question"] = "key rotation steps"
    rows[1]["relevant_card_ids"] = rows[1]["relevant_card_ids"] + ", kb00000000002"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    qs = RE.load_queries(sheet)
    assert [q.query_id for q in qs] == [rows[0]["query_id"], rows[1]["query_id"]]  # row without a question skipped
    assert len(qs[1].relevant) == 2 and isinstance(qs[0].filters, dict)


def test_card_check(store, tmp_path):
    sheet = tmp_path / "check.csv"
    assert RE.export_card_check(store, n=3, out_csv=sheet) == 3
    with open(sheet, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r, mark in zip(rows, ["y", "y", "n"], strict=True):
        r["faithful"] = mark
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rate = RE.card_check_rate(sheet)
    assert rate["checked"] == 3 and rate["rate"] == pytest.approx(0.667) and rate["ci95_low"] < 0.667


# --------------------------------------------------------------------------- successor task


def test_assignment_is_counterbalanced():
    qs = [f"q{i}" for i in range(6)]
    plan = S.assign(qs, ["P1", "P2", "P3", "P4"], seed=3)
    for conds in plan.values():
        assert sorted(conds.values()).count("with_cards") == 3
    for q in qs:
        assert {plan["P1"][q], plan["P2"][q]} == {"with_cards", "without_cards"}


def test_sheets_blind_rating_and_analysis(tmp_path):
    questions = {f"q{i}": f"Question {i}?" for i in range(4)}
    paths = S.export_participant_sheets(questions, ["P1", "P2", "P3"], root=tmp_path)
    for p, path in paths.items():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            good = r["condition"] == "with_cards"
            r["answer"] = f"{p} answer to {r['question_id']}"
            r["minutes"] = "4" if good else "9"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    responses = S.read_responses(list(paths.values()))
    sheet, key = S.export_rating_sheet(responses, root=tmp_path)
    with open(sheet, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "condition" not in header and "participant" not in header  # blind
    with open(key, newline="", encoding="utf-8") as f:
        cond = {r["answer_id"]: r["condition"] for r in csv.DictReader(f)}
    with open(sheet, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["answer_id"] for r in rows] == sorted(r["answer_id"] for r in rows)
    for r in rows:
        r["rating_1_to_5"] = "5" if cond[r["answer_id"]] == "with_cards" else "2"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = S.analyse(sheet, key)
    assert res["participants"] == 3 and res["ratings"] == 12 and res["note"].startswith("indicative")
    assert res["rating"]["mean_diff_with_minus_without"] == 3.0 and res["minutes"]["mean_diff_with_minus_without"] == -5.0
    assert "wilcoxon_p" in res["rating"]
