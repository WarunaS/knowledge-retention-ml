"""X1 tests: labelling own logs, the run-once lock, drift measures, no text in the report."""

import csv
import json

import numpy as np
import pytest

from krs import registry
from krs.evaluate import stats, transfer
from krs.evaluate.metrics import evaluate_runs
from krs.features import build_features
from krs.inference import load_predictor
from krs.io import read_jsonl, write_jsonl
from krs.label.taxonomy import Domain, KnowledgeType
from krs.models.baselines import train_baseline
from krs.schema import Label, LabelledSegment, LabelSource, Modality, Segment, SourceDataset, Split

VOCAB = {("decision", "architecture_design"): "we chose queue rather than inline because tradeoff",
         ("procedural", "devops_cicd"): "first build docker image then push pipeline",
         ("none", "other"): "thanks ok great cheers"}
# own logs say the same things in other words: a model trained on the public wording should drop
SHIFTED = {("decision", "architecture_design"): "opted for broker over direct call given throughput",
           ("procedural", "devops_cicd"): "containerise app and ship via release workflow",
           ("none", "other"): "thanks ok great cheers"}


def _rows(n, split, prefix, source, vocab=VOCAB):
    out, i = [], 0
    modality = Modality.AGENTIC if source in (SourceDataset.SWE_AGENT, SourceDataset.OWN_LOGS) else Modality.CONVERSATIONAL
    for (kt, dom), words in vocab.items():
        for _ in range(n):
            sid = f"{prefix}{i}:0"
            seg = Segment(segment_id=sid, conversation_id=f"{prefix}{i}", source_dataset=source, modality=modality,
                          turn_span=(0, 1), text=f"USER: {words} {i}", split=split)
            out.append(LabelledSegment(segment=seg, label=Label(segment_id=sid, knowledge_types=[KnowledgeType(kt)],
                                                                 domains=[Domain(dom)], source=LabelSource.LLM)))
            i += 1
    return out


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path / "data"))
    sp = tmp_path / "data" / "splits"
    W, S, O = SourceDataset.WILDCHAT, SourceDataset.SWE_AGENT, SourceDataset.OWN_LOGS
    write_jsonl(sp / "train.jsonl", _rows(10, Split.TRAIN, "t", W) + _rows(10, Split.TRAIN, "ta", S))
    write_jsonl(sp / "validation.jsonl", _rows(4, Split.VALIDATION, "v", W))
    write_jsonl(sp / "test.jsonl", _rows(4, Split.TEST, "x", W) + _rows(4, Split.TEST, "xa", S))
    write_jsonl(sp / "transfer.jsonl", _rows(5, Split.TRANSFER, "own", O, SHIFTED))
    f = tmp_path / "f.yaml"
    f.write_text("with_context: [true]\ntfidf:\n  min_df: 1\nembeddings:\n  enabled: false\ntokenizer:\n  enabled: false\n")
    build_features(f)
    res, models = tmp_path / "results", tmp_path / "models"
    train_baseline({"run_id": "B1", "model": "tfidf_lr", "C_grid": [10.0]}, results_root=res)
    e = tmp_path / "e.yaml"
    e.write_text("split: test\nruns: [B1]\nbootstrap: {n: 50, seed: 1}\n")
    evaluate_runs(e, results_root=res)
    registry.register("B1", results_root=res, models_dir=models)
    cfg = tmp_path / "x1.yaml"
    cfg.write_text("run_id: X1\nbootstrap_n: 200\nseed: 1\n")
    return tmp_path, res, models, cfg


def test_x1_report_drop_and_no_text(world):
    _, res, models, cfg = world
    rep = transfer.run(cfg, results_root=res, predictor=load_predictor(models_dir=models))
    kt = rep["knowledge_types"]
    assert rep["transfer_segments"] == 15 and rep["gold_agentic_segments"] == 12 and rep["runs_so_far"] == 1
    assert kt["gold"]["macro_f1"] > kt["transfer"]["macro_f1"]  # shifted wording: the model drops
    assert kt["drop_gold_minus_transfer"]["diff"] > 0 and "drop_gold_agentic_minus_transfer" in kt
    assert set(kt["label_share_transfer"]) == {k.value for k in KnowledgeType}
    text = (res / "X1" / "report.json").read_text()
    for words in ("broker", "queue", "docker", "USER:"):
        assert words not in text  # ids and numbers only
    assert np.load(res / "X1" / "predictions_transfer.npz")["knowledge_types"].shape[0] == 15


def test_x1_runs_once(world):
    _, res, models, cfg = world
    pred = load_predictor(models_dir=models)
    transfer.run(cfg, results_root=res, predictor=pred)
    with pytest.raises(transfer.TransferLockError, match="runs once"):
        transfer.run(cfg, results_root=res, predictor=pred)
    with pytest.raises(transfer.TransferLockError, match="reason"):
        transfer.run(cfg, force=True, results_root=res, predictor=pred)
    rep = transfer.run(cfg, force=True, reason="fixed a bug in redaction", results_root=res, predictor=pred)
    lock = json.loads((res / "X1" / "x1_lock.json").read_text())
    assert rep["runs_so_far"] == 2 and lock["runs"][1] == {**lock["runs"][1], "forced": True,
                                                           "reason": "fixed a bug in redaction"}


def test_x1_needs_transfer_split(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    cfg = tmp_path / "x1.yaml"
    cfg.write_text("run_id: X1\n")
    with pytest.raises(FileNotFoundError, match="transfer"):
        transfer.run(cfg, results_root=tmp_path / "res", predictor=object())


def test_transfer_sheet_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="own logs"):
        transfer.export_transfer_sheet(5)
    segs = [r.segment for r in _rows(3, Split.TRANSFER, "own", SourceDataset.OWN_LOGS)]
    write_jsonl(tmp_path / "segments" / "own_logs.jsonl", segs)
    assert transfer.export_transfer_sheet(5, seed=1) == 5
    sheet = tmp_path / "labels" / "transfer_sheet.csv"
    with open(sheet, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["k:procedural"], r["d:devops_cicd"] = "x", "x"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n, problems = transfer.import_transfer_sheet()
    assert n == 5 and not problems
    labs = list(read_jsonl(tmp_path / "labels" / "transfer.jsonl", Label))
    assert all(lab.source == LabelSource.RATER_1 for lab in labs)


def test_unpaired_bootstrap_diff():
    rng = np.random.default_rng(0)
    y1 = rng.integers(0, 2, (80, 4))
    y2 = rng.integers(0, 2, (60, 4))
    good, bad = y1.copy(), y2 ^ (rng.random(y2.shape) < 0.4)
    d = stats.unpaired_bootstrap_diff(y1, good, np.arange(80) // 2, y2, bad, np.arange(60) // 2, n=200)
    assert d["diff"] > 0 and d["ci_low"] > 0
