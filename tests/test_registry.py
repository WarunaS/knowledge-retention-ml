"""Stages 11-12 tests: run tracking, runs index, choosing by validation, registering with a model card."""

import json
import pickle
import shutil

import numpy as np
import pytest

from krs import registry, tracking
from krs.evaluate.metrics import evaluate_runs
from krs.features import build_features, load_features
from krs.io import write_jsonl
from krs.label.taxonomy import Domain, KnowledgeType
from krs.models.baselines import predict_lr, train_baseline
from krs.schema import Label, LabelledSegment, LabelSource, Modality, Segment, SourceDataset, Split

VOCAB = {("decision", "architecture_design"): "we chose queue rather than inline because tradeoff",
         ("procedural", "devops_cicd"): "first build docker image then push pipeline",
         ("none", "other"): "thanks ok great cheers"}


def _rows(n, split, prefix):
    out, i = [], 0
    for (kt, dom), words in VOCAB.items():
        for _ in range(n):
            sid = f"{prefix}{i}:0"
            seg = Segment(segment_id=sid, conversation_id=f"{prefix}{i}", source_dataset=SourceDataset.WILDCHAT,
                          modality=Modality.CONVERSATIONAL, turn_span=(0, 1), text=f"USER: {words} {i}", split=split)
            out.append(LabelledSegment(segment=seg, label=Label(segment_id=sid, knowledge_types=[KnowledgeType(kt)],
                                                                 domains=[Domain(dom)], source=LabelSource.LLM)))
            i += 1
    return out


@pytest.fixture
def trained(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path / "data"))
    sp = tmp_path / "data" / "splits"
    write_jsonl(sp / "train.jsonl", _rows(10, Split.TRAIN, "t"))
    write_jsonl(sp / "validation.jsonl", _rows(4, Split.VALIDATION, "v"))
    write_jsonl(sp / "test.jsonl", _rows(4, Split.TEST, "x"))
    (sp / "split_report.json").write_text(json.dumps({"train": {"segments": 30}, "validation": {"segments": 12}}))
    fcfg = tmp_path / "f.yaml"
    fcfg.write_text("with_context: [true]\ntfidf:\n  min_df: 1\nembeddings:\n  enabled: false\ntokenizer:\n  enabled: false\n")
    build_features(fcfg)
    res = tmp_path / "results"
    train_baseline({"run_id": "B0", "model": "majority"}, results_root=res)
    train_baseline({"run_id": "B1", "model": "tfidf_lr", "C_grid": [1.0, 10.0]}, results_root=res)
    ecfg = tmp_path / "e.yaml"
    ecfg.write_text("split: test\nruns: [B0, B1]\nbootstrap: {n: 100, seed: 1}\ncomparisons:\n"
                    "  - {a: B1, b: B0, hypothesis: H1, test: better}\n")
    evaluate_runs(ecfg, results_root=res)
    return tmp_path, res


def test_run_info_and_index(trained):
    _, res = trained
    info = json.loads((res / "B1" / "run_info.json").read_text())
    assert info["git_commit"] and len(info["git_commit"]) == 40  # the repo is a git repository
    assert "numpy" in info["packages"] and "created_at" in info
    df = tracking.index_runs(res)
    assert set(df["run_id"]) == {"B0", "B1"} and df["has_model"].all() and df["has_test_predictions"].all()
    assert (res / "runs_index.csv").exists()


def test_best_run_uses_validation(trained):
    _, res = trained
    assert registry.best_run(["B0", "B1"], res) == "B1"
    with pytest.raises(registry.RegistrationError):
        registry.best_run(["nope"], res)


def test_register_refuses_unevaluated(trained):
    _, res = trained
    shutil.copytree(res / "B1", res / "B9")  # a run with a model but no gold report
    with pytest.raises(registry.RegistrationError, match="gold"):
        registry.register("B9", results_root=res, models_dir=res.parent / "models")
    with pytest.raises(registry.RegistrationError, match="no saved model"):
        registry.register("missing", results_root=res, models_dir=res.parent / "models")


def test_register_versions_card_and_reload(trained):
    root, res = trained
    models = root / "models"
    v1 = registry.register("B1", results_root=res, models_dir=models)
    v2 = registry.register("B1", results_root=res, models_dir=models)
    assert (v1, v2) == ("b1-v1", "b1-v2")
    reg = registry.list_versions(models)
    assert reg["current"] == "b1-v2" and [v["version"] for v in reg["versions"]] == ["b1-v1", "b1-v2"]
    card = (models / v1 / "model_card.md").read_text()
    assert "Knowledge-type macro-F1" in card and "95% CI" in card and "H1: B1" in card and "Not for monitoring" in card
    for f in ("model/classifiers.pkl", "model/tfidf.pkl", "thresholds.json", "metrics_gold.json", "run_info.json"):
        assert (models / v1 / f).exists(), f
    with pytest.raises(registry.RegistrationError, match="exists"):
        registry.register("B1", version="b1-v1", results_root=res, models_dir=models)

    loaded = registry.load(models_dir=models)
    assert loaded["version"] == "b1-v2" and loaded["config"]["model"] == "tfidf_lr"
    with open(loaded["model_dir"] / "classifiers.pkl", "rb") as f:
        clf = pickle.load(f)
    X, _, _ = load_features("test", "tfidf", True)
    saved = np.load(res / "B1" / "predictions_test.npz")
    assert np.allclose(predict_lr(clf["knowledge_types"], X), saved["p_knowledge_types"], atol=1e-6)  # same model


def test_load_without_registry(tmp_path):
    with pytest.raises(registry.RegistrationError, match="no registered model"):
        registry.load(models_dir=tmp_path)
