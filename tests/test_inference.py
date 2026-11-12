"""Stage 14 tests: registered model -> batch predictions. Each model type must reproduce, at inference,
the exact probabilities it produced when it was trained (same model, same thresholds)."""

import json

import numpy as np
import pytest

from krs import inference, registry
from krs.evaluate.metrics import evaluate_runs
from krs.features import build_features
from krs.io import read_jsonl, write_jsonl
from krs.label.taxonomy import Domain, KnowledgeType
from krs.models.baselines import train_baseline
from krs.schema import (
    Label,
    LabelledSegment,
    LabelSource,
    Modality,
    Prediction,
    Segment,
    SourceDataset,
    Split,
)

VOCAB = {("decision", "architecture_design"): "we chose queue rather than inline because tradeoff",
         ("procedural", "devops_cicd"): "first build docker image then push pipeline",
         ("none", "other"): "thanks ok great cheers"}


class FakeEmbedder:
    def encode(self, texts, **kw):
        keys = ["queue", "docker", "thanks"]
        return np.array([[float(k in t) for k in keys] + [0.0] for t in texts], dtype=np.float32)


def _rows(n, split, prefix, source=SourceDataset.WILDCHAT):
    out, i = [], 0
    for (kt, dom), words in VOCAB.items():
        for _ in range(n):
            sid = f"{prefix}{i}:0"
            seg = Segment(segment_id=sid, conversation_id=f"{prefix}{i}", source_dataset=source,
                          modality=Modality.CONVERSATIONAL, turn_span=(0, 1), text=f"USER: {words} {i}",
                          context_text="USER: before", split=split)
            out.append(LabelledSegment(segment=seg, label=Label(segment_id=sid, knowledge_types=[KnowledgeType(kt)],
                                                                 domains=[Domain(dom)], source=LabelSource.LLM)))
            i += 1
    return out


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path / "data"))
    data = tmp_path / "data"
    rows = {"train": _rows(10, Split.TRAIN, "t"), "validation": _rows(4, Split.VALIDATION, "v"),
            "test": _rows(4, Split.TEST, "x", SourceDataset.DEVGPT)}
    for name, rs in rows.items():
        write_jsonl(data / "splits" / f"{name}.jsonl", rs)
    # segments to score: the test segments (devgpt) plus unlabelled wildchat ones
    write_jsonl(data / "segments" / "devgpt.jsonl", [r.segment for r in rows["test"]])
    write_jsonl(data / "segments" / "wildchat.jsonl", [r.segment for r in _rows(2, Split.TRAIN, "new")])
    fcfg = tmp_path / "f.yaml"
    fcfg.write_text("with_context: [true]\ntfidf:\n  min_df: 1\ntokenizer:\n  enabled: false\n")
    build_features(fcfg, embed_model=FakeEmbedder())
    res, models = tmp_path / "results", tmp_path / "models"
    train_baseline({"run_id": "B1", "model": "tfidf_lr", "C_grid": [10.0]}, results_root=res)
    train_baseline({"run_id": "B2", "model": "minilm_lr", "C_grid": [10.0]}, results_root=res)
    ecfg = tmp_path / "e.yaml"
    ecfg.write_text("split: test\nruns: [B1, B2]\nbootstrap: {n: 50, seed: 1}\n")
    evaluate_runs(ecfg, results_root=res)
    icfg = tmp_path / "i.yaml"
    icfg.write_text("version: null\ndevice: cpu\nbatch_size: 5\n")
    return tmp_path, res, models, icfg


def _check_matches_training(pred_file, res, run_id):
    preds = {p.segment_id: p for p in read_jsonl(pred_file, Prediction)}
    saved = np.load(res / run_id / "predictions_test.npz")
    for i, sid in enumerate(saved["segment_ids"]):
        p = preds[sid]
        got = np.array([p.knowledge_type_probs[KnowledgeType(k)] for k in p.knowledge_type_probs])
        assert np.allclose(got, saved["p_knowledge_types"][i], atol=1e-3)
        assert ("none" not in [k.value for k in p.knowledge_types]) == p.knowledge_bearing
    return preds


def test_tfidf_model_inference_resume_and_report(world):
    root, res, models, icfg = world
    v = registry.register("B1", results_root=res, models_dir=models)
    out = root / "preds.jsonl"
    rep = inference.classify(icfg, out_path=out, models_dir=models)
    assert rep["model_version"] == v and rep["scored"] == 18 and rep["cost_usd"] == 0.0
    assert rep["seconds_per_1k"] is not None and 0 < rep["knowledge_bearing_share"] < 1
    preds = _check_matches_training(out, res, "B1")
    assert preds["x0:0"].knowledge_types == [KnowledgeType.DECISION]  # learnable data: right answer
    assert json.loads((root / "preds_report.json").read_text())["scored"] == 18
    again = inference.classify(icfg, out_path=out, models_dir=models)  # resume: nothing left to do
    assert again["scored"] == 0 and again["skipped_already_done"] == 18
    flagged = list(inference.knowledge_bearing(out))
    assert len(flagged) == rep["knowledge_bearing"] and all(p.knowledge_bearing for p in flagged)


def test_embedding_model_and_source_filter(world):
    root, res, models, _ = world
    registry.register("B2", results_root=res, models_dir=models)
    cfg = root / "only_devgpt.yaml"
    cfg.write_text("version: null\nbatch_size: 64\nsources: [devgpt]\n")
    pred = inference.load_predictor(models_dir=models, embed_model=FakeEmbedder())
    rep = inference.classify(cfg, out_path=root / "p2.jsonl", predictor=pred)
    assert rep["scored"] == 12  # wildchat file filtered out
    _check_matches_training(root / "p2.jsonl", res, "B2")


def test_no_registered_model(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    with pytest.raises(registry.RegistrationError):
        inference.load_predictor(models_dir=tmp_path / "models")


@pytest.mark.slow
def test_deberta_inference_matches_training(world):
    from krs.models import finetune

    root, res, models, icfg = world
    cfg = {"run_id": "M1tiny", "model": "deberta", "base_model": "hf-internal-testing/tiny-random-deberta-v2",
           "epochs": 1, "batch_size": 8, "max_length": 64, "device": "cpu", "fp16": False, "save_model": True}
    finetune.train(cfg, results_root=res)
    ecfg = root / "e2.yaml"
    ecfg.write_text("split: test\nruns: [M1tiny]\nbootstrap: {n: 20, seed: 1}\n")
    evaluate_runs(ecfg, results_root=res)
    registry.register("M1tiny", results_root=res, models_dir=models)
    out = root / "pm.jsonl"
    rep = inference.classify(icfg, out_path=out, models_dir=models)
    assert rep["scored"] == 18 and rep["run_id"] == "M1tiny"
    _check_matches_training(out, res, "M1tiny")
