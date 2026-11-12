"""Stages 7-8 and 10 tests. Fast tests use synthetic data and fakes; the two `slow` tests run the real
fine-tune and SetFit code paths with tiny test-sized models (a few MB, seconds on a laptop)."""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from krs.features import build_features
from krs.io import write_jsonl
from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES, Domain, KnowledgeType
from krs.models import baselines, common, finetune, setfit_run, teacher
from krs.schema import Label, LabelledSegment, LabelSource, Modality, Segment, SourceDataset, Split

# words decide the label, so a working model must learn it
VOCAB = {
    ("decision", "architecture_design"): "we chose queue rather than inline because load tradeoff",
    ("procedural", "devops_cicd"): "first build docker image then push pipeline deploy steps",
    ("none", "other"): "thanks ok great cheers",
}


def _rows(n_per, split, prefix):
    rows, i = [], 0
    for (kt, dom), words in VOCAB.items():
        for _ in range(n_per):
            sid = f"{prefix}{i}:0"
            seg = Segment(segment_id=sid, conversation_id=f"{prefix}{i}", source_dataset=SourceDataset.WILDCHAT,
                          modality=Modality.CONVERSATIONAL, turn_span=(0, 1), text=f"USER: {words} {i}",
                          context_text="USER: earlier", split=split)
            lab = Label(segment_id=sid, knowledge_types=[KnowledgeType(kt)], domains=[Domain(dom)], source=LabelSource.LLM)
            rows.append(LabelledSegment(segment=seg, label=lab))
            i += 1
    return rows


class FakeEmbedder:
    def encode(self, texts, **kw):
        keys = ["queue", "docker", "thanks"]
        return np.array([[float(k in t) for k in keys] + [0.0] for t in texts], dtype=np.float32)


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    d = tmp_path / "splits"
    write_jsonl(d / "train.jsonl", _rows(10, Split.TRAIN, "t"))
    write_jsonl(d / "validation.jsonl", _rows(4, Split.VALIDATION, "v"))
    write_jsonl(d / "test.jsonl", _rows(4, Split.TEST, "x"))
    cfg = tmp_path / "f.yaml"
    cfg.write_text("with_context: [true, false]\ntfidf:\n  min_df: 1\ntokenizer:\n  enabled: false\n")
    build_features(cfg, embed_model=FakeEmbedder())
    return tmp_path


# --------------------------------------------------------------------------- common


def test_decode_rules():
    p_kt = np.zeros((3, len(KNOWLEDGE_TYPES)))
    p_kt[0, [KNOWLEDGE_TYPES.index("none"), KNOWLEDGE_TYPES.index("decision")]] = 0.9  # none + decision
    p_kt[1, KNOWLEDGE_TYPES.index("procedural")] = 0.3  # nothing above threshold
    p_dom = np.zeros((3, len(DOMAINS)))
    p_dom[2, DOMAINS.index("security")] = 0.2
    kt, dom = common.decode(p_kt, p_dom)
    assert kt[0, KNOWLEDGE_TYPES.index("none")] == 0 and kt[0, KNOWLEDGE_TYPES.index("decision")] == 1
    assert kt[1, KNOWLEDGE_TYPES.index("procedural")] == 1  # argmax fallback
    assert (kt.sum(1) >= 1).all() and (dom.sum(1) >= 1).all() and dom[2, DOMAINS.index("security")] == 1


def test_macro_f1_ignores_absent_labels_and_thresholds():
    y = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    assert common.macro_f1(y, y) == 1.0  # third label absent everywhere: not averaged
    p = np.array([[0.3, 0.1, 0.0], [0.1, 0.8, 0.0], [0.35, 0.1, 0.0]])
    th = common.tune_thresholds(p, y)
    assert th[0] <= 0.3 and th[2] == 0.5


def test_encode_cuts_context_first():
    class Tok:  # same surface as a transformers 5 tokenizer: cls/sep ids + encode
        cls_token_id, sep_token_id = -1, -2

        def encode(self, text, add_special_tokens=False):
            return [int(w) for w in text.split()]

    tok = Tok()
    text = " ".join(str(i) for i in range(100, 106))  # 6 tokens
    ctx = " ".join(str(i) for i in range(10))  # 10 tokens
    ids = finetune.encode(tok, text, ctx, max_length=12)
    assert len(ids) == 12 and ids[-7:-1] == list(range(100, 106))  # segment kept whole
    assert ids[1:4] == [7, 8, 9]  # only the END of the context survives
    long_text = " ".join(str(i) for i in range(200, 230))
    ids = finetune.encode(tok, long_text, ctx, max_length=12)
    assert len(ids) == 12 and 9 not in ids and ids[0] == -1 and ids[-1] == -2  # context dropped, segment cut
    assert finetune.encode(tok, text, "", 12) == [-1, *range(100, 106), -2]


# --------------------------------------------------------------------------- baselines


@pytest.mark.parametrize("model", ["majority", "tfidf_lr", "minilm_lr"])
def test_baselines(data, tmp_path, model):
    cfg = {"run_id": f"r_{model}", "model": model, "with_context": True, "C_grid": [1.0, 10.0]}
    m = baselines.train_baseline(cfg, results_root=tmp_path / "results")
    out = tmp_path / "results" / f"r_{model}"
    pred = np.load(out / "predictions_test.npz")
    assert pred["p_knowledge_types"].shape == (12, len(KNOWLEDGE_TYPES)) and len(pred["segment_ids"]) == 12
    assert json.loads((out / "metrics_validation.json").read_text())["model"] == model
    assert (out / "thresholds.json").exists() and (out / "config.yaml").exists()
    if model == "majority":
        assert m["macro_f1_mean"] < 0.6
    else:
        assert m["macro_f1_mean"] > 0.9  # words decide the label: a working model learns it


def test_runner_unknown_model(monkeypatch):
    from krs.models import runner

    monkeypatch.setattr(runner, "load_config", lambda name: {"model": "nope"})
    with pytest.raises(ValueError, match="unknown model"):
        runner.run_experiment("Z")


# --------------------------------------------------------------------------- SetFit (cross-fitting)


def test_folds_keep_conversations_and_cover_all():
    rows = _rows(5, Split.TEST, "x")
    rows += [r.model_copy(update={"segment": r.segment.model_copy(update={"segment_id": r.segment.segment_id + "b"})})
             for r in rows[:3]]  # second segment of the same conversations
    folds = setfit_run.conversation_folds(rows, k=3, seed=1)
    by_conv = {}
    for r, f in zip(rows, folds, strict=True):
        assert by_conv.setdefault(r.segment.conversation_id, f) == f
    assert set(folds) == {0, 1, 2}


def test_few_shot_indices():
    Y = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]])
    idx = setfit_run.few_shot_indices(Y, np.arange(5), n_per_class=1, seed=0)
    assert len(idx) == 2 and Y[idx].sum(0).tolist() == [1, 1]


def test_setfit_never_predicts_its_training_rows(data, tmp_path):
    seen = []

    def fake_fit_predict(train_texts, Y, pred_texts, cfg):
        assert not set(train_texts) & set(pred_texts)  # cross-fitting: no overlap
        seen.extend(pred_texts)
        return np.full((len(pred_texts), Y.shape[1]), 0.6)

    m = setfit_run.train({"run_id": "M2", "folds": 3, "n_per_class": 2}, results_root=tmp_path / "res",
                         fit_predict=fake_fit_predict)
    assert len(seen) == 2 * 12  # every gold row predicted once per axis
    pred = np.load(tmp_path / "res" / "M2" / "predictions_test.npz")
    assert pred["p_domains"].shape == (12, len(DOMAINS)) and "cross-fitting" in m["scheme"]


# --------------------------------------------------------------------------- teacher


class FakeClient:
    def __init__(self):
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **params):
        content = params["messages"][0]["content"]
        if "x0:" in content or "x0 " in content or "USER: we chose queue rather than inline because load tradeoff 0" in content:
            raise RuntimeError("boom")
        body = {"knowledge_types": ["decision"], "domains": ["architecture_design"], "rationale": "r"}
        return NS(content=[NS(type="text", text=json.dumps(body))], stop_reason="end_turn", model="claude-opus-5-5",
                  usage=NS(input_tokens=600, output_tokens=100, cache_read_input_tokens=0, cache_creation_input_tokens=0))


def test_teacher_dry_run_and_confirmed_run(data, tmp_path):
    est = teacher.run({"run_id": "T"})
    assert est["dry_run"] is True and est["segments"] == 12 and est["estimated_cost_usd"] > 0
    m = teacher.run({"run_id": "T", "llm": {"workers": 1}}, confirm=True, client=FakeClient(),
                    results_root=tmp_path / "res")
    assert m["labelled"] == 11 and m["failed"] == 1 and len(m["missing_segments"]) == 1
    pred = np.load(tmp_path / "res" / "T" / "predictions_test.npz")
    assert pred["p_knowledge_types"].sum() == 11  # one 1 per labelled segment, zeros for the failed one


# --------------------------------------------------------------------------- slow: real code paths, tiny models


@pytest.mark.slow
def test_finetune_tiny_model(data, tmp_path):
    cfg = {"run_id": "M1tiny", "base_model": "hf-internal-testing/tiny-random-deberta-v2", "epochs": 1,
           "batch_size": 8, "max_length": 64, "device": "cpu", "fp16": False, "save_model": True, "lr": 1e-3}
    m = finetune.train(cfg, results_root=tmp_path / "res")
    out = tmp_path / "res" / "M1tiny"
    assert m["best_epoch"] == 1 and (out / "model" / "weights.pt").exists()
    assert np.load(out / "predictions_test.npz")["p_domains"].shape == (12, len(DOMAINS))
    curve = finetune.train({**cfg, "run_id": "A2tiny", "train_sizes": [6, 12], "save_model": False},
                           results_root=tmp_path / "res")
    assert set(curve["learning_curve"]) == {6, 12}


@pytest.mark.slow
def test_setfit_tiny_model(data, tmp_path):
    cfg = {"run_id": "M2tiny", "base_model": "sentence-transformers-testing/stsb-bert-tiny-safetensors",
           "folds": 2, "n_per_class": 3, "num_iterations": 2, "batch_size": 8}
    setfit_run.train(cfg, results_root=tmp_path / "res")
    pred = np.load(tmp_path / "res" / "M2tiny" / "predictions_test.npz")
    assert pred["p_knowledge_types"].shape == (12, len(KNOWLEDGE_TYPES))
