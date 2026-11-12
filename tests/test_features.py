"""Stage 5 tests. Embedding model and tokenizer are faked (no downloads, no heavy compute)."""

import json

import numpy as np
import pytest

from krs import features
from krs.io import write_jsonl
from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES, Domain, KnowledgeType
from krs.schema import Label, LabelledSegment, LabelSource, Modality, Segment, SourceDataset, Split


class FakeEmbedder:
    dim = 8

    def encode(self, texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
        return np.array([[len(t) % 7] * self.dim for t in texts], dtype=np.float32)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text.split()) + 2))}


def _row(sid, text, ctx, split, kts=("decision",), doms=("backend_apis",)):
    seg = Segment(segment_id=sid, conversation_id=sid.split(":")[0], source_dataset=SourceDataset.WILDCHAT,
                  modality=Modality.CONVERSATIONAL, turn_span=(0, 1), text=text, context_text=ctx, split=split)
    lab = Label(segment_id=sid, knowledge_types=[KnowledgeType(k) for k in kts],
                domains=[Domain(d) for d in doms], source=LabelSource.LLM)
    return LabelledSegment(segment=seg, label=lab)


@pytest.fixture
def splits(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    d = tmp_path / "splits"
    train = [_row(f"t{i}:0", f"USER: use postgres index for query {i}", "USER: earlier context words", Split.TRAIN)
             for i in range(6)] + [_row("t9:0", "USER: thanks", "", Split.TRAIN, kts=("none",), doms=("other",))]
    write_jsonl(d / "train.jsonl", train)
    write_jsonl(d / "validation.jsonl", [_row("v0:0", "USER: postgres query", "", Split.VALIDATION)])
    write_jsonl(d / "test.jsonl", [_row("x0:0", "USER: zebrafishword postgres", "", Split.TEST,
                                        kts=("procedural", "domain"), doms=("qa_testing", "general_programming"))])
    cfg = tmp_path / "features.yaml"
    cfg.write_text("with_context: [true, false]\ntfidf:\n  min_df: 1\n  ngram_range: [1, 2]\n"
                   "embeddings:\n  enabled: true\ntokenizer:\n  enabled: true\n  max_length: 6\n")
    return tmp_path, cfg


def test_label_matrices_order():
    rows = [_row("a:0", "x", "", Split.TRAIN, kts=("none",), doms=("general_programming",))]
    kt, dom = features.label_matrices(rows)
    assert kt.shape == (1, len(KNOWLEDGE_TYPES)) and dom.shape == (1, len(DOMAINS))
    assert kt[0, KNOWLEDGE_TYPES.index("none")] == 1 and kt.sum() == 1
    assert dom[0, DOMAINS.index("general_programming")] == 1


def test_texts_views():
    r = _row("a:0", "USER: q", "USER: before", Split.TRAIN)
    assert "before" in features.texts([r], True)[0] and features.texts([r], False) == ["USER: q"]


def test_build_features_fits_on_train_only(splits):
    root, cfg = splits
    rep = features.build_features(cfg, embed_model=FakeEmbedder(), tokenizer=FakeTokenizer())
    out = root / "features"
    import pickle
    with open(out / "tfidf_ctx.pkl", "rb") as f:
        vec = pickle.load(f)
    assert "zebrafishword" not in vec.vocabulary_  # word only in test never enters the vocabulary
    assert "postgres" in vec.vocabulary_
    assert rep["splits"] == {"train": 7, "validation": 1, "test": 1}
    X, ykt, ydom = features.load_features("test", "tfidf", True)
    assert X.shape == (1, rep["tfidf_ctx"]["vocabulary"]) and ykt.sum() == 2 and ydom.sum() == 2
    E, _, _ = features.load_features("train", "emb", False)
    assert E.shape == (7, FakeEmbedder.dim)
    assert rep["tfidf_noctx"]["vocabulary"] < rep["tfidf_ctx"]["vocabulary"]  # context words only in ctx view
    labels = np.load(out / "test_labels.npz")
    assert list(labels["segment_ids"]) == ["x0:0"]
    tl = rep["token_lengths"]
    assert tl["ctx"]["median"] > tl["noctx"]["median"] and 0 <= tl["ctx"]["share_over_limit"] <= 1
    assert json.loads((out / "features_report.json").read_text())["splits"]["train"] == 7


def test_no_train_split(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    cfg = tmp_path / "f.yaml"
    cfg.write_text("embeddings:\n  enabled: false\ntokenizer:\n  enabled: false\n")
    with pytest.raises(FileNotFoundError, match="Stage 6"):
        features.build_features(cfg)


def test_embeddings_and_tokenizer_can_be_switched_off(splits):
    root, _ = splits
    cfg = root / "light.yaml"
    cfg.write_text("with_context: [true]\ntfidf:\n  min_df: 1\nembeddings:\n  enabled: false\ntokenizer:\n  enabled: false\n")
    rep = features.build_features(cfg)
    assert "embeddings_ctx_dim" not in rep and "token_lengths" not in rep
    assert not list((root / "features").glob("*_emb_*.npy"))
