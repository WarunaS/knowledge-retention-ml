"""Stage 6 tests: conversation-level split, held-out test/transfer, leakage check, report."""

import json

import pytest

from krs import split
from krs.io import write_jsonl
from krs.label.taxonomy import Domain, KnowledgeType
from krs.schema import Label, LabelledSegment, LabelSource, Modality, Segment, SourceDataset, Split


def _seg(sid, cid, source=SourceDataset.WILDCHAT):
    return Segment(segment_id=sid, conversation_id=cid, source_dataset=source, modality=Modality.CONVERSATIONAL,
                   turn_span=(0, 1), text="t")


def _lab(sid, src=LabelSource.LLM, kt="decision"):
    return Label(segment_id=sid, knowledge_types=[KnowledgeType(kt)], domains=[Domain.BACKEND_APIS], source=src)


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    segs, weak = [], []
    for src, n in ((SourceDataset.WILDCHAT, 30), (SourceDataset.DEVGPT, 10), (SourceDataset.SWE_AGENT, 3)):
        for c in range(n):
            for k in range(3):  # 3 segments per conversation
                sid = f"{src.value}{c}:{k}"
                segs.append(_seg(sid, f"{src.value}{c}", src))
                weak.append(_lab(sid, kt="none" if k == 2 else "decision"))
    write_jsonl(tmp_path / "segments" / "all.jsonl", segs)
    # gold: two segments from conversations wildchat0 and devgpt0 (their weak rows must be dropped)
    write_jsonl(tmp_path / "labels" / "gold.jsonl", [_lab("wildchat0:1", LabelSource.ADJUDICATED),
                                                     _lab("devgpt0:0", LabelSource.ADJUDICATED)])
    write_jsonl(tmp_path / "labels" / "weak.jsonl", weak)
    cfg = tmp_path / "split.yaml"
    cfg.write_text("val_frac: 0.2\nseed: 7\n")
    return tmp_path, cfg


def test_make_splits(data):
    root, cfg = data
    rep = split.make_splits(cfg)
    train = split.load_split("train")
    val = split.load_split("validation")
    test = split.load_split("test")
    assert rep["weak_rows_removed_held_out_conversation"] == 6  # 2 conversations x 3 segments
    assert len(test) == 2 and all(r.segment.split == Split.TEST for r in test)
    assert all(r.segment.split == Split.TRAIN for r in train) and all(r.segment.split == Split.VALIDATION for r in val)
    assert len(train) + len(val) == 43 * 3 - 6
    held = {"wildchat0", "devgpt0"}
    assert not held & {r.segment.conversation_id for r in train + val}
    assert set(rep["validation"]["by_source"]) == {"wildchat", "devgpt", "swe_agent"}  # every source in validation
    assert rep["train"]["knowledge_type_share"]["none"] == pytest.approx(1 / 3, abs=0.02)
    assert json.loads((root / "splits" / "split_report.json").read_text())["test"]["segments"] == 2


def test_conversations_never_split(data):
    _, cfg = data
    split.make_splits(cfg)
    rows = {name: split.load_split(name) for name in ("train", "validation", "test")}
    owner = {}
    for name, rs in rows.items():
        for r in rs:
            assert owner.setdefault(r.segment.conversation_id, name) == name


def test_split_is_reproducible_and_seed_matters(data, tmp_path):
    _, cfg = data
    split.make_splits(cfg, out_dir=tmp_path / "a")
    split.make_splits(cfg, out_dir=tmp_path / "b")
    a = (tmp_path / "a" / "validation.jsonl").read_text()
    assert a == (tmp_path / "b" / "validation.jsonl").read_text()
    other = tmp_path / "other.yaml"
    other.write_text("val_frac: 0.2\nseed: 8\n")
    split.make_splits(other, out_dir=tmp_path / "c")
    assert a != (tmp_path / "c" / "validation.jsonl").read_text()


def test_leakage_check_raises():
    row = lambda sid, cid: LabelledSegment(segment=_seg(sid, cid), label=_lab(sid))
    split.check_leakage({"train": [row("a:0", "a")], "test": [row("b:0", "b")]})
    with pytest.raises(split.LeakageError):
        split.check_leakage({"train": [row("a:0", "a")], "test": [row("a:1", "a")]})


def test_missing_weak_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    cfg = tmp_path / "s.yaml"
    cfg.write_text("seed: 1\n")
    with pytest.raises(FileNotFoundError, match="Stage 4b"):
        split.make_splits(cfg)


def test_works_before_gold_exists(data):
    root, cfg = data
    (root / "labels" / "gold.jsonl").unlink()
    rep = split.make_splits(cfg)
    assert rep["test_present"] is False and rep["weak_rows_removed_held_out_conversation"] == 0
    assert not (root / "splits" / "test.jsonl").exists()
