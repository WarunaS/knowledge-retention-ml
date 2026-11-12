"""Stage 9 tests: metrics, conversation bootstrap, paired comparisons, the evaluation runner."""

import json

import numpy as np
import pytest

from krs.evaluate import metrics, stats
from krs.io import write_jsonl
from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES, Domain, KnowledgeType
from krs.models.common import run_dir, save_predictions
from krs.schema import Label, LabelledSegment, LabelSource, Modality, Segment, SourceDataset, Split

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- stats


def test_macro_f1_fixed_known_values():
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    assert stats.macro_f1_fixed(y, y, np.array([0, 1])) == 1.0
    yhat = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])  # label0: tp2 fp1 -> 0.8; label1: tp1 fn1 -> 0.667
    assert stats.macro_f1_fixed(y, yhat, np.array([0, 1])) == pytest.approx((0.8 + 2 / 3) / 2)
    assert stats.macro_f1_fixed(y[:, :1] * 0, y[:, :1] * 0, np.array([0])) == 0.0  # undefined label skipped


def test_bootstrap_perfect_and_reproducible():
    y = rng.integers(0, 2, (60, 4))
    groups = np.repeat(np.arange(20), 3)
    ci = stats.bootstrap_ci(y, y, groups, n=200)
    assert ci == {"macro_f1": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    noisy = y ^ (rng.random(y.shape) < 0.3)
    a = stats.bootstrap_ci(y, noisy, groups, n=200, seed=1)
    assert a == stats.bootstrap_ci(y, noisy, groups, n=200, seed=1)
    assert a["ci_low"] < a["macro_f1"] < a["ci_high"]


def test_conversation_bootstrap_is_wider_than_segment_bootstrap():
    # 30 conversations x 5 segments; errors are shared within a conversation (correlated)
    groups = np.repeat(np.arange(30), 5)
    y = np.tile(np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1]]), (30, 1))
    wrong_conv = rng.random(30) < 0.4
    yhat = y.copy()
    yhat[np.repeat(wrong_conv, 5)] ^= 1
    by_conv = stats.bootstrap_ci(y, yhat, groups, n=400, seed=3)
    by_seg = stats.bootstrap_ci(y, yhat, np.arange(len(y)), n=400, seed=3)
    assert by_conv["ci_high"] - by_conv["ci_low"] > by_seg["ci_high"] - by_seg["ci_low"]


def test_paired_bootstrap_detects_better_model_and_margin():
    y = rng.integers(0, 2, (200, 5))
    groups = np.repeat(np.arange(100), 2)
    good = y ^ (rng.random(y.shape) < 0.05)
    bad = y ^ (rng.random(y.shape) < 0.4)
    res = stats.paired_bootstrap(y, good, bad, groups, n=300)
    assert res["diff"] > 0 and res["ci_low"] > 0 and res["p_value"] < 0.01
    same = stats.paired_bootstrap(y, good, good, groups, n=100)
    assert same["diff"] == 0 and same["p_value"] == 1.0
    assert stats.within_margin({"ci_low": -0.03}, 0.05) and not stats.within_margin({"ci_low": -0.08}, 0.05)


# --------------------------------------------------------------------------- runner on synthetic gold


@pytest.fixture
def gold_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    rows, kts, doms = [], ["decision", "procedural", "none"], ["backend_apis", "qa_testing", "other"]
    for c in range(30):
        for k in range(2):
            sid = f"c{c}:{k}"
            src = SourceDataset.WILDCHAT if c % 2 else SourceDataset.SWE_AGENT
            seg = Segment(segment_id=sid, conversation_id=f"c{c}", source_dataset=src, modality=Modality.CONVERSATIONAL,
                          turn_span=(0, 1), text="t", split=Split.TEST)
            lab = Label(segment_id=sid, knowledge_types=[KnowledgeType(kts[c % 3])], domains=[Domain(doms[(c + k) % 3])],
                        source=LabelSource.ADJUDICATED)
            rows.append(LabelledSegment(segment=seg, label=lab))
    write_jsonl(tmp_path / "splits" / "test.jsonl", rows)
    gold = metrics.load_gold()
    res = tmp_path / "results"

    def save(run_id, kt, dom, order=None):
        ids = gold["segment_ids"] if order is None else gold["segment_ids"][order]
        kt, dom = (kt, dom) if order is None else (kt[order], dom[order])
        th = (np.full(len(KNOWLEDGE_TYPES), 0.5), np.full(len(DOMAINS), 0.5))
        save_predictions(run_dir(run_id, res), "test", ids, kt.astype(float), dom.astype(float), *th)

    perfect = (gold["knowledge_types"], gold["domains"])
    save("M1", *perfect, order=rng.permutation(len(rows)))  # stored in a different order: must be realigned
    noisy_kt = gold["knowledge_types"].copy()
    noisy_kt[::3] = 0
    noisy_kt[::3, KNOWLEDGE_TYPES.index("domain")] = 1
    save("B2", noisy_kt, gold["domains"])
    save("T", *perfect)
    cfg = tmp_path / "evaluate.yaml"
    cfg.write_text(
        "split: test\nruns: [B2, M1, T, A1]\nbootstrap: {n: 200, seed: 1}\n"
        "comparisons:\n  - {a: M1, b: B2, hypothesis: H1, test: better}\n"
        "  - {a: M1, b: T, hypothesis: H2, test: within, margin: 0.05}\n  - {a: M1, b: A1, hypothesis: H3, test: better}\n"
        "confusion_for: [B2]\n")
    return tmp_path, cfg, res


def test_evaluate_runs(gold_setup):
    _, cfg, res = gold_setup
    s = metrics.evaluate_runs(cfg, results_root=res)
    out = res / "eval"
    assert s["gold_segments"] == 60 and s["gold_conversations"] == 30 and s["bootstrap"]["by"] == "conversation"
    assert s["runs_evaluated"] == ["B2", "M1", "T"] and s["runs_missing"] == ["A1"]
    m1 = json.loads((out / "report_M1.json").read_text())
    assert m1["knowledge_types"]["macro_f1"] == 1.0 and m1["domains"]["exact_match"] == 1.0  # realigned by id
    assert m1["knowledge_bearing_f1"] == 1.0 and set(m1["domains"]["macro_f1_by_source"]) == {"wildchat", "swe_agent"}
    b2 = json.loads((out / "report_B2.json").read_text())
    assert b2["knowledge_types"]["macro_f1"] < 1.0
    h1, h2, h3 = s["comparisons"]
    assert h1["knowledge_types"]["a_better"] is True and h1["knowledge_types"]["diff"] > 0
    assert h2["knowledge_types"]["within_margin"] is True
    assert h3["status"].startswith("skipped")
    assert (out / "results_table.md").read_text().startswith("| run |")
    assert (out / "confusion_B2_knowledge_types.png").stat().st_size > 1000


def test_missing_prediction_raises(gold_setup):
    _, _, res = gold_setup
    gold = metrics.load_gold()
    th = (np.full(len(KNOWLEDGE_TYPES), 0.5), np.full(len(DOMAINS), 0.5))
    save_predictions(run_dir("Short", res), "test", gold["segment_ids"][:10],
                     gold["knowledge_types"][:10].astype(float), gold["domains"][:10].astype(float), *th)
    with pytest.raises(KeyError, match="no prediction"):
        metrics.aligned_predictions("Short", gold, results_root=res)


def test_label_confusion_counts():
    y = np.array([[1, 0, 0], [0, 1, 0]])
    yhat = np.array([[1, 1, 0], [1, 0, 0]])
    cm = metrics.label_confusion(y, yhat)
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 0] == 1 and cm[1, 1] == 0


def test_no_gold_split(tmp_path, monkeypatch):
    monkeypatch.setenv("KRS_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="gold labelling"):
        metrics.load_gold()
