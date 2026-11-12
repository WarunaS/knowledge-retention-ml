"""Stage 4b tests. The Claude client is faked: nothing is sent and nothing is billed."""

import csv
import json
from types import SimpleNamespace as NS

import pytest

from krs.io import read_jsonl, write_jsonl
from krs.label import agreement, gold, llm_labeller, noise, sampling
from krs.label.taxonomy import DOMAINS, Domain, KnowledgeType
from krs.schema import Label, LabelSource, Modality, Segment, SourceDataset

LLM = {"model": "claude-opus-5-5", "effort": "low", "max_tokens": 4096, "workers": 2,
       "prices": {"input": 4.0, "output": 20.0, "cache_read": 0.2, "cache_write": 5.0}}


def seg(sid, cid, source=SourceDataset.WILDCHAT, text="USER: q\nASSISTANT: a", ctx=""):
    modality = Modality.AGENTIC if source in (SourceDataset.SWE_AGENT, SourceDataset.OPENHANDS) else Modality.CONVERSATIONAL
    return Segment(segment_id=sid, conversation_id=cid, source_dataset=source, modality=modality,
                   turn_span=(0, 1), text=text, context_text=ctx)


def message(payload, stop="end_turn", model="claude-opus-5-5"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text=text)], stop_reason=stop, model=model,
              usage=NS(input_tokens=700, output_tokens=120, cache_read_input_tokens=2500, cache_creation_input_tokens=0))


GOOD = {"knowledge_types": ["decision"], "domains": ["backend_apis"], "rationale": "chose X over Y"}


class FakeClient:
    """Mimics the parts of anthropic.Anthropic the labeller uses."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []
        self.batches_created = []
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(batches=NS(create=self._batch_create, retrieve=self._batch_retrieve,
                                      results=self._batch_results))

    def _create(self, **params):
        self.calls.append(params)
        content = params["messages"][0]["content"]
        for key, ans in self.answers.items():
            if key in content:
                if isinstance(ans, Exception):
                    raise ans
                return ans
        return message(GOOD)

    def _batch_create(self, requests):
        self.batches_created.append(requests)
        return NS(id=f"batch_{len(self.batches_created)}")

    def _batch_retrieve(self, bid):
        return NS(processing_status="ended")

    def _batch_results(self, bid):
        reqs = self.batches_created[int(bid.split("_")[1]) - 1]
        out = []
        for r in reqs:
            cid = r["custom_id"]
            content = r["params"]["messages"][0]["content"]
            if "ERRORED" in content:
                out.append(NS(custom_id=cid, result=NS(type="errored")))
            else:
                out.append(NS(custom_id=cid, result=NS(type="succeeded", message=self._create(**r["params"]))))
        return out


# --------------------------------------------------------------------------- sampling


@pytest.fixture
def segdir(tmp_path):
    d = tmp_path / "segments"
    write_jsonl(d / "wildchat.jsonl", [seg(f"w{i}:0", f"w{i}") for i in range(40)]
                + [seg(f"w{i}:2", f"w{i}") for i in range(40)])
    write_jsonl(d / "swe_agent.jsonl", [seg(f"s{i}:0", f"s{i}", SourceDataset.SWE_AGENT) for i in range(200)])
    write_jsonl(d / "devgpt.jsonl", [seg(f"d{i}:0", f"d{i}", SourceDataset.DEVGPT) for i in range(5)])
    return d


def test_stratified_equal_with_small_source(segdir):
    refs = sampling.segment_refs(segdir)
    chosen = sampling.stratified_sample(refs, 60, seed=1)
    by = {s: sum(r.source == s for r in chosen) for s in ("wildchat", "swe_agent", "devgpt")}
    assert len(chosen) == 60 and by["devgpt"] == 5 and by["wildchat"] >= 27 and by["swe_agent"] >= 27
    assert sampling.stratified_sample(refs, 60, seed=1) == chosen


def test_pool_excludes_gold_conversations(segdir, tmp_path):
    gold_ids = sampling.select_gold(30, seed=3, segments_dir=segdir, out=tmp_path / "gold.json")
    pool = sampling.select_pool(100, gold_ids, seed=3, segments_dir=segdir, out=tmp_path / "pool.json")
    gold_convs = {g.rsplit(":", 1)[0] for g in gold_ids}
    assert not {p.rsplit(":", 1)[0] for p in pool} & gold_convs
    assert sampling.read_ids(tmp_path / "gold.json") == gold_ids


def test_load_segments_reports_missing(segdir):
    assert set(sampling.load_segments(["w1:0", "s2:0"], segdir)) == {"w1:0", "s2:0"}
    with pytest.raises(KeyError):
        sampling.load_segments(["nope"], segdir)


# --------------------------------------------------------------------------- request + parsing


def test_request_params_shape():
    p = llm_labeller.request_params(seg("a", "c", ctx="USER: before"), "CODEBOOK", LLM)
    assert p["model"] == "claude-opus-5-5" and "temperature" not in p and "thinking" not in p
    assert p["system"][1]["cache_control"] == {"type": "ephemeral"} and p["system"][1]["text"] == "CODEBOOK"
    assert p["output_config"]["effort"] == "low" and p["output_config"]["format"]["type"] == "json_schema"
    assert "<context>" in p["messages"][0]["content"] and "<segment>" in p["messages"][0]["content"]
    schema = llm_labeller.label_schema()
    assert schema["properties"]["domains"]["items"]["enum"] == DOMAINS and "general_programming" in DOMAINS


def test_custom_id_is_valid_and_stable():
    cid = llm_labeller.custom_id("swe_agent:django__django-11099:3f2a:14")
    assert len(cid) <= 64 and cid.replace("_", "").replace("-", "").isalnum()
    assert cid == llm_labeller.custom_id("swe_agent:django__django-11099:3f2a:14")


@pytest.mark.parametrize("msg,reason", [
    (message(GOOD, stop="refusal"), "refusal"),
    (message(GOOD, stop="max_tokens"), "max_tokens"),
    (message("not json"), "invalid"),
    (message({"knowledge_types": ["none", "decision"], "domains": ["other"], "rationale": ""}), "invalid"),
    (message({"knowledge_types": [], "domains": ["other"], "rationale": ""}), "invalid"),
])
def test_parse_rejects_bad_answers(msg, reason):
    with pytest.raises(llm_labeller.LabelError, match=reason):
        llm_labeller.parse_message("s1", msg, "v1", LabelSource.LLM)


def test_parse_good_answer_dedupes_and_keeps_raw():
    m = message({"knowledge_types": ["procedural", "procedural"], "domains": ["qa_testing"], "rationale": "r"})
    lab = llm_labeller.parse_message("s1", m, "v1", LabelSource.LLM)
    assert lab.knowledge_types == [KnowledgeType.PROCEDURAL] and lab.domains == [Domain.QA_TESTING]
    assert lab.rater_id == "claude-opus-5-5" and json.loads(lab.raw_response)["rationale"] == "r"


def test_cost():
    u = {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 0}
    assert llm_labeller.cost_usd(u, LLM["prices"]) == pytest.approx(6.0)
    assert llm_labeller.cost_usd(u, LLM["prices"], batch=True) == pytest.approx(3.0)


# --------------------------------------------------------------------------- sync + batch with fake client


def test_label_sync_resume_errors_and_report(tmp_path):
    segs = [seg(f"x{i}", f"c{i}", text=f"USER: q{i}") for i in range(5)]
    client = FakeClient({"q3": RuntimeError("boom"), "q4": message("garbage")})
    out = tmp_path / "teacher.jsonl"
    rep = llm_labeller.label_sync(segs, out, LLM, client=client, codebook="CB")
    assert rep["labelled"] == 3 and rep["failed"] == 2 and rep["cost_usd"] > 0 and rep["latency_s_median"] is not None
    assert client.calls[0]["fallbacks"] == "default" and client.calls[0]["betas"] == [llm_labeller.FALLBACK_BETA]
    errs = [json.loads(line) for line in (tmp_path / "teacher_errors.jsonl").read_text().splitlines()]
    assert {e["segment_id"] for e in errs} == {"x3", "x4"}
    client2 = FakeClient()
    rep2 = llm_labeller.label_sync(segs, out, LLM, client=client2, codebook="CB")  # resume: only failures redone
    assert rep2["skipped_already_done"] == 3 and rep2["labelled"] == 2 and len(client2.calls) == 2
    assert len(list(read_jsonl(out, Label))) == 5


def test_batch_submit_collect(tmp_path):
    segs = [seg(f"b{i}", f"c{i}", text=f"USER: q{i}") for i in range(7)] + [seg("bad", "cx", text="ERRORED")]
    client = FakeClient()
    state = tmp_path / "batch_state.json"
    sub = llm_labeller.submit_batch(segs, state, {**LLM, "batch_chunk": 3}, client=client, codebook="CB",
                                    done=["b0"])
    assert sub == {"batches": 3, "requests": 7}
    first = client.batches_created[0][0]
    assert "fallbacks" not in first["params"] and first["custom_id"].startswith("s")
    out = tmp_path / "weak.jsonl"
    got = llm_labeller.collect_batch(state, out, LLM, client=client)
    assert got["labelled"] == 6 and got["failed"] == 1 and got["cost_usd"] > 0
    assert {lab.segment_id for lab in read_jsonl(out, Label)} == {f"b{i}" for i in range(1, 7)}
    again = llm_labeller.collect_batch(state, out, LLM, client=client)  # safe to re-run
    assert again["labelled"] == 0


# --------------------------------------------------------------------------- gold sheets + agreement


def _fill(csv_path, answers):
    """answers: segment_id -> (types, domains). Writes x marks like a rater would."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        ks, ds = answers.get(r["segment_id"], ((), ()))
        for k in ks:
            r[f"k:{k}"] = "x"
        for d in ds:
            r[f"d:{d}"] = "X"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_rater_sheet_round_trip_and_validation(tmp_path):
    segs = [seg(f"g{i}", f"c{i}") for i in range(4)]
    sheet = tmp_path / "rater1.csv"
    assert gold.export_rater_sheet(segs, sheet) == 4
    _fill(sheet, {"g0": (["decision"], ["backend_apis"]), "g1": (["none"], ["other"]),
                  "g2": (["none", "domain"], ["other"]), "g3": (["procedural"], [])})
    labels, problems = gold.import_rater_sheet(sheet, LabelSource.RATER_1, "waruna")
    assert [lab.segment_id for lab in labels] == ["g0", "g1"]
    assert len(problems) == 2 and "g2" in problems[0] and "g3" in problems[1]


def test_kappa_values():
    assert agreement.binary_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert agreement.binary_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
    assert agreement.binary_kappa([False] * 3, [False] * 3) is None


def _lab(sid, ks, ds, src, rater):
    return Label(segment_id=sid, knowledge_types=[KnowledgeType(k) for k in ks], domains=[Domain(d) for d in ds],
                 source=src, rater_id=rater)


def test_agreement_disagreements_and_merge(tmp_path):
    r1 = [_lab("a", ["decision"], ["backend_apis"], LabelSource.RATER_1, "r1"),
          _lab("b", ["procedural"], ["qa_testing"], LabelSource.RATER_1, "r1"),
          _lab("c", ["none"], ["other"], LabelSource.RATER_1, "r1")]
    r2 = [_lab("a", ["decision"], ["backend_apis"], LabelSource.RATER_2, "r2"),
          _lab("b", ["procedural", "domain"], ["qa_testing"], LabelSource.RATER_2, "r2"),
          _lab("c", ["none"], ["general_programming"], LabelSource.RATER_2, "r2")]
    ag = agreement.agreement(r1, r2)
    assert ag["segments_compared"] == 3 and ag["knowledge_type"]["exact_match"] == pytest.approx(2 / 3, abs=1e-3)
    assert ag["knowledge_type"]["kappa_per_label"]["decision"] == 1.0
    assert agreement.disagreements(r1, r2) == ["b", "c"]

    segs = {s: seg(s, s) for s in "abc"}
    adj = tmp_path / "adjudicate.csv"
    assert agreement.export_adjudication_sheet(r1, r2, segs, adj) == 2
    with open(adj, encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "rater1" in header and "rater2" in header
    final, problems = agreement.merge_gold(r1, r2, None)
    assert [f.segment_id for f in final] == ["a"] and len(problems) == 2
    _fill(adj, {"b": (["procedural"], ["qa_testing"]), "c": (["none"], ["general_programming"])})
    final, problems = agreement.merge_gold(r1, r2, adj)
    assert not problems and [f.segment_id for f in final] == ["a", "b", "c"]
    assert all(f.source == LabelSource.ADJUDICATED for f in final)


# --------------------------------------------------------------------------- noise check


def test_noise_sample_and_rate(segdir, tmp_path):
    weak = tmp_path / "weak.jsonl"
    write_jsonl(weak, [_lab(f"w{i}:0", ["decision"], ["backend_apis"], LabelSource.LLM, "m") for i in range(30)])
    out = tmp_path / "noise.csv"
    assert noise.export_noise_sample(weak, out, n=10, segments_dir=segdir) == 10
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["types_correct"] = "n" if i < 2 else "y"
        r["domains_correct"] = "y"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rate = noise.noise_rate(out)
    assert rate["knowledge_type"]["noise_rate"] == pytest.approx(0.2) and rate["domain"]["wrong"] == 0
