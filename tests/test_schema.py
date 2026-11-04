import pytest
from pydantic import ValidationError

from krs.io import read_jsonl, write_jsonl
from krs.label.taxonomy import Domain, KnowledgeType
from krs.schema import (
    Conversation,
    KnowledgeCard,
    Label,
    LabelSource,
    Modality,
    Role,
    Segment,
    SourceDataset,
    Turn,
)


def _conv() -> Conversation:
    return Conversation(
        conversation_id="c1",
        source_dataset=SourceDataset.WILDCHAT,
        licence="ODC-BY",
        modality=Modality.CONVERSATIONAL,
        turns=[
            Turn(index=0, role=Role.SYSTEM, content="sys"),
            Turn(index=1, role=Role.USER, content="How do I add an index?"),
            Turn(index=2, role=Role.ASSISTANT, content="CREATE INDEX ..."),
            Turn(index=3, role=Role.USER, content="Why not a composite one?"),
            Turn(index=4, role=Role.ASSISTANT, content="Because ..."),
        ],
    )


def test_n_exchanges_ignores_system_turns():
    assert _conv().n_exchanges() == 2


def test_none_knowledge_type_is_exclusive():
    with pytest.raises(ValidationError):
        Label(
            segment_id="s1",
            knowledge_types=[KnowledgeType.NONE, KnowledgeType.DECISION],
            source=LabelSource.LLM,
        )


def test_gold_sources():
    assert LabelSource.ADJUDICATED.is_gold
    assert not LabelSource.LLM.is_gold


def test_card_requires_provenance():
    with pytest.raises(ValidationError):
        KnowledgeCard(
            card_id="k1",
            source_segment_ids=[],
            conversation_id="c1",
            domains=[Domain.BACKEND_APIS],
            knowledge_types=[KnowledgeType.PROCEDURAL],
            title="t",
            summary="s",
            evidence="e",
            confidence=0.9,
            model_version="m1-v1",
        )


def test_decision_card_requires_decision():
    with pytest.raises(ValidationError):
        KnowledgeCard(
            card_id="k1",
            source_segment_ids=["s1"],
            conversation_id="c1",
            domains=[Domain.BACKEND_APIS],
            knowledge_types=[KnowledgeType.DECISION],
            title="t",
            summary="s",
            evidence="e",
            confidence=0.9,
            model_version="m1-v1",
        )


def test_segment_model_input_ablation():
    s = Segment(
        segment_id="s1",
        conversation_id="c1",
        source_dataset=SourceDataset.WILDCHAT,
        modality=Modality.CONVERSATIONAL,
        turn_span=(1, 4),
        context_text="prev",
        text="target",
    )
    assert "prev" in s.model_input(with_context=True)
    assert s.model_input(with_context=False) == "target"


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "convs.jsonl"
    assert write_jsonl(path, [_conv()]) == 1
    [back] = list(read_jsonl(path, Conversation))
    assert back == _conv()
