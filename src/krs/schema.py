"""Shared data contract for every lifecycle stage.

Lock these models in week one (architecture plan, section 10). Every stage reads and
writes them as JSON Lines, so stages can run in separate notebooks or machines.

    Conversation -> Segment -> Label / Prediction -> KnowledgeCard
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from itertools import pairwise
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from krs.label.taxonomy import Domain, KnowledgeType


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class Modality(str, Enum):
    CONVERSATIONAL = "conversational"  # chatbot logs (WildChat, DevGPT)
    AGENTIC = "agentic"  # agent runs with tool calls (SWE-agent, OpenHands, own logs)


class SourceDataset(str, Enum):
    WILDCHAT = "wildchat"
    DEVGPT = "devgpt"
    SWE_AGENT = "swe_agent"
    OPENHANDS = "openhands"
    OWN_LOGS = "own_logs"  # transfer test only, never training


class LabelSource(str, Enum):
    LLM = "llm"
    REPO_METADATA = "repo_metadata"  # DevGPT repository language/topics as a weak domain signal
    RATER_1 = "human_rater_1"
    RATER_2 = "human_rater_2"
    ADJUDICATED = "adjudicated"

    @property
    def is_gold(self) -> bool:
        return self in {LabelSource.RATER_1, LabelSource.RATER_2, LabelSource.ADJUDICATED}


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"  # gold set only
    TRANSFER = "transfer"  # consented own logs, scored once


class PersonRole(str, Enum):
    """Job role of the person behind a conversation. Recorded from the team directory, never predicted."""

    SOFTWARE_ENGINEER = "software_engineer"
    TECH_LEAD = "tech_lead"
    ARCHITECT = "architect"
    BUSINESS_ANALYST = "business_analyst"
    QA_ENGINEER = "qa_engineer"
    DATA_SCIENTIST = "data_scientist"
    OTHER = "other"


class WorkContext(BaseModel):
    """Who, which project and when. Carried from Conversation to Segment to KnowledgeCard.

    Used as retrieval filters and to order the decision log in handover packs. All optional:
    public corpora usually have a project (repository) but no person.
    """

    author_id: str | None = None  # pseudonymous; never a real name
    role: PersonRole | None = None
    project_id: str | None = None  # e.g. repository "owner/name"
    occurred_at: datetime | None = None


# --------------------------------------------------------------------------- stage 2: collect


class Turn(BaseModel):
    index: int = Field(ge=0)
    role: Role
    content: str
    tool_name: str | None = None


class Conversation(WorkContext):
    """One full dialogue or agent run, as emitted by a loader. Loaders never filter."""

    conversation_id: str
    source_dataset: SourceDataset
    licence: str
    modality: Modality
    turns: list[Turn]
    language: str | None = None
    toxic: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def n_exchanges(self) -> int:
        """Number of user turns directly followed by an assistant turn."""
        roles = [t.role for t in self.turns if t.role != Role.SYSTEM]
        return sum(
            1 for a, b in pairwise(roles) if a == Role.USER and b == Role.ASSISTANT
        )


# --------------------------------------------------------------------------- stage 3: clean


class Segment(WorkContext):
    """Unit of labelling and classification: one user-assistant pair plus one preceding pair.

    The context pair is kept separate from the target pair so ablation A1 (no context)
    only has to drop `context_text`.
    """

    segment_id: str
    conversation_id: str  # splits are made on this key
    source_dataset: SourceDataset
    modality: Modality
    turn_span: tuple[int, int]  # first and last turn index covered, including context
    context_text: str = ""
    text: str
    split: Split | None = None

    def model_input(self, with_context: bool = True) -> str:
        if with_context and self.context_text:
            return f"[CONTEXT]\n{self.context_text}\n[SEGMENT]\n{self.text}"
        return self.text


# --------------------------------------------------------------------------- stage 4b: label


class Label(BaseModel):
    segment_id: str
    domains: list[Domain] = Field(default_factory=list)
    knowledge_types: list[KnowledgeType] = Field(min_length=1)
    source: LabelSource
    rater_id: str | None = None
    codebook_version: str = "v1"
    raw_response: str | None = None  # LLM output kept for auditability
    created_at: datetime = Field(default_factory=_now)

    @field_validator("knowledge_types")
    @classmethod
    def _none_is_exclusive(cls, v: list[KnowledgeType]) -> list[KnowledgeType]:
        if KnowledgeType.NONE in v and len(v) > 1:
            raise ValueError("'none' cannot be combined with another knowledge type")
        return v

    @property
    def knowledge_bearing(self) -> bool:
        return KnowledgeType.NONE not in self.knowledge_types


# --------------------------------------------------------------------------- stage 14: inference


class Prediction(BaseModel):
    segment_id: str
    model_version: str
    run_id: str
    domain_probs: dict[Domain, float]
    knowledge_type_probs: dict[KnowledgeType, float]
    knowledge_bearing: bool
    created_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------- application plane


class KnowledgeCard(WorkContext):
    """Typed, grounded summary of knowledge from flagged segments. Provenance is mandatory."""

    card_id: str
    source_segment_ids: list[str] = Field(min_length=1)
    conversation_id: str
    domains: list[Domain]
    knowledge_types: list[KnowledgeType]
    title: str
    summary: str
    decision: str | None = None  # decision cards: what was chosen over what, and why
    tools_involved: list[str] = Field(default_factory=list)
    evidence: str  # quoted span from the source that grounds the summary
    confidence: float = Field(ge=0.0, le=1.0)
    faithfulness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str
    created_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def _decision_cards_need_decision(self) -> KnowledgeCard:
        if KnowledgeType.DECISION in self.knowledge_types and not self.decision:
            raise ValueError("decision cards must fill the 'decision' field")
        return self
