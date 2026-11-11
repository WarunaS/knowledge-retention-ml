"""Two-axis knowledge taxonomy (v2 research plan, section 5).

Both axes are multi-label. This module is the single code definition of the labels;
`codebook.md` next to it holds the human-readable definitions, examples and cues.
"""

from __future__ import annotations

from enum import Enum


class Domain(str, Enum):
    """Domain axis. Technical labels follow Stack Overflow Developer Survey role/tag categories."""

    MOBILE = "mobile"
    WEB_FRONTEND = "web_frontend"
    BACKEND_APIS = "backend_apis"
    DATA_ANALYTICS = "data_analytics"
    CLOUD_INFRA = "cloud_infra"
    DEVOPS_CICD = "devops_cicd"
    SECURITY = "security"
    QA_TESTING = "qa_testing"
    ARCHITECTURE_DESIGN = "architecture_design"
    # Business analyst knowledge: requirements, business rules, stakeholder decisions.
    # Expected to be rare in public developer corpora; report its scores with that caveat.
    BUSINESS_REQUIREMENTS = "business_requirements"
    OTHER = "other"


class KnowledgeType(str, Enum):
    """Knowledge-type axis. Carried over from the group proposal and the SECI framing."""

    DOMAIN = "domain"  # facts and expertise about a technology or problem area
    PROCEDURAL = "procedural"  # how-to steps, workflows, commands
    DECISION = "decision"  # a choice made over alternatives, with a reason
    CONTEXTUAL = "contextual"  # project or organisational context, constraints
    NONE = "none"  # no reusable knowledge in the segment


DOMAINS: list[str] = [d.value for d in Domain]
KNOWLEDGE_TYPES: list[str] = [k.value for k in KnowledgeType]
