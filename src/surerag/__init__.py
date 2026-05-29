"""SURE-RAG evidence sufficiency verification protocol."""

from surerag.core.verifier import SureRAGVerifier
from surerag.protocol.labels import ACTIONS, ANSWER_LABELS, PAIR_LABELS, PROTOCOL_VERSION
from surerag.protocol.schema import (
    AuditTrace,
    Claim,
    ClaimAudit,
    Evidence,
    EvidenceSufficiencyRequest,
    EvidenceSufficiencyResult,
    LabelDistribution,
    ModelCapabilities,
    ModelInfo,
    PairDistribution,
    PairRelationScore,
)

__all__ = [
    "ACTIONS",
    "ANSWER_LABELS",
    "PAIR_LABELS",
    "PROTOCOL_VERSION",
    "AuditTrace",
    "Claim",
    "ClaimAudit",
    "Evidence",
    "EvidenceSufficiencyRequest",
    "EvidenceSufficiencyResult",
    "LabelDistribution",
    "ModelCapabilities",
    "ModelInfo",
    "PairDistribution",
    "PairRelationScore",
    "SureRAGVerifier",
]
