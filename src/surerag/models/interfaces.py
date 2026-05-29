"""Core component protocols for verifier implementations."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from surerag.protocol.schema import (
    Claim,
    EvidenceSufficiencyRequest,
    EvidenceSufficiencyResult,
    LabelDistribution,
    PairRelationScore,
)


class ClaimEvidencePair(BaseModel):
    request_id: str | None = None
    question: str
    answer: str
    claim_id: str
    claim_text: str
    evidence_id: str
    evidence_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceVerifier(Protocol):
    def verify(
        self, request: EvidenceSufficiencyRequest | dict[str, Any]
    ) -> EvidenceSufficiencyResult:
        ...


class ClaimBuilder(Protocol):
    def build_claims(self, request: EvidenceSufficiencyRequest) -> list[Claim]:
        ...


class PairRelationScorer(Protocol):
    def score_pairs(self, pairs: list[ClaimEvidencePair]) -> list[PairRelationScore]:
        ...


class Aggregator(Protocol):
    def predict_proba(self, features: dict[str, float]) -> LabelDistribution:
        ...


class Calibrator(Protocol):
    def calibrate(self, probs: LabelDistribution) -> LabelDistribution:
        ...
