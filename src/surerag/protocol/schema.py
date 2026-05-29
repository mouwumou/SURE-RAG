"""Pydantic models for the evidence sufficiency protocol."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from surerag.protocol.labels import ACTIONS, ANSWER_LABELS, PROTOCOL_VERSION

_PROB_SUM_TOL = 1e-6


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _duplicate_ids(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _validate_probability_values(values: list[float], name: str) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} probabilities must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} probabilities must be non-negative")
    total = sum(values)
    if abs(total - 1.0) > _PROB_SUM_TOL:
        raise ValueError(f"{name} probabilities must sum to 1.0; got {total}")


class Claim(BaseModel):
    id: str
    text: str
    source: Literal["provided", "heuristic", "llm", "system"] = "provided"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _support_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            normalized = dict(data)
            if "id" not in normalized and "claim_id" in normalized:
                normalized["id"] = normalized["claim_id"]
            if "text" not in normalized and "claim" in normalized:
                normalized["text"] = normalized["claim"]
            return normalized
        return data

    @field_validator("id", "text")
    @classmethod
    def _validate_text(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)


class Evidence(BaseModel):
    id: str
    text: str
    title: str | None = None
    source_id: str | None = None
    uri: str | None = None
    rank: int | None = None
    score: float | None = None
    score_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _support_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            normalized = dict(data)
            if "id" not in normalized and "evidence_id" in normalized:
                normalized["id"] = normalized["evidence_id"]
            if "score" not in normalized and "retrieval_score" in normalized:
                normalized["score"] = normalized["retrieval_score"]
            return normalized
        return data

    @field_validator("id", "text")
    @classmethod
    def _validate_text(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)


class RequestContext(BaseModel):
    retrieval_query: str | None = None
    generator: str | None = None
    retriever: str | None = None
    framework: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingOptions(BaseModel):
    return_pair_scores: bool = True
    return_features: bool = True
    routing_policy: dict[str, Any] | None = None


class EvidenceSufficiencyRequest(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    id: str | None = None
    trace_id: str | None = None
    question: str
    answer: str
    claims: list[Claim] | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    context: RequestContext = Field(default_factory=RequestContext)
    options: RoutingOptions = Field(default_factory=RoutingOptions)

    @model_validator(mode="before")
    @classmethod
    def _support_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            normalized = dict(data)
            if "claims" not in normalized and "atomic_claims" in normalized:
                normalized["claims"] = normalized["atomic_claims"]
            return normalized
        return data

    @field_validator("question", "answer")
    @classmethod
    def _validate_text(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def _validate_request(self) -> EvidenceSufficiencyRequest:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol_version: {self.protocol_version}")

        if self.claims:
            duplicates = _duplicate_ids([claim.id for claim in self.claims])
            if duplicates:
                raise ValueError(f"duplicate claim ids: {sorted(duplicates)}")

        duplicates = _duplicate_ids([evidence.id for evidence in self.evidence])
        if duplicates:
            raise ValueError(f"duplicate evidence ids: {sorted(duplicates)}")

        return self


class LabelDistribution(BaseModel):
    supported: float = Field(ge=0.0)
    refuted: float = Field(ge=0.0)
    insufficient: float = Field(ge=0.0)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_distribution(self) -> LabelDistribution:
        _validate_probability_values(self.as_ordered_list(), "answer label")
        return self

    def as_ordered_list(self) -> list[float]:
        return [self.supported, self.refuted, self.insufficient]

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class PairDistribution(BaseModel):
    support: float = Field(ge=0.0)
    refute: float = Field(ge=0.0)
    neutral: float = Field(ge=0.0)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_distribution(self) -> PairDistribution:
        _validate_probability_values(self.as_ordered_list(), "pair relation")
        return self

    def as_ordered_list(self) -> list[float]:
        return [self.support, self.refute, self.neutral]


class PairRelationScore(BaseModel):
    claim_id: str
    evidence_id: str
    probs: PairDistribution
    pred: Literal["support", "refute", "neutral"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimAudit(BaseModel):
    id: str
    text: str
    status: Literal["supported", "refuted", "insufficient"]
    support_score: float
    refute_score: float
    neutral_score: float
    best_supporting_evidence_ids: list[str] = Field(default_factory=list)
    best_refuting_evidence_ids: list[str] = Field(default_factory=list)


class AuditTrace(BaseModel):
    claims: list[ClaimAudit] = Field(default_factory=list)
    pair_scores: list[PairRelationScore] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)


class ModelCapabilities(BaseModel):
    three_way_label: bool = True
    safe_unsafe_label: bool = True
    pair_audit: bool = False
    claim_level: bool = False
    multi_claim: bool = False
    calibrated_probs: bool = False
    selective_score: bool = False
    routing: bool = True
    supports_refuted_vs_insufficient: bool = True


class ModelInfo(BaseModel):
    implementation: str
    model_id: str | None = None
    pair_scorer: str | None = None
    aggregator: str | None = None
    calibration: str | None = None
    capabilities: ModelCapabilities


class EvidenceSufficiencyResult(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    id: str | None = None
    label: Literal["supported", "refuted", "insufficient"]
    safe_to_answer: bool
    action: Literal["answer", "abstain", "retrieve_more", "regenerate", "human_review"]
    probs: LabelDistribution
    confidence: float = Field(ge=0.0)
    selective_score: float | None = None
    threshold: float | None = None
    reason_codes: list[str] = Field(default_factory=list)
    message: dict[str, str | None] = Field(default_factory=dict)
    audit: AuditTrace = Field(default_factory=AuditTrace)
    model: ModelInfo
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_result(self) -> EvidenceSufficiencyResult:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol_version: {self.protocol_version}")
        if self.label not in ANSWER_LABELS:
            raise ValueError(f"invalid answer label: {self.label}")
        if self.action not in ACTIONS:
            raise ValueError(f"invalid action: {self.action}")
        return self
