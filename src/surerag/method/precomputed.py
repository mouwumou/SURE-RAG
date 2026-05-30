"""Verify requests from precomputed pair-level relation scores.

This module is for method demonstrations only. It does not score pairs with a
trained model and does not train an aggregator.
"""

from __future__ import annotations

from typing import Any

from surerag.core.claim_builder import DefaultClaimBuilder
from surerag.core.policy import RoutingPolicy
from surerag.core.selective import compute_selective_score, predictive_entropy
from surerag.core.verifier import SureRAGVerifier
from surerag.protocol.labels import PROTOCOL_VERSION
from surerag.protocol.schema import (
    AuditTrace,
    EvidenceSufficiencyRequest,
    EvidenceSufficiencyResult,
    LabelDistribution,
    ModelCapabilities,
    ModelInfo,
    PairRelationScore,
)

DEMO_WARNING = "PRECOMPUTED_PAIR_SCORES_DEMO_ONLY"


def verify_with_precomputed_pair_scores(
    request: EvidenceSufficiencyRequest | dict[str, Any],
    pair_scores: list[PairRelationScore | dict[str, Any]],
    policy: RoutingPolicy | None = None,
) -> EvidenceSufficiencyResult:
    """Aggregate precomputed pair scores into a protocol result.

    The input pair scores are assumed to be synthetic or externally produced.
    This helper demonstrates the SURE-RAG aggregation concept without shipping
    a neural pair scorer, sklearn aggregator, or model artifact.
    """

    parsed = (
        request
        if isinstance(request, EvidenceSufficiencyRequest)
        else EvidenceSufficiencyRequest.model_validate(request)
    )
    active_policy = policy or RoutingPolicy.default()
    claims = parsed.claims or DefaultClaimBuilder().build_claims(parsed)
    scores = [
        score if isinstance(score, PairRelationScore) else PairRelationScore.model_validate(score)
        for score in pair_scores
    ]

    if not parsed.evidence:
        probs = LabelDistribution(supported=0.0, refuted=0.0, insufficient=1.0)
        decision = active_policy.route(
            label="insufficient",
            confidence=1.0,
            selective_score=0.0,
            features={"evidence_count": 0.0},
            no_evidence=True,
        )
        return EvidenceSufficiencyResult(
            protocol_version=PROTOCOL_VERSION,
            id=parsed.id,
            label="insufficient",
            safe_to_answer=False,
            action=decision.action,
            probs=probs,
            confidence=1.0,
            selective_score=0.0,
            threshold=active_policy.selective_score_gte,
            reason_codes=decision.reason_codes,
            message={
                "developer": "No runtime evidence was supplied for verification.",
                "user": None,
            },
            audit=AuditTrace(features={"evidence_count": 0.0, "num_claims": float(len(claims))}),
            model=_demo_model_info(),
            warnings=[DEMO_WARNING],
        )

    _validate_pair_references(parsed, scores)
    claim_audits, features = SureRAGVerifier._audit_and_features(claims, parsed.evidence, scores)
    probs = _aggregate_demo_probs(features)
    label = _label_from_claim_audits(claim_audits)
    confidence = max(probs.as_ordered_list())
    features["predictive_entropy"] = predictive_entropy(probs)
    selective_score = compute_selective_score(
        probs,
        features,
        beta=active_policy.selective_beta,
    )
    decision = active_policy.route(
        label=label,
        confidence=confidence,
        selective_score=selective_score,
        features=features,
    )

    return EvidenceSufficiencyResult(
        protocol_version=PROTOCOL_VERSION,
        id=parsed.id,
        label=label,
        safe_to_answer=label == "supported" and decision.action == "answer",
        action=decision.action,
        probs=probs,
        confidence=confidence,
        selective_score=selective_score,
        threshold=active_policy.selective_score_gte,
        reason_codes=decision.reason_codes,
        message={"developer": "Aggregated precomputed pair scores for method demo.", "user": None},
        audit=AuditTrace(claims=claim_audits, pair_scores=scores, features=features),
        model=_demo_model_info(),
        warnings=[DEMO_WARNING],
    )


def _aggregate_demo_probs(features: dict[str, float]) -> LabelDistribution:
    raw = {
        "supported": (
            0.05
            + 0.80 * features.get("supported_claim_fraction", 0.0)
            + 0.30 * features.get("support_max", 0.0)
            - 0.40 * features.get("refute_max", 0.0)
        ),
        "refuted": (
            0.05
            + 0.85 * features.get("refuted_claim_fraction", 0.0)
            + 0.55 * features.get("refute_max", 0.0)
        ),
        "insufficient": (
            0.05
            + 0.90 * features.get("insufficient_claim_fraction", 0.0)
            + 0.45 * features.get("claim_coverage_deficit", 1.0)
            + 0.25 * features.get("neutral_mean", 0.0)
        ),
    }
    clipped = {label: max(0.0, value) for label, value in raw.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return LabelDistribution(supported=0.0, refuted=0.0, insufficient=1.0)
    return LabelDistribution(**{label: value / total for label, value in clipped.items()})


def _label_from_claim_audits(claim_audits: list[Any]) -> str:
    statuses = [audit.status for audit in claim_audits]
    if any(status == "refuted" for status in statuses):
        return "refuted"
    if statuses and all(status == "supported" for status in statuses):
        return "supported"
    return "insufficient"


def _validate_pair_references(
    request: EvidenceSufficiencyRequest,
    pair_scores: list[PairRelationScore],
) -> None:
    claims = request.claims or DefaultClaimBuilder().build_claims(request)
    claim_ids = {claim.id for claim in claims}
    evidence_ids = {evidence.id for evidence in request.evidence}
    for score in pair_scores:
        if score.claim_id not in claim_ids:
            raise ValueError(f"unknown claim_id in pair score: {score.claim_id}")
        if score.evidence_id not in evidence_ids:
            raise ValueError(f"unknown evidence_id in pair score: {score.evidence_id}")


def _demo_model_info() -> ModelInfo:
    return ModelInfo(
        implementation="sure-rag-precomputed-demo",
        model_id="synthetic-precomputed-pair-scores",
        pair_scorer="precomputed_demo_scores",
        aggregator="rule_based_demo_aggregation",
        calibration="none",
        capabilities=ModelCapabilities(
            three_way_label=True,
            safe_unsafe_label=True,
            pair_audit=True,
            claim_level=True,
            multi_claim=True,
            calibrated_probs=False,
            selective_score=True,
            routing=True,
            supports_refuted_vs_insufficient=True,
        ),
    )
