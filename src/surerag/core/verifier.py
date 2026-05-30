"""Reference verifier pipeline for SURE-RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from surerag.core.claim_builder import DefaultClaimBuilder
from surerag.core.policy import RoutingPolicy
from surerag.core.selective import compute_selective_score, predictive_entropy
from surerag.models.interfaces import (
    Aggregator,
    Calibrator,
    ClaimBuilder,
    ClaimEvidencePair,
    PairRelationScorer,
)
from surerag.models.toy import TOY_WARNING, ToyLexicalAggregator, ToyLexicalPairScorer
from surerag.protocol.labels import ANSWER_LABELS, PROTOCOL_VERSION
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
    PairRelationScore,
)


def _normalize_distribution(probs: LabelDistribution) -> LabelDistribution:
    raw = {
        "supported": max(0.0, probs.supported),
        "refuted": max(0.0, probs.refuted),
        "insufficient": max(0.0, probs.insufficient),
    }
    total = sum(raw.values())
    if total <= 0:
        return LabelDistribution(supported=0.0, refuted=0.0, insufficient=1.0)
    return LabelDistribution(**{key: value / total for key, value in raw.items()})


def _default_model_info() -> ModelInfo:
    return ModelInfo(
        implementation="sure-rag-reference",
        model_id="toy-lexical",
        pair_scorer="toy_lexical",
        aggregator="toy_heuristic",
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


class SureRAGVerifier:
    def __init__(
        self,
        claim_builder: ClaimBuilder,
        pair_scorer: PairRelationScorer,
        aggregator: Aggregator,
        calibrator: Calibrator | None = None,
        policy: RoutingPolicy | None = None,
        model_info: ModelInfo | None = None,
    ) -> None:
        self.claim_builder = claim_builder
        self.pair_scorer = pair_scorer
        self.aggregator = aggregator
        self.calibrator = calibrator
        self.policy = policy or RoutingPolicy.default()
        self.model_info = model_info or _default_model_info()

    @classmethod
    def toy(cls) -> SureRAGVerifier:
        return cls(
            claim_builder=DefaultClaimBuilder(),
            pair_scorer=ToyLexicalPairScorer(),
            aggregator=ToyLexicalAggregator(),
            policy=RoutingPolicy.default(),
            model_info=_default_model_info(),
        )

    @classmethod
    def from_artifact(cls, path: str | Path) -> SureRAGVerifier:
        from surerag.artifact import load_artifact

        return load_artifact(path)

    @classmethod
    def from_pretrained(cls, model_id_or_path: str) -> SureRAGVerifier:
        path = Path(model_id_or_path)
        if path.exists():
            return cls.from_artifact(path)
        raise NotImplementedError(
            "from_pretrained currently supports local artifacts only in the initial skeleton"
        )

    def verify(
        self, request: EvidenceSufficiencyRequest | dict[str, Any]
    ) -> EvidenceSufficiencyResult:
        parsed = (
            request
            if isinstance(request, EvidenceSufficiencyRequest)
            else EvidenceSufficiencyRequest.model_validate(request)
        )
        policy = self._policy_for_request(parsed)
        claims = self.claim_builder.build_claims(parsed)
        warnings = self._component_warnings()
        warnings.extend(getattr(self.claim_builder, "last_warnings", []))

        if not parsed.evidence:
            return self._no_evidence_result(parsed, claims, policy, warnings)

        pairs = self._build_pairs(parsed, claims)
        pair_scores = self.pair_scorer.score_pairs(pairs)
        claim_audits, features = self._audit_and_features(claims, parsed.evidence, pair_scores)

        probs = _normalize_distribution(self.aggregator.predict_proba(features))
        if self.calibrator is not None:
            probs = _normalize_distribution(self.calibrator.calibrate(probs))

        label = self._label_from_probs(probs)
        confidence = max(probs.as_ordered_list())
        features["predictive_entropy"] = predictive_entropy(probs)
        selective_score = compute_selective_score(probs, features, beta=policy.selective_beta)
        decision = policy.route(
            label=label,
            confidence=confidence,
            selective_score=selective_score,
            features=features,
        )

        audit = AuditTrace(
            claims=claim_audits,
            pair_scores=pair_scores if parsed.options.return_pair_scores else [],
            features=features if parsed.options.return_features else {},
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
            threshold=policy.selective_score_gte,
            reason_codes=decision.reason_codes,
            message={"developer": self._developer_message(label, decision.action), "user": None},
            audit=audit,
            model=self.model_info,
            warnings=warnings,
            errors=[],
        )

    def _policy_for_request(self, request: EvidenceSufficiencyRequest) -> RoutingPolicy:
        if request.options.routing_policy is None:
            return self.policy
        return RoutingPolicy.model_validate(request.options.routing_policy)

    def _component_warnings(self) -> list[str]:
        warnings: list[str] = []
        if getattr(self.pair_scorer, "warning", None) == TOY_WARNING:
            warnings.append(TOY_WARNING)
        return warnings

    def _no_evidence_result(
        self,
        request: EvidenceSufficiencyRequest,
        claims: list[Claim],
        policy: RoutingPolicy,
        warnings: list[str],
    ) -> EvidenceSufficiencyResult:
        probs = LabelDistribution(supported=0.0, refuted=0.0, insufficient=1.0)
        decision = policy.route(
            label="insufficient",
            confidence=1.0,
            selective_score=0.0,
            features={"evidence_count": 0.0},
            no_evidence=True,
        )
        claim_audits = [
            ClaimAudit(
                id=claim.id,
                text=claim.text,
                status="insufficient",
                support_score=0.0,
                refute_score=0.0,
                neutral_score=1.0,
            )
            for claim in claims
        ]
        audit = AuditTrace(
            claims=claim_audits,
            pair_scores=[],
            features={"num_claims": float(len(claims)), "evidence_count": 0.0},
        )
        return EvidenceSufficiencyResult(
            protocol_version=PROTOCOL_VERSION,
            id=request.id,
            label="insufficient",
            safe_to_answer=False,
            action=decision.action,
            probs=probs,
            confidence=1.0,
            selective_score=0.0,
            threshold=policy.selective_score_gte,
            reason_codes=decision.reason_codes,
            message={
                "developer": "No runtime evidence was supplied for verification.",
                "user": None,
            },
            audit=audit,
            model=self.model_info,
            warnings=warnings,
            errors=[],
        )

    @staticmethod
    def _build_pairs(
        request: EvidenceSufficiencyRequest,
        claims: list[Claim],
    ) -> list[ClaimEvidencePair]:
        pairs: list[ClaimEvidencePair] = []
        for claim in claims:
            for evidence in request.evidence:
                pairs.append(
                    ClaimEvidencePair(
                        request_id=request.id,
                        question=request.question,
                        answer=request.answer,
                        claim_id=claim.id,
                        claim_text=claim.text,
                        evidence_id=evidence.id,
                        evidence_text=evidence.text,
                        metadata={
                            "evidence_rank": evidence.rank,
                            "score_type": evidence.score_type,
                        },
                    )
                )
        return pairs

    @staticmethod
    def _audit_and_features(
        claims: list[Claim],
        evidence: list[Evidence],
        pair_scores: list[PairRelationScore],
    ) -> tuple[list[ClaimAudit], dict[str, float]]:
        pair_by_claim: dict[str, list[PairRelationScore]] = {claim.id: [] for claim in claims}
        for score in pair_scores:
            pair_by_claim.setdefault(score.claim_id, []).append(score)

        audits: list[ClaimAudit] = []
        support_values: list[float] = []
        refute_values: list[float] = []
        neutral_values: list[float] = []

        for claim in claims:
            scores = pair_by_claim.get(claim.id, [])
            support_scores = [score.probs.support for score in scores]
            refute_scores = [score.probs.refute for score in scores]
            neutral_scores = [score.probs.neutral for score in scores]
            support_max = max(support_scores, default=0.0)
            refute_max = max(refute_scores, default=0.0)
            neutral_max = max(neutral_scores, default=1.0)
            status = _claim_status(support_max, refute_max)

            support_values.extend(support_scores)
            refute_values.extend(refute_scores)
            neutral_values.extend(neutral_scores)

            audits.append(
                ClaimAudit(
                    id=claim.id,
                    text=claim.text,
                    status=status,
                    support_score=support_max,
                    refute_score=refute_max,
                    neutral_score=neutral_max,
                    best_supporting_evidence_ids=[
                        score.evidence_id for score in scores if score.probs.support == support_max
                    ][:3],
                    best_refuting_evidence_ids=[
                        score.evidence_id for score in scores if score.probs.refute == refute_max
                    ][:3],
                )
            )

        num_claims = max(1, len(claims))
        supported_fraction = sum(1 for audit in audits if audit.status == "supported") / num_claims
        refuted_fraction = sum(1 for audit in audits if audit.status == "refuted") / num_claims
        insufficient_fraction = (
            sum(1 for audit in audits if audit.status == "insufficient") / num_claims
        )
        retrieval_features = _retrieval_features(evidence)

        features = {
            "num_claims": float(len(claims)),
            "supported_claim_fraction": supported_fraction,
            "refuted_claim_fraction": refuted_fraction,
            "insufficient_claim_fraction": insufficient_fraction,
            "claim_coverage_deficit": 1.0 - supported_fraction,
            "support_max": _max(support_values),
            "support_mean": _mean(support_values),
            "support_min": _min(support_values),
            "support_std": _std(support_values),
            "refute_max": _max(refute_values),
            "refute_mean": _mean(refute_values),
            "refute_min": _min(refute_values),
            "refute_std": _std(refute_values),
            "neutral_max": _max(neutral_values),
            "neutral_mean": _mean(neutral_values),
            "neutral_min": _min(neutral_values),
            "neutral_std": _std(neutral_values),
            "conflict_score": _max(support_values) * _max(refute_values),
            "evidence_disagreement_mean": _disagreement_mean(pair_scores),
            "evidence_disagreement_max": _disagreement_max(pair_scores),
            **retrieval_features,
        }
        return audits, features

    @staticmethod
    def _label_from_probs(probs: LabelDistribution) -> str:
        values = probs.as_dict()
        return max(ANSWER_LABELS, key=lambda label: values[label])

    @staticmethod
    def _developer_message(label: str, action: str) -> str:
        if label == "supported" and action == "answer":
            return "Evidence supports the candidate answer above the routing threshold."
        if label == "refuted":
            return "Evidence appears to contradict the candidate answer."
        if label == "insufficient":
            return "Evidence is insufficient for the candidate answer."
        return "Verification completed with a conservative routing action."


def _claim_status(support_max: float, refute_max: float) -> str:
    if refute_max >= 0.5:
        return "refuted"
    if support_max >= 0.5 and support_max - refute_max >= 0.1:
        return "supported"
    return "insufficient"


def _retrieval_features(evidence: list[Evidence]) -> dict[str, float]:
    scores = [item.score for item in evidence if item.score is not None]
    token_count = sum(len(item.text.split()) for item in evidence)
    if not scores:
        return {
            "retrieval_uncertainty": 0.0,
            "evidence_count": float(len(evidence)),
            "evidence_token_count": float(token_count),
            "retrieval_scores_available": 0.0,
        }
    numeric = np.array(scores, dtype=float)
    if not np.all((0.0 <= numeric) & (numeric <= 1.0)):
        min_score = float(np.min(numeric))
        max_score = float(np.max(numeric))
        if max_score > min_score:
            numeric = (numeric - min_score) / (max_score - min_score)
        else:
            numeric = np.ones_like(numeric)
    return {
        "retrieval_uncertainty": float(1.0 - np.mean(numeric)),
        "evidence_count": float(len(evidence)),
        "evidence_token_count": float(token_count),
        "retrieval_scores_available": 1.0,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return max(values, default=0.0)


def _min(values: list[float]) -> float:
    return min(values, default=0.0)


def _disagreement_mean(pair_scores: list[PairRelationScore]) -> float:
    values = _js_disagreement_values(pair_scores)
    return float(np.mean(values)) if values else 0.0


def _disagreement_max(pair_scores: list[PairRelationScore]) -> float:
    values = _js_disagreement_values(pair_scores)
    return max(values, default=0.0)


def _js_disagreement_values(pair_scores: list[PairRelationScore]) -> list[float]:
    if not pair_scores:
        return []
    matrix = np.array(
        [
            [score.probs.support, score.probs.refute, score.probs.neutral]
            for score in pair_scores
        ],
        dtype=float,
    )
    mean_distribution = np.mean(matrix, axis=0)
    return [_js_divergence(row, mean_distribution) for row in matrix]


def _entropy(values: np.ndarray) -> float:
    clipped = np.clip(values, 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    clipped_p = np.clip(p, 1e-12, 1.0)
    clipped_q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(clipped_p * np.log(clipped_p / clipped_q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    mean = 0.5 * (p + q)
    return 0.5 * (_kl_divergence(p, mean) + _kl_divergence(q, mean))
