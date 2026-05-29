import math

import pytest

from surerag.core.verifier import SureRAGVerifier
from surerag.protocol.schema import Claim, Evidence, PairDistribution, PairRelationScore


def _pair_score(
    evidence_id: str,
    support: float,
    refute: float,
    neutral: float,
) -> PairRelationScore:
    return PairRelationScore(
        claim_id="c1",
        evidence_id=evidence_id,
        probs=PairDistribution(support=support, refute=refute, neutral=neutral),
        pred=max(
            {"support": support, "refute": refute, "neutral": neutral},
            key={"support": support, "refute": refute, "neutral": neutral}.get,
        ),
    )


def test_pair_feature_semantics_use_product_conflict_and_js_disagreement() -> None:
    claims = [Claim(id="c1", text="A claim")]
    evidence = [
        Evidence(id="e1", text="Evidence one"),
        Evidence(id="e2", text="Evidence two"),
        Evidence(id="e3", text="Evidence three"),
    ]
    pair_scores = [
        _pair_score("e1", 0.9, 0.05, 0.05),
        _pair_score("e2", 0.1, 0.8, 0.1),
        _pair_score("e3", 0.2, 0.1, 0.7),
    ]

    _, features = SureRAGVerifier._audit_and_features(claims, evidence, pair_scores)

    assert features["support_max"] == pytest.approx(0.9)
    assert features["refute_max"] == pytest.approx(0.8)
    assert features["neutral_mean"] == pytest.approx((0.05 + 0.1 + 0.7) / 3)
    assert features["conflict_score"] == pytest.approx(0.9 * 0.8)
    assert features["evidence_disagreement_mean"] >= 0.0
    assert features["evidence_disagreement_max"] >= features["evidence_disagreement_mean"]
    assert all(math.isfinite(value) for value in features.values())


def test_missing_retrieval_scores_do_not_crash_feature_extraction() -> None:
    claims = [Claim(id="c1", text="A claim")]
    evidence = [Evidence(id="e1", text="Evidence without a score")]
    pair_scores = [_pair_score("e1", 0.6, 0.1, 0.3)]

    _, features = SureRAGVerifier._audit_and_features(claims, evidence, pair_scores)

    assert features["retrieval_scores_available"] == 0.0
    assert features["retrieval_uncertainty"] == 0.0


def test_identical_pair_distributions_have_near_zero_js_disagreement() -> None:
    claims = [Claim(id="c1", text="A claim")]
    evidence = [
        Evidence(id="e1", text="Evidence one"),
        Evidence(id="e2", text="Evidence two"),
    ]
    pair_scores = [
        _pair_score("e1", 0.7, 0.2, 0.1),
        _pair_score("e2", 0.7, 0.2, 0.1),
    ]

    _, features = SureRAGVerifier._audit_and_features(claims, evidence, pair_scores)

    assert features["evidence_disagreement_mean"] == pytest.approx(0.0, abs=1e-12)
    assert features["evidence_disagreement_max"] == pytest.approx(0.0, abs=1e-12)


def test_divergent_pair_distributions_have_positive_js_disagreement() -> None:
    claims = [Claim(id="c1", text="A claim")]
    evidence = [
        Evidence(id="e1", text="Evidence one"),
        Evidence(id="e2", text="Evidence two"),
    ]
    pair_scores = [
        _pair_score("e1", 0.98, 0.01, 0.01),
        _pair_score("e2", 0.01, 0.98, 0.01),
    ]

    _, features = SureRAGVerifier._audit_and_features(claims, evidence, pair_scores)

    assert features["evidence_disagreement_mean"] > 0.0
    assert features["evidence_disagreement_max"] > 0.0
