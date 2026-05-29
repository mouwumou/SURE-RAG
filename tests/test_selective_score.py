import pytest

from surerag import SureRAGVerifier
from surerag.core.policy import RoutingPolicy
from surerag.core.selective import compute_selective_score
from surerag.protocol.schema import LabelDistribution


def test_selective_score_uses_explicit_beta_and_can_be_negative() -> None:
    probs = LabelDistribution(supported=0.2, refuted=0.3, insufficient=0.5)
    features = {
        "claim_coverage_deficit": 1.0,
        "evidence_disagreement_mean": 1.0,
        "conflict_score": 1.0,
        "retrieval_uncertainty": 1.0,
    }

    score = compute_selective_score(probs, features, beta=10.0)

    assert score < 0.0


def test_selective_score_beta_zero_equals_supported_probability() -> None:
    probs = LabelDistribution(supported=0.42, refuted=0.23, insufficient=0.35)
    features = {
        "claim_coverage_deficit": 1.0,
        "evidence_disagreement_mean": 1.0,
        "conflict_score": 1.0,
        "retrieval_uncertainty": 1.0,
    }

    assert compute_selective_score(probs, features, beta=0.0) == pytest.approx(0.42)


def test_larger_selective_beta_decreases_score_when_uncertainty_is_positive() -> None:
    probs = LabelDistribution(supported=0.7, refuted=0.2, insufficient=0.1)
    features = {
        "claim_coverage_deficit": 0.3,
        "evidence_disagreement_mean": 0.2,
        "conflict_score": 0.1,
        "retrieval_uncertainty": 0.1,
    }

    small_beta = compute_selective_score(probs, features, beta=0.1)
    large_beta = compute_selective_score(probs, features, beta=1.0)

    assert large_beta < small_beta


def test_routing_policy_selective_beta_affects_verifier_score() -> None:
    request = {
        "question": "Who wrote The Hobbit?",
        "answer": "J.R.R. Tolkien",
        "evidence": [{"id": "e1", "text": "The Hobbit was written by J.R.R. Tolkien."}],
    }
    low_beta_verifier = SureRAGVerifier.toy()
    low_beta_verifier.policy = RoutingPolicy(selective_beta=0.0)
    high_beta_verifier = SureRAGVerifier.toy()
    high_beta_verifier.policy = RoutingPolicy(selective_beta=1.0)

    low_beta_result = low_beta_verifier.verify(request)
    high_beta_result = high_beta_verifier.verify(request)

    assert low_beta_result.selective_score is not None
    assert high_beta_result.selective_score is not None
    assert high_beta_result.selective_score < low_beta_result.selective_score


def test_supported_below_threshold_still_abstains() -> None:
    decision = RoutingPolicy.default().route(
        label="supported",
        confidence=0.9,
        selective_score=0.1,
    )
    assert decision.action == "abstain"


def test_invalid_selective_beta_fails() -> None:
    with pytest.raises(ValueError):
        compute_selective_score(
            LabelDistribution(supported=0.8, refuted=0.1, insufficient=0.1),
            {},
            beta=-0.1,
        )
