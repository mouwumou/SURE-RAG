import math

import pytest

from surerag import SureRAGVerifier
from surerag.protocol.schema import LabelDistribution, PairDistribution


def test_label_distribution_accepts_normalized_values() -> None:
    probs = LabelDistribution(supported=0.5, refuted=0.25, insufficient=0.25)
    assert probs.as_ordered_list() == [0.5, 0.25, 0.25]


def test_label_distribution_rejects_bad_sum() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        LabelDistribution(supported=0.5, refuted=0.25, insufficient=0.20)


def test_pair_distribution_accepts_normalized_values() -> None:
    probs = PairDistribution(support=0.8, refute=0.1, neutral=0.1)
    assert probs.as_ordered_list() == [0.8, 0.1, 0.1]


def test_pair_distribution_rejects_bad_sum() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        PairDistribution(support=0.5, refute=0.25, neutral=0.20)


def test_negative_or_non_finite_probabilities_fail() -> None:
    with pytest.raises(ValueError):
        LabelDistribution(supported=-0.1, refuted=0.6, insufficient=0.5)
    with pytest.raises(ValueError):
        PairDistribution(support=math.nan, refute=0.1, neutral=0.9)


def test_tiny_probability_drift_within_tolerance_passes() -> None:
    probs = LabelDistribution(supported=0.5, refuted=0.25, insufficient=0.2500005)
    assert sum(probs.as_ordered_list()) == pytest.approx(1.0000005)


def test_toy_verifier_returns_valid_normalized_distributions() -> None:
    result = SureRAGVerifier.toy().verify(
        {
            "question": "Who wrote The Hobbit?",
            "answer": "J.R.R. Tolkien",
            "evidence": [{"id": "e1", "text": "The Hobbit was written by J.R.R. Tolkien."}],
        }
    )
    assert sum(result.probs.as_ordered_list()) == pytest.approx(1.0)
    for pair_score in result.audit.pair_scores:
        assert sum(pair_score.probs.as_ordered_list()) == pytest.approx(1.0)
