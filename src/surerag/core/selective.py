"""Selective scoring utilities."""

from __future__ import annotations

import math

from surerag.protocol.schema import LabelDistribution


def predictive_entropy(probs: LabelDistribution) -> float:
    values = [value for value in probs.as_ordered_list() if value > 0]
    if not values:
        return 0.0
    entropy = -sum(value * math.log(value) for value in values)
    return entropy / math.log(3)


def compute_selective_score(
    probs: LabelDistribution,
    features: dict[str, float],
    *,
    beta: float,
) -> float:
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("selective beta must be finite and non-negative")
    uncertainty = (
        predictive_entropy(probs)
        + features.get("claim_coverage_deficit", 0.0)
        + features.get("evidence_disagreement_mean", 0.0)
        + features.get("conflict_score", 0.0)
        + features.get("retrieval_uncertainty", 0.0)
    )
    return probs.supported - beta * uncertainty
