"""Deterministic CPU-only toy components for smoke tests and examples."""

from __future__ import annotations

import math
import re
from collections import Counter

from surerag.models.interfaces import ClaimEvidencePair
from surerag.protocol.labels import ANSWER_LABELS
from surerag.protocol.schema import LabelDistribution, PairDistribution, PairRelationScore

TOY_WARNING = "TOY_BACKEND_NOT_FOR_PRODUCTION"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}
_NEGATION_TERMS = {"no", "not", "never", "none", "false", "incorrect", "cannot", "neither"}


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def _normalize(raw: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(0.0, float(value)) for key, value in raw.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {key: 1.0 / len(clipped) for key in clipped}
    return {key: value / total for key, value in clipped.items()}


def _cosine_overlap(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    shared_tokens = left_counts.keys() & right_counts.keys()
    dot = sum(left_counts[token] * right_counts[token] for token in shared_tokens)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _fraction_present(needles: list[str], haystack: list[str]) -> float:
    if not needles:
        return 0.0
    haystack_set = set(haystack)
    return sum(1 for token in set(needles) if token in haystack_set) / max(1, len(set(needles)))


class ToyLexicalPairScorer:
    """A tiny lexical relation scorer for deterministic CPU smoke tests."""

    warning = TOY_WARNING

    def score_pairs(self, pairs: list[ClaimEvidencePair]) -> list[PairRelationScore]:
        scores: list[PairRelationScore] = []
        for pair in pairs:
            claim_tokens = _tokens(f"{pair.question} {pair.answer} {pair.claim_text}")
            answer_tokens = _tokens(pair.answer)
            evidence_tokens = _tokens(pair.evidence_text)

            lexical_overlap = _cosine_overlap(claim_tokens, evidence_tokens)
            answer_overlap = _fraction_present(answer_tokens, evidence_tokens)
            negation_mismatch = self._negation_mismatch(pair, evidence_tokens)

            support_signal = min(1.0, 0.75 * lexical_overlap + 0.45 * answer_overlap)
            refute_signal = min(1.0, 0.72 * negation_mismatch)
            neutral_signal = max(0.05, 1.0 - max(support_signal, refute_signal))

            if refute_signal > 0:
                support_signal *= 0.55

            probs = _normalize(
                {
                    "support": 0.05 + support_signal,
                    "refute": 0.05 + refute_signal,
                    "neutral": 0.05 + neutral_signal,
                }
            )
            pred = max(("support", "refute", "neutral"), key=lambda label: probs[label])
            scores.append(
                PairRelationScore(
                    claim_id=pair.claim_id,
                    evidence_id=pair.evidence_id,
                    probs=PairDistribution(**probs),
                    pred=pred,
                    metadata={"backend": "toy_lexical"},
                )
            )
        return scores

    @staticmethod
    def _negation_mismatch(pair: ClaimEvidencePair, evidence_tokens: list[str]) -> float:
        claim_text = f"{pair.answer} {pair.claim_text}".lower()
        claim_has_negation = any(term in _tokens(claim_text) for term in _NEGATION_TERMS)
        evidence_has_negation = any(term in evidence_tokens for term in _NEGATION_TERMS)
        answer_overlap = _fraction_present(_tokens(pair.answer), evidence_tokens)
        if evidence_has_negation != claim_has_negation and answer_overlap > 0:
            return 1.0
        if evidence_has_negation and not claim_has_negation:
            return 0.65
        return 0.0


class ToyLexicalAggregator:
    """Heuristic answer-level aggregator for the toy backend."""

    label_order = ANSWER_LABELS

    def predict_proba(self, features: dict[str, float]) -> LabelDistribution:
        support = features.get("support_max", 0.0)
        refute = features.get("refute_max", 0.0)
        neutral = features.get("neutral_mean", 0.0)
        coverage = features.get("supported_claim_fraction", 0.0)
        deficit = features.get("claim_coverage_deficit", 1.0)
        conflict = features.get("conflict_score", 0.0)

        raw = {
            "supported": 0.02 + 0.78 * support + 0.22 * coverage - 0.24 * refute,
            "refuted": 0.02 + 0.88 * refute + 0.10 * conflict,
            "insufficient": 0.05 + 0.50 * neutral + 0.42 * deficit + 0.12 * (1.0 - support),
        }
        return LabelDistribution(**_normalize(raw))
