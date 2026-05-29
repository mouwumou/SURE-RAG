"""Default heuristic claim builder."""

from __future__ import annotations

import re

from surerag.protocol.schema import Claim, EvidenceSufficiencyRequest

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


class DefaultClaimBuilder:
    """Build claims without requiring an LLM or external framework."""

    max_claims = 8

    def __init__(self) -> None:
        self.last_warnings: list[str] = []

    def build_claims(self, request: EvidenceSufficiencyRequest) -> list[Claim]:
        self.last_warnings = []
        if request.claims:
            return list(request.claims)

        answer = request.answer.strip()
        if len(answer.split()) <= 12 or len(answer) <= 80:
            return [self._single_claim(request, answer)]

        parts = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(answer) if part.strip()]
        if len(parts) <= 1:
            return [self._single_claim(request, answer)]

        self.last_warnings.append("HEURISTIC_CLAIM_DECOMPOSITION")
        claims: list[Claim] = []
        for index, sentence in enumerate(parts[: self.max_claims], start=1):
            claims.append(Claim(id=f"c{index}", text=sentence, source="heuristic"))
        return claims

    @staticmethod
    def _single_claim(request: EvidenceSufficiencyRequest, answer: str) -> Claim:
        return Claim(
            id="c1",
            text=f'For the question "{request.question.strip()}", the answer is "{answer}".',
            source="heuristic",
        )
