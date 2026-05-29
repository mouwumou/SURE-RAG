from surerag.core.claim_builder import DefaultClaimBuilder
from surerag.protocol.schema import EvidenceSufficiencyRequest


def test_uses_provided_claims_unchanged() -> None:
    request = EvidenceSufficiencyRequest.model_validate(
        {
            "question": "q",
            "answer": "a",
            "claims": [{"id": "custom", "text": "custom claim"}],
            "evidence": [],
        }
    )
    claims = DefaultClaimBuilder().build_claims(request)
    assert claims[0].id == "custom"
    assert claims[0].source == "provided"


def test_builds_single_heuristic_claim() -> None:
    request = EvidenceSufficiencyRequest.model_validate({"question": "q", "answer": "a"})
    builder = DefaultClaimBuilder()
    claims = builder.build_claims(request)
    assert len(claims) == 1
    assert claims[0].source == "heuristic"
    assert builder.last_warnings == []


def test_builds_multi_sentence_claims_with_warning() -> None:
    request = EvidenceSufficiencyRequest.model_validate(
        {
            "question": "q",
            "answer": (
                "The first answer sentence contains enough words for decomposition. "
                "The second answer sentence also contains enough context for splitting."
            ),
        }
    )
    builder = DefaultClaimBuilder()
    claims = builder.build_claims(request)
    assert [claim.text for claim in claims] == [
        "The first answer sentence contains enough words for decomposition.",
        "The second answer sentence also contains enough context for splitting.",
    ]
    assert builder.last_warnings == ["HEURISTIC_CLAIM_DECOMPOSITION"]


def test_short_answer_with_periods_stays_single_claim() -> None:
    request = EvidenceSufficiencyRequest.model_validate(
        {"question": "Who wrote The Hobbit?", "answer": "J.R.R. Tolkien"}
    )
    claims = DefaultClaimBuilder().build_claims(request)
    assert len(claims) == 1
