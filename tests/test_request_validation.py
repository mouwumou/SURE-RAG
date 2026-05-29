import pytest

from surerag.protocol.schema import EvidenceSufficiencyRequest


def test_duplicate_evidence_ids_fail() -> None:
    with pytest.raises(ValueError, match="duplicate evidence"):
        EvidenceSufficiencyRequest.model_validate(
            {
                "question": "q",
                "answer": "a",
                "evidence": [{"id": "e1", "text": "x"}, {"id": "e1", "text": "y"}],
            }
        )


def test_duplicate_claim_ids_fail() -> None:
    with pytest.raises(ValueError, match="duplicate claim"):
        EvidenceSufficiencyRequest.model_validate(
            {
                "question": "q",
                "answer": "a",
                "claims": [{"id": "c1", "text": "x"}, {"id": "c1", "text": "y"}],
                "evidence": [],
            }
        )


def test_training_style_aliases_parse() -> None:
    request = EvidenceSufficiencyRequest.model_validate(
        {
            "question": "q",
            "answer": "a",
            "atomic_claims": [{"claim_id": "c1", "claim": "a is true"}],
            "evidence": [{"evidence_id": "e1", "text": "a is true", "retrieval_score": 0.7}],
        }
    )
    assert request.claims is not None
    assert request.claims[0].id == "c1"
    assert request.claims[0].text == "a is true"
    assert request.evidence[0].id == "e1"
    assert request.evidence[0].score == 0.7


def test_empty_evidence_allowed_for_inference() -> None:
    request = EvidenceSufficiencyRequest.model_validate(
        {"question": "Who wrote The Hobbit?", "answer": "J.R.R. Tolkien", "evidence": []}
    )
    assert request.evidence == []
