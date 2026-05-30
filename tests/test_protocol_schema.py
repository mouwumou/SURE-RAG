import pytest

from surerag import SureRAGVerifier
from surerag.protocol.json_schema import (
    export_protocol_schema,
    request_json_schema,
    result_json_schema,
)
from surerag.protocol.schema import EvidenceSufficiencyRequest, EvidenceSufficiencyResult


def test_minimal_request_parses() -> None:
    request = EvidenceSufficiencyRequest.model_validate(
        {
            "question": "Who wrote The Hobbit?",
            "answer": "J.R.R. Tolkien",
            "evidence": [{"id": "e1", "text": "The Hobbit was written by J.R.R. Tolkien."}],
        }
    )
    assert request.protocol_version == "evidence-sufficiency.v1"
    assert request.evidence[0].score is None


def test_json_schema_export_contains_request_and_result() -> None:
    schema = export_protocol_schema()
    assert schema["protocol_version"] == "evidence-sufficiency.v1"
    assert request_json_schema()["title"] == "EvidenceSufficiencyRequest"
    assert result_json_schema()["title"] == "EvidenceSufficiencyResult"


def test_result_schema_validates_toy_output() -> None:
    result = SureRAGVerifier.toy().verify(
        {
            "question": "Who wrote The Hobbit?",
            "answer": "J.R.R. Tolkien",
            "evidence": [{"id": "e1", "text": "The Hobbit was written by J.R.R. Tolkien."}],
        }
    )
    assert EvidenceSufficiencyResult.model_validate(result.model_dump()) == result


def _result_payload(label: str, action: str, safe_to_answer: bool) -> dict[str, object]:
    return {
        "label": label,
        "safe_to_answer": safe_to_answer,
        "action": action,
        "probs": {"supported": 0.7, "refuted": 0.2, "insufficient": 0.1},
        "confidence": 0.7,
        "model": {
            "implementation": "test",
            "model_id": "test-model",
            "capabilities": {
                "three_way_label": True,
                "safe_unsafe_label": True,
                "routing": True,
                "supports_refuted_vs_insufficient": True,
            },
        },
    }


@pytest.mark.parametrize(
    ("label", "action", "safe_to_answer"),
    [
        ("supported", "answer", True),
        ("supported", "abstain", False),
    ],
)
def test_result_schema_accepts_valid_safe_to_answer_invariants(
    label: str,
    action: str,
    safe_to_answer: bool,
) -> None:
    result = EvidenceSufficiencyResult.model_validate(
        _result_payload(label, action, safe_to_answer)
    )
    assert result.safe_to_answer is safe_to_answer


@pytest.mark.parametrize(
    ("label", "action", "safe_to_answer"),
    [
        ("supported", "answer", False),
        ("supported", "abstain", True),
        ("refuted", "regenerate", True),
        ("insufficient", "retrieve_more", True),
        ("refuted", "answer", True),
    ],
)
def test_result_schema_rejects_invalid_safe_to_answer_invariants(
    label: str,
    action: str,
    safe_to_answer: bool,
) -> None:
    with pytest.raises(ValueError, match="safe_to_answer must equal"):
        EvidenceSufficiencyResult.model_validate(_result_payload(label, action, safe_to_answer))


def test_result_schema_validates_empty_evidence_toy_output() -> None:
    result = SureRAGVerifier.toy().verify(
        {"question": "Who wrote The Hobbit?", "answer": "J.R.R. Tolkien", "evidence": []}
    )
    validated = EvidenceSufficiencyResult.model_validate(result.model_dump())
    assert validated.label == "insufficient"
    assert validated.safe_to_answer is False


def test_invalid_empty_question_and_answer_fail() -> None:
    with pytest.raises(ValueError):
        EvidenceSufficiencyRequest.model_validate({"question": " ", "answer": "x"})
    with pytest.raises(ValueError):
        EvidenceSufficiencyRequest.model_validate({"question": "x", "answer": ""})
