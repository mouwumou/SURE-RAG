import pytest

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
    from surerag import SureRAGVerifier

    result = SureRAGVerifier.toy().verify(
        {
            "question": "Who wrote The Hobbit?",
            "answer": "J.R.R. Tolkien",
            "evidence": [{"id": "e1", "text": "The Hobbit was written by J.R.R. Tolkien."}],
        }
    )
    assert EvidenceSufficiencyResult.model_validate(result.model_dump()) == result


def test_invalid_empty_question_and_answer_fail() -> None:
    with pytest.raises(ValueError):
        EvidenceSufficiencyRequest.model_validate({"question": " ", "answer": "x"})
    with pytest.raises(ValueError):
        EvidenceSufficiencyRequest.model_validate({"question": "x", "answer": ""})
