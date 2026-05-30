from surerag import SureRAGVerifier
from surerag.models.toy import TOY_WARNING


def test_toy_verifier_returns_valid_result() -> None:
    result = SureRAGVerifier.toy().verify(
        {
            "question": "Who wrote The Hobbit?",
            "answer": "J.R.R. Tolkien",
            "evidence": [{"id": "e1", "text": "The Hobbit was written by J.R.R. Tolkien."}],
        }
    )
    assert result.label in {"supported", "refuted", "insufficient"}
    assert result.action in {"answer", "abstain", "retrieve_more", "regenerate", "human_review"}
    assert abs(sum(result.probs.as_ordered_list()) - 1.0) < 1e-9
    assert TOY_WARNING in result.warnings
    assert result.model.capabilities.three_way_label is True
    assert result.safe_to_answer == (result.label == "supported" and result.action == "answer")


def test_empty_evidence_returns_insufficient_without_exception() -> None:
    result = SureRAGVerifier.toy().verify(
        {"question": "Who wrote The Hobbit?", "answer": "J.R.R. Tolkien", "evidence": []}
    )
    assert result.label == "insufficient"
    assert result.action in {"retrieve_more", "abstain"}
    assert result.safe_to_answer is False
    assert result.probs.insufficient == 1.0
    assert "NO_EVIDENCE" in result.reason_codes
