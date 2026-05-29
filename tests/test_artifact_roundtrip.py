from surerag import SureRAGVerifier
from surerag.artifact import save_artifact


def test_toy_artifact_roundtrip(tmp_path) -> None:
    artifact_path = tmp_path / "toy_artifact"
    verifier = SureRAGVerifier.toy()
    save_artifact(verifier, artifact_path)

    assert (artifact_path / "verifier_manifest.json").exists()
    assert (artifact_path / "protocol_schema.json").exists()
    assert (artifact_path / "policy" / "routing_policy.json").exists()

    loaded = SureRAGVerifier.from_artifact(artifact_path)
    result = loaded.verify(
        {
            "question": "Who wrote The Hobbit?",
            "answer": "J.R.R. Tolkien",
            "evidence": [{"id": "e1", "text": "The Hobbit was written by J.R.R. Tolkien."}],
        }
    )
    assert result.protocol_version == "evidence-sufficiency.v1"
    assert abs(sum(result.probs.as_ordered_list()) - 1.0) < 1e-9
