import subprocess
import sys

from surerag import SureRAGVerifier


def test_cli_export_toy_creates_loadable_artifact(tmp_path) -> None:
    output = tmp_path / "toy_artifact"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "surerag.cli",
            "export",
            "--backend",
            "toy",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output / "verifier_manifest.json").exists()
    assert (output / "protocol_schema.json").exists()
    assert (output / "policy" / "routing_policy.json").exists()

    result = SureRAGVerifier.from_artifact(output).verify(
        {
            "question": "Who wrote The Hobbit?",
            "answer": "J.R.R. Tolkien",
            "evidence": [{"id": "e1", "text": "The Hobbit was written by J.R.R. Tolkien."}],
        }
    )
    assert result.protocol_version == "evidence-sufficiency.v1"
