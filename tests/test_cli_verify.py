import json
import subprocess
import sys


def test_cli_verify_toy(tmp_path) -> None:
    output = tmp_path / "verified.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "surerag.cli",
            "verify",
            "--input",
            "examples/minimal_request.jsonl",
            "--output",
            str(output),
            "--backend",
            "toy",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {row["protocol_version"] for row in rows} == {"evidence-sufficiency.v1"}
    assert rows[1]["label"] == "insufficient"
