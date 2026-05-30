import subprocess
import sys


def test_precomputed_pair_score_demo_runs_and_shows_three_labels() -> None:
    completed = subprocess.run(
        [sys.executable, "examples/run_sure_rag_demo.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    output = completed.stdout
    assert "label=supported" not in output
    assert "demo_supported_full" in output
    assert "demo_insufficient_partial" in output
    assert "demo_refuted_answer" in output
    assert "supported" in output
    assert "insufficient" in output
    assert "refuted" in output
    assert "synthetic/precomputed demonstration scores, not model predictions" in output


def test_precomputed_pair_score_demo_prints_key_features() -> None:
    completed = subprocess.run(
        [sys.executable, "examples/run_sure_rag_demo.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    output = completed.stdout
    for feature in [
        "support_max",
        "refute_max",
        "neutral_mean",
        "claim_coverage_deficit",
        "conflict_score",
        "evidence_disagreement_mean",
        "retrieval_uncertainty",
    ]:
        assert feature in output
