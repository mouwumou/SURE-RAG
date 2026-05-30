from pathlib import Path

import tomllib


def test_readme_showcase_positioning_and_key_results() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "answer-conditioned evidence sufficiency verifier" in readme
    assert "not a general-purpose hallucination detector" in readme
    assert "The toy backend is not the trained SURE-RAG reference model" in readme
    assert "0.9075 Macro-F1" in readme
    assert "0.8951 +/- 0.0069 Macro-F1" in readme
    assert "0.6516 Macro-F1" in readme
    assert "0.1642" in readme
    assert "0.2588" in readme


def test_required_showcase_docs_exist_and_mark_paper_results_boundary() -> None:
    required_docs = [
        Path("docs/method.md"),
        Path("docs/results.md"),
        Path("docs/limitations.md"),
        Path("docs/roadmap.md"),
    ]
    for path in required_docs:
        assert path.exists()

    results = Path("docs/results.md").read_text(encoding="utf-8")
    assert "paper results, not outputs produced by the toy backend" in results
    assert "0.9075" in results
    assert "0.8888 +/- 0.0109" in results
    assert "0.7389" in results
    assert "0.3343" in results


def test_no_forbidden_showcase_dependencies_added() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = list(pyproject["project"].get("dependencies", []))
    optional_values: list[str] = []
    for values in pyproject["project"].get("optional-dependencies", {}).values():
        optional_values.extend(values)
    declared = "\n".join(dependencies + optional_values).lower()

    for forbidden in [
        "scikit-learn",
        "sklearn",
        "joblib",
        "torch",
        "transformers",
        "fastapi",
        "langchain",
        "llama-index",
        "haystack",
        "guardrails",
        "nemo",
    ]:
        assert forbidden not in declared
