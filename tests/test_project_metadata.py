from pathlib import Path

import tomllib


def test_license_metadata_matches_license_file() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    license_text = Path("LICENSE").read_text(encoding="utf-8")

    assert pyproject["project"]["license"]["text"] == "Apache-2.0"
    assert "License :: OSI Approved :: Apache Software License" in pyproject["project"][
        "classifiers"
    ]
    assert "Apache License" in license_text


def test_gitignore_keeps_private_outputs_and_agent_traces_out() -> None:
    lines = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    active_patterns = {
        line.strip() for line in lines if line.strip() and not line.strip().startswith("#")
    }

    assert "data/" in active_patterns
    assert "outputs/" in active_patterns
    assert "reports/" in active_patterns
    assert "artifacts/" in active_patterns
    assert "checkpoints/" in active_patterns
    assert "/models/" in active_patterns or "models/" in active_patterns
    assert "legacy/" in active_patterns
    assert "reference/" in active_patterns
    assert ".codex/" in active_patterns
    assert "agent_logs/" in active_patterns
    assert "agent_runs/" in active_patterns
    assert "codex_runs/" in active_patterns
    assert "*.joblib" in active_patterns
    assert "*.pkl" in active_patterns
    assert "*.pt" in active_patterns
    assert "*.pth" in active_patterns
    assert "*.bin" in active_patterns
    assert "*.safetensors" in active_patterns
    if Path("codex").exists():
        assert "codex/" not in active_patterns
    else:
        assert "codex/" in active_patterns
        assert "AGENTS.md" in active_patterns
    assert "uv.lock" not in active_patterns


def test_readme_states_toy_backend_is_not_trained_reference() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "protocol skeleton" in readme
    assert "The toy backend is not the trained SURE-RAG reference model" in readme
    assert "trained pair-scorer and SURE aggregation reference backend" in readme
    assert "not a general-purpose hallucination detector" in readme
