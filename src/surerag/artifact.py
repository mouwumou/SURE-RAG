"""Artifact save/load helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from surerag.core.claim_builder import DefaultClaimBuilder
from surerag.core.policy import RoutingPolicy
from surerag.models.toy import ToyLexicalAggregator, ToyLexicalPairScorer
from surerag.protocol.errors import ArtifactError
from surerag.protocol.json_schema import export_protocol_schema
from surerag.protocol.labels import ANSWER_LABELS, PAIR_TRAIN_LABELS, PROTOCOL_VERSION
from surerag.protocol.schema import ModelCapabilities, ModelInfo


def save_artifact(verifier: Any, path: str | Path) -> None:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "policy").mkdir(exist_ok=True)

    policy = getattr(verifier, "policy", RoutingPolicy.default())
    model_info = getattr(verifier, "model_info", None)
    if model_info is None:
        raise ArtifactError("verifier is missing model_info")

    manifest = {
        "artifact_type": "surerag_verifier",
        "protocol_version": PROTOCOL_VERSION,
        "implementation": model_info.implementation,
        "created_at": datetime.now(UTC).isoformat(),
        "labels": {
            "answer": ANSWER_LABELS,
            "pair": PAIR_TRAIN_LABELS,
        },
        "components": {
            "claim_builder": "default_claim_builder",
            "pair_scorer": model_info.pair_scorer,
            "aggregator": model_info.aggregator,
            "calibrator": model_info.calibration,
            "policy": "selective_threshold",
        },
        "capabilities": model_info.capabilities.model_dump(),
        "training_data": {
            "dataset": "none",
            "split_policy": "none",
            "notes": (
                "Controlled evidence sufficiency verifier; not a generic hallucination detector."
            ),
        },
    }
    _write_json(output / "verifier_manifest.json", manifest)
    export_protocol_schema(output / "protocol_schema.json")
    _write_json(output / "policy" / "routing_policy.json", policy.to_json_dict())


def load_artifact(path: str | Path) -> Any:
    from surerag.core.verifier import SureRAGVerifier

    root = Path(path)
    manifest_path = root / "verifier_manifest.json"
    if not manifest_path.exists():
        raise ArtifactError(f"missing verifier manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != "surerag_verifier":
        raise ArtifactError("artifact_type must be 'surerag_verifier'")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ArtifactError(f"unsupported protocol_version: {manifest.get('protocol_version')}")

    components = manifest.get("components", {})
    pair_scorer = components.get("pair_scorer")
    aggregator = components.get("aggregator")
    if pair_scorer != "toy_lexical" or aggregator != "toy_heuristic":
        raise ArtifactError(
            "initial skeleton can only load toy_lexical/toy_heuristic artifacts"
        )

    policy_path = root / "policy" / "routing_policy.json"
    policy = (
        RoutingPolicy.model_validate(_read_json(policy_path))
        if policy_path.exists()
        else RoutingPolicy.default()
    )
    model_info = ModelInfo(
        implementation=manifest["implementation"],
        model_id="toy-lexical",
        pair_scorer=pair_scorer,
        aggregator=aggregator,
        calibration=components.get("calibrator"),
        capabilities=ModelCapabilities.model_validate(manifest.get("capabilities", {})),
    )
    return SureRAGVerifier(
        claim_builder=DefaultClaimBuilder(),
        pair_scorer=ToyLexicalPairScorer(),
        aggregator=ToyLexicalAggregator(),
        policy=policy,
        model_info=model_info,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
