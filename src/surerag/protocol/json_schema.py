"""JSON Schema export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from surerag.protocol.labels import PROTOCOL_VERSION
from surerag.protocol.schema import EvidenceSufficiencyRequest, EvidenceSufficiencyResult


def request_json_schema() -> dict[str, Any]:
    return EvidenceSufficiencyRequest.model_json_schema()


def result_json_schema() -> dict[str, Any]:
    return EvidenceSufficiencyResult.model_json_schema()


def export_protocol_schema(path: str | Path | None = None) -> dict[str, Any]:
    schema = {
        "protocol_version": PROTOCOL_VERSION,
        "request": request_json_schema(),
        "result": result_json_schema(),
    }
    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return schema
