# SURE-RAG

SURE-RAG defines a framework-agnostic protocol and reference implementation for RAG answer verification. Given a question, candidate answer, and retrieved evidence, a verifier predicts whether the evidence supports, refutes, or is insufficient for the answer, then routes the RAG system to answer, abstain, retrieve more evidence, regenerate, or escalate.

## Current Status

This package currently contains the CPU-only protocol skeleton: Pydantic protocol schemas, JSON Schema export, a routing policy, CLI verification, toy artifact save/load, and a deterministic toy backend. The toy backend is not the trained SURE-RAG reference model; it exists only for smoke tests, examples, and integration checks.

The trained pair-scorer and SURE aggregation reference backend are planned for a later extraction task from the private research implementation.

## What It Is Not

SURE-RAG is an evidence sufficiency verifier for RAG answer routing, not a general-purpose hallucination detector. The reference model is primarily validated on controlled short-answer multi-hop sufficiency data. The protocol supports multi-claim inputs, but full long-form claim coverage remains future work.

## Install

```bash
uv sync --all-extras
```

or:

```bash
pip install -e ".[dev]"
```

## Python Quickstart

```python
from surerag import SureRAGVerifier

verifier = SureRAGVerifier.toy()
result = verifier.verify({
    "question": "Who wrote The Hobbit?",
    "answer": "J.R.R. Tolkien",
    "evidence": [{"id": "e1", "text": "The Hobbit was written by J.R.R. Tolkien."}],
})

print(result.label, result.action, result.probs.model_dump())
```

The toy backend is deterministic and CPU-only. It is for smoke tests and examples, not production verification.

## CLI Quickstart

```bash
surerag verify \
  --input examples/minimal_request.jsonl \
  --output /tmp/surerag_output.jsonl \
  --backend toy
```

Export a toy artifact:

```bash
surerag export --backend toy --output /tmp/surerag_toy_artifact
```

## Protocol

The stable protocol version is `evidence-sufficiency.v1`. Requests contain a `question`, candidate `answer`, optional `claims`, and retrieved `evidence`. Results contain the three-way label, routing action, probabilities, selective score, audit trace, and model capabilities.

JSON Schema can be exported from Python:

```python
from surerag.protocol.json_schema import export_protocol_schema

schema = export_protocol_schema()
```

## Artifacts

Toy verifier artifacts include:

```text
verifier_manifest.json
protocol_schema.json
policy/routing_policy.json
```

They can be saved and loaded with:

```python
from surerag import SureRAGVerifier
from surerag.artifact import save_artifact

verifier = SureRAGVerifier.toy()
save_artifact(verifier, "artifacts/surerag_verifier")
loaded = SureRAGVerifier.from_artifact("artifacts/surerag_verifier")
```

The toy artifact format exercises the manifest and policy contract. It does not contain neural pair-scorer weights, a trained SURE aggregator, or calibration artifacts yet.

## Framework Adapters

Framework adapters for LangChain, LlamaIndex, Haystack, Guardrails AI, NeMo, and HTTP serving are planned as optional layers. Core protocol and toy verification do not import or require those dependencies.
