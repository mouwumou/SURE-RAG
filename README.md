# SURE-RAG

SURE-RAG is an answer-conditioned evidence sufficiency verifier for selective RAG. Given a question, a candidate answer, and retrieved evidence, it predicts supported, refuted, or insufficient, then routes the system to answer, abstain, retrieve more, regenerate, or escalate.

## Why SURE-RAG?

Relevance is not sufficiency. A passage can be topical but still fail to justify the answer. Evidence can be partial, missing a hop, or contradictory.

In a standard RAG pipeline, SURE-RAG is commonly used after a candidate answer is generated and before user-facing output. More generally, it verifies candidate answer content: a final answer, an answer draft, or an intermediate claim set.

## What SURE-RAG Verifies

Input:
- question
- candidate answer
- retrieved evidence

Output:
- `supported`
- `refuted`
- `insufficient`
- action: `answer` / `abstain` / `retrieve_more` / `regenerate` / `human_review`

## Method At A Glance

```text
query
  -> retrieve evidence
  -> propose candidate answer / claims
  -> pair-level support-refute-neutral scoring
  -> set-level SURE aggregation
  -> supported/refuted/insufficient + selective route
```

The protocol separates the model judgement from the runtime action. Only `supported + answer` may be marked safe to answer.

## Key Results

These are paper results, not outputs from the toy backend or the precomputed-score demo in this repository.

| Setting | Model | Result |
|---|---|---|
| HotpotQA-RAG v3 | SURE-RAG calibrated | 0.9075 Macro-F1 |
| HotpotQA-RAG v3 | SURE-RAG raw | 0.8951 +/- 0.0069 Macro-F1 |
| HotpotQA-RAG v3 | DeBERTa mean-pool | 0.6516 Macro-F1 |
| Selective answering | SURE-RAG Risk@30 | 0.1642 |
| Selective answering | Mean-pool Risk@30 | 0.2588 |

Boundary result: SURE-RAG is weaker than GPT-4o on HaluBench unsafe detection, which supports the claim that controlled evidence sufficiency verification and natural hallucination detection are different tasks.

## Current Repository Status

This repository currently includes:
- a lightweight protocol skeleton;
- protocol schemas and JSON Schema export;
- a deterministic toy backend for smoke tests;
- routing policy and safety invariants;
- CLI verification and toy artifact export;
- a runnable precomputed-pair-score method demo.

The toy backend is not the trained SURE-RAG reference model. The precomputed-score demo is not a model evaluation. The trained pair-scorer and SURE aggregation reference backend, including inference artifacts, are planned for a later release task.

## Quickstart

```bash
uv sync --all-extras
```

or:

```bash
pip install -e ".[dev]"
```

Run the toy verifier:

```bash
surerag verify \
  --input examples/minimal_request.jsonl \
  --output /tmp/surerag_output.jsonl \
  --backend toy
```

Run the method demo:

```bash
python examples/run_sure_rag_demo.py
```

## Runnable Demo

The demo uses synthetic, precomputed pair-level support/refute/neutral scores to show the SURE aggregation idea without shipping a neural verifier, sklearn aggregator, training pipeline, or model weights.

It contains three illustrative conditions:
- supported/full evidence;
- insufficient/partial evidence;
- refuted/contradictory candidate answer.

## Protocol And Artifacts

The stable protocol version is `evidence-sufficiency.v1`. Requests contain a `question`, candidate `answer`, optional `claims`, and retrieved `evidence`. Results contain the three-way label, routing action, probabilities, selective score, audit trace, and model capabilities.

Toy verifier artifacts include:

```text
verifier_manifest.json
protocol_schema.json
policy/routing_policy.json
```

Export a toy artifact:

```bash
surerag export --backend toy --output /tmp/surerag_toy_artifact
```

## Roadmap

Near-term showcase/release:
- paper landing page and method demo;
- trained inference artifacts;
- model card and dataset card.

Later:
- training pipeline;
- framework adapters;
- HTTP server;
- full reproducibility bundle.

## Limitations

SURE-RAG is an evidence sufficiency verifier for RAG answer routing, not a general-purpose hallucination detector. The research reference model is primarily validated on controlled short-answer multi-hop sufficiency data. The protocol supports multi-claim inputs, but full long-form claim coverage remains future work.

The current toy backend and precomputed-score demo are illustrative. They are not the trained SURE-RAG research backend and should not be used as factuality benchmarks.

## Citation

Final publication metadata is not included yet. Until a final citation is available, cite the project as:

```bibtex
@misc{sure-rag,
  title = {SURE-RAG: Evidence Sufficiency Verification for Selective Retrieval-Augmented Generation},
  author = {SURE-RAG Contributors},
  year = {2026},
  note = {Project repository}
}
```
