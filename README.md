# SURE-RAG

**SURE-RAG verifies whether retrieved evidence is sufficient to justify a RAG answer.**

RAG systems retrieve relevant passages, but relevance is not verification: a passage can mention the right entities while still missing a required fact, omitting a reasoning hop, or contradicting the generated answer.

SURE-RAG frames this gap as **answer-conditioned evidence sufficiency verification**. Given a question, a candidate answer, and retrieved evidence, it predicts:

- `supported`: the evidence justifies the candidate answer;
- `refuted`: the evidence contradicts the candidate answer;
- `insufficient`: the evidence is relevant, partial, or topical but does not justify the candidate answer.

The result is not just a score. It is a routing decision: `answer`, `abstain`, `retrieve_more`, `regenerate`, or `human_review`.

## Where SURE-RAG Fits

SURE-RAG is not a retriever, generator, reranker, or general-purpose hallucination detector. It is a **verification and routing layer** for RAG systems.

The core abstraction is:

```text
Verify(question, candidate_answer, retrieved_evidence)
    -> supported / refuted / insufficient
    -> answer / abstain / retrieve_more / regenerate / human_review
```

In a standard retrieve-then-generate RAG pipeline, SURE-RAG is commonly used after a candidate answer is produced and before the answer is shown to the user:

```text
user query
  -> retriever
  -> retrieved evidence
  -> generator produces candidate answer
  -> SURE-RAG verifies evidence sufficiency
  -> answer / abstain / retrieve_more / regenerate
```

More generally, SURE-RAG verifies **candidate answer content**, not necessarily only a final answer. In agentic or iterative RAG systems, the same verifier can check an answer draft, an intermediate claim set, or multiple candidate answers before final response generation:

```text
retrieve evidence
  -> propose claims or draft answer
  -> SURE-RAG verifies candidate content
  -> supported? finalize
  -> insufficient? retrieve more
  -> refuted? revise or regenerate
```

This is why the project is framed as an **answer-conditioned evidence sufficiency verifier**, not merely as a post-generation output filter.

## Why SURE-RAG?

A RAG answer can fail in several different ways:

- The evidence may fully support the answer.
- The evidence may contradict the answer.
- The evidence may be relevant but incomplete.
- The evidence may contain only one hop of a multi-hop chain.
- The evidence may be internally conflicted.

Collapsing these cases into `grounded` / `not grounded` loses operational information. `refuted` and `insufficient` should trigger different system behavior: a contradicted answer should usually be regenerated or corrected, while an insufficient answer should usually trigger more retrieval or abstention.

## One-Minute Example

Question:

> Was the author of *The Hobbit* born earlier than the author of *A Game of Thrones*?

Candidate answer:

> Yes.

Full evidence supports the answer:

```text
The Hobbit -> J.R.R. Tolkien
Tolkien -> born in 1892
A Game of Thrones -> George R. R. Martin
Martin -> born in 1948
```

Partial evidence is `insufficient` if one required birth year or bridge fact is missing.

A contradictory candidate answer such as `No` is `refuted` by the same full evidence.

## Method at a Glance

```text
question + candidate answer + retrieved evidence
        -> candidate claims
        -> claim/evidence relation scores
           support / refute / neutral
        -> set-level SURE aggregation
           coverage + relation strength + disagreement + conflict + retrieval uncertainty
        -> supported / refuted / insufficient
        -> answer routing
```

SURE-RAG treats evidence sufficiency as a **set-level property**. A single supportive-looking passage can be misleading if another required hop is missing; a strong refutation can be diluted by neutral passages under mean pooling. SURE-RAG aggregates local relation signals into answer-level sufficiency features and a selective routing decision.

The protocol separates model judgment from runtime action. Only `supported + answer` may be marked safe to answer.

## Current Repository Status

This repository is currently a **showcase + protocol package**, not yet a full trained model release.

Included:

- protocol schemas and JSON Schema export;
- runtime routing policy with safety invariants;
- a deterministic CPU-only toy backend for smoke tests;
- CLI verification and toy artifact export;
- a runnable precomputed-pair-score method demo;
- paper result summaries and limitations.

Not included yet:

- trained pair-scorer weights;
- trained SURE aggregation backend;
- benchmark data or paper output files;
- full training pipeline;
- framework adapters or HTTP service.

The toy backend is not the trained SURE-RAG reference model. The precomputed-score demo is not a model evaluation. Trained inference artifacts and model cards are planned as a later release.

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

Expected demo behavior:

```text
demo_supported_full          supported      answer
demo_insufficient_partial    insufficient   retrieve_more
demo_refuted_answer          refuted        regenerate
```

The demo uses synthetic, precomputed pair-level support/refute/neutral scores to show the SURE aggregation idea without shipping a neural verifier, sklearn aggregator, training pipeline, or model weights. It prints key aggregation signals such as `support_max`, `refute_max`, `neutral_mean`, `claim_coverage_deficit`, `conflict_score`, `evidence_disagreement_mean`, and `retrieval_uncertainty`.

## Key Results

These are paper results, not outputs from the toy backend or the precomputed-score demo in this repository.

| Setting | Model | Result |
|---|---|:---|
| HotpotQA-RAG v3 | SURE-RAG calibrated | Macro-F1 0.9075 ± 0.0060 |
| HotpotQA-RAG v3 | SURE-RAG raw | Macro-F1 0.8951 ± 0.0069 |
| HotpotQA-RAG v3 | Concat cross-encoder | Macro-F1 0.8888 ± 0.0109 |
| HotpotQA-RAG v3 | DeBERTa mean-pool | Macro-F1 0.6516 ± 0.0595 |
| HotpotQA-RAG v3 | GPT-4o | Macro-F1 0.7284 |
| Selective answering | SURE-RAG Risk@30 | 0.1642 |
| Selective answering | Mean-pool Risk@30 | 0.2588 |

These results isolate the value of answer-level sufficiency aggregation: SURE-RAG and pooling baselines share the same pair-level verifier, but SURE-RAG aggregates local support/refute/neutral signals into set-level coverage, disagreement, conflict, and uncertainty features.

Boundary result: SURE-RAG is weaker than GPT-4o on HaluBench unsafe detection, which supports the claim that controlled evidence sufficiency verification and natural hallucination detection are different tasks.

## What SURE-RAG Is Not

SURE-RAG is not:

- a replacement for retrieval or generation;
- a general hallucination detector;
- a toxicity, policy, or moderation guardrail;
- a validated long-form factuality checker;
- a production safety system without domain-specific validation.

The research reference model is primarily validated on controlled short-answer multi-hop sufficiency data. The protocol supports multi-claim inputs, but full long-form claim coverage remains future work.

## Protocol and Artifacts

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
- model card and dataset card;
- clear documentation for protocol usage and paper results.

Later:

- training pipeline;
- framework adapters;
- HTTP server;
- full reproducibility bundle.

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
