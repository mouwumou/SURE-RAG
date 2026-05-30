# Method

SURE-RAG is an answer-conditioned evidence sufficiency verifier for selective RAG. Given a question, a candidate answer, and retrieved evidence, it predicts whether the evidence supports, refutes, or is insufficient for the answer.

In a standard RAG pipeline, SURE-RAG is commonly used after a candidate answer is generated and before user-facing output. More generally, it verifies candidate answer content: a final answer, an answer draft, or an intermediate claim set.

## Relevance Is Not Sufficiency

Retrieved passages can be relevant without being sufficient. A passage may mention the right entities while missing a required comparison, omitting a bridge fact, or contradicting the candidate answer.

SURE-RAG preserves three protocol labels:

- `supported`: the evidence is sufficient for the candidate answer.
- `refuted`: the evidence contradicts the candidate answer.
- `insufficient`: the evidence is topical or partial but does not justify the candidate answer.

## Pair-Level To Set-Level

The method decomposes candidate answer content into claims, scores claim/evidence pairs as support, refute, or neutral, then aggregates the full evidence set into an answer-level sufficiency decision.

```text
question + candidate answer + evidence
  -> candidate claims
  -> pair-level support/refute/neutral
  -> set-level sufficiency aggregation
  -> supported/refuted/insufficient
  -> answer routing
```

Routing actions are separate from labels:

- `answer`
- `abstain`
- `retrieve_more`
- `regenerate`
- `human_review`

Only `supported + answer` is safe to answer.

## Boundary

SURE-RAG is not a general-purpose hallucination detector. It verifies whether the retrieved evidence is sufficient for candidate answer content. That is different from detecting every possible unsafe or false natural-language output.
