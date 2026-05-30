# Limitations

SURE-RAG is an evidence sufficiency verifier for RAG answer routing, not a general-purpose hallucination detector.

Current limitations:

- The research reference model is primarily validated on controlled short-answer multi-hop sufficiency data.
- There is moderate shortcut risk in controlled benchmark construction.
- Long-form claim coverage is supported by the protocol but not fully validated.
- HaluBench transfer is limited: GPT-4o reaches 0.7389 unsafe-F1 while SURE-RAG reaches 0.3343.
- The toy backend is not the research model.
- The precomputed-score demo is illustrative and does not produce paper metrics.

The repository intentionally does not include real model weights, private benchmark data, paper outputs, or training artifacts.
