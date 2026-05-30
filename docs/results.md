# Results

These are paper results, not outputs produced by the toy backend or the precomputed-score demo in this repository.

## HotpotQA-RAG v3

| Model | Macro-F1 |
|---|---:|
| SURE-RAG calibrated | 0.9075 |
| SURE-RAG raw | 0.8951 +/- 0.0069 |
| Concat cross-encoder | 0.8888 +/- 0.0109 |
| DeBERTa mean-pool | 0.6516 |

## Selective Answering

| Model | Risk@30 |
|---|---:|
| SURE-RAG | 0.1642 |
| DeBERTa mean-pool | 0.2588 |

## Boundary Result

On HaluBench unsafe-F1, GPT-4o reaches 0.7389 while SURE-RAG reaches 0.3343. This transfer limitation supports the distinction between controlled evidence sufficiency verification and broader natural hallucination detection.

## Artifact Note

The current repository ships a protocol package, toy backend, and precomputed-score method demo. It does not yet ship trained inference artifacts, benchmark data, or paper reproduction outputs.
