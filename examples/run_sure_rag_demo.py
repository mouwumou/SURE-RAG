"""Run the SURE-RAG precomputed pair-score method demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from surerag.method import verify_with_precomputed_pair_scores

FEATURES = [
    "support_max",
    "refute_max",
    "neutral_mean",
    "claim_coverage_deficit",
    "conflict_score",
    "evidence_disagreement_mean",
    "retrieval_uncertainty",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    requests = json.loads((root / "sure_rag_demo_request.json").read_text(encoding="utf-8"))
    pair_score_payload = json.loads(
        (root / "sure_rag_demo_pair_scores.json").read_text(encoding="utf-8")
    )
    pair_scores = pair_score_payload["scores"]

    print("SURE-RAG precomputed-pair-score method demo")
    print("Pair scores are synthetic/precomputed demonstration scores, not model predictions.")
    print()
    print(f"{'example':30} {'label':14} {'action':14} {'safe'}")
    print("-" * 72)

    results = []
    for example in requests["examples"]:
        result = verify_with_precomputed_pair_scores(example, pair_scores[example["id"]])
        results.append(result)
        print(f"{example['id']:30} {result.label:14} {result.action:14} {result.safe_to_answer}")
        for feature in FEATURES:
            value = result.audit.features.get(feature, 0.0)
            print(f"  {feature}: {value:.4f}")
        print()

    print("JSON summary")
    print(
        json.dumps(
            [
                {
                    "id": result.id,
                    "label": result.label,
                    "action": result.action,
                    "safe_to_answer": result.safe_to_answer,
                    "probs": result.probs.model_dump(),
                }
                for result in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
