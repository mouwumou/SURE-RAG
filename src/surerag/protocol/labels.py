"""Canonical protocol labels and helpers."""

PROTOCOL_VERSION = "evidence-sufficiency.v1"

ANSWER_LABELS = ["supported", "refuted", "insufficient"]
PAIR_LABELS = ["support", "refute", "neutral", "unknown"]
PAIR_TRAIN_LABELS = ["support", "refute", "neutral"]
BINARY_LABELS = ["safe", "unsafe"]
ACTIONS = ["answer", "abstain", "retrieve_more", "regenerate", "human_review"]


def binary_from_answer_label(label: str) -> str:
    if label == "supported":
        return "safe"
    if label in {"refuted", "insufficient"}:
        return "unsafe"
    raise ValueError(f"invalid answer label: {label}")


def is_safe_label(label: str) -> bool:
    return label == "supported"
