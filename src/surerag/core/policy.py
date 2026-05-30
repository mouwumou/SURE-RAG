"""Runtime routing policy for verifier results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Action = Literal["answer", "abstain", "retrieve_more", "regenerate", "human_review"]


class RoutingDecision(BaseModel):
    action: Action
    reason_codes: list[str] = Field(default_factory=list)


class RoutingPolicy(BaseModel):
    answer_label: Literal["supported"] = "supported"
    selective_score_gte: float = 0.55
    selective_beta: float = Field(default=0.1, ge=0.0)
    confidence_gte: float = 0.70
    on_supported_below_threshold: Action = "abstain"
    on_refuted: Action = "regenerate"
    on_insufficient: Action = "retrieve_more"
    on_no_evidence: Action = "retrieve_more"
    on_low_confidence: Action = "abstain"
    on_high_conflict: Action = "human_review"
    high_conflict_gte: float | None = None

    @model_validator(mode="after")
    def _reject_unsafe_answer_routes(self) -> RoutingPolicy:
        unsafe_routes = {
            "on_supported_below_threshold": self.on_supported_below_threshold,
            "on_refuted": self.on_refuted,
            "on_insufficient": self.on_insufficient,
            "on_no_evidence": self.on_no_evidence,
            "on_low_confidence": self.on_low_confidence,
            "on_high_conflict": self.on_high_conflict,
        }
        answer_routes = [name for name, action in unsafe_routes.items() if action == "answer"]
        if answer_routes:
            joined = ", ".join(sorted(answer_routes))
            raise ValueError(f"unsafe routing policy cannot set answer for: {joined}")
        return self

    @model_validator(mode="before")
    @classmethod
    def _support_brief_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        answer_if = normalized.pop("answer_if", None)
        if isinstance(answer_if, dict):
            if answer_if.get("label") == "supported":
                normalized["answer_label"] = "supported"
            if "selective_score_gte" in answer_if:
                normalized["selective_score_gte"] = answer_if["selective_score_gte"]
            if "selective_beta" in answer_if:
                normalized["selective_beta"] = answer_if["selective_beta"]
            if "confidence_gte" in answer_if:
                normalized["confidence_gte"] = answer_if["confidence_gte"]
        return normalized

    @classmethod
    def default(cls) -> RoutingPolicy:
        return cls()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "answer_if": {
                "label": self.answer_label,
                "selective_score_gte": self.selective_score_gte,
                "selective_beta": self.selective_beta,
                "confidence_gte": self.confidence_gte,
            },
            "on_supported_below_threshold": self.on_supported_below_threshold,
            "on_refuted": self.on_refuted,
            "on_insufficient": self.on_insufficient,
            "on_no_evidence": self.on_no_evidence,
            "on_low_confidence": self.on_low_confidence,
            "on_high_conflict": self.on_high_conflict,
            "high_conflict_gte": self.high_conflict_gte,
        }

    def route(
        self,
        *,
        label: str,
        confidence: float,
        selective_score: float | None,
        features: dict[str, float] | None = None,
        no_evidence: bool = False,
    ) -> RoutingDecision:
        if no_evidence:
            return RoutingDecision(action=self.on_no_evidence, reason_codes=["NO_EVIDENCE"])

        active_features = features or {}
        if (
            self.high_conflict_gte is not None
            and active_features.get("conflict_score", 0.0) >= self.high_conflict_gte
        ):
            return RoutingDecision(action=self.on_high_conflict, reason_codes=["HIGH_CONFLICT"])

        if label == "supported":
            if confidence < self.confidence_gte:
                return RoutingDecision(
                    action=self.on_low_confidence,
                    reason_codes=["SUPPORTED_LOW_CONFIDENCE"],
                )
            if selective_score is None or selective_score < self.selective_score_gte:
                return RoutingDecision(
                    action=self.on_supported_below_threshold,
                    reason_codes=["SUPPORTED_BELOW_SELECTIVE_THRESHOLD"],
                )
            return RoutingDecision(action="answer", reason_codes=["SUPPORTED_HIGH_CONFIDENCE"])

        if label == "refuted":
            return RoutingDecision(action=self.on_refuted, reason_codes=["REFUTED"])

        if label == "insufficient":
            return RoutingDecision(action=self.on_insufficient, reason_codes=["INSUFFICIENT"])

        raise ValueError(f"invalid answer label: {label}")
