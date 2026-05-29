from surerag.core.policy import RoutingPolicy


def test_supported_above_threshold_answers() -> None:
    decision = RoutingPolicy.default().route(
        label="supported", confidence=0.9, selective_score=0.8, features={}
    )
    assert decision.action == "answer"


def test_supported_below_threshold_abstains() -> None:
    decision = RoutingPolicy.default().route(
        label="supported", confidence=0.9, selective_score=0.1, features={}
    )
    assert decision.action == "abstain"


def test_refuted_and_insufficient_default_actions() -> None:
    policy = RoutingPolicy.default()
    assert policy.route(label="refuted", confidence=0.9, selective_score=0.0).action == "regenerate"
    assert (
        policy.route(label="insufficient", confidence=0.9, selective_score=0.0).action
        == "retrieve_more"
    )


def test_no_evidence_routes_retrieve_more() -> None:
    decision = RoutingPolicy.default().route(
        label="insufficient", confidence=1.0, selective_score=0.0, no_evidence=True
    )
    assert decision.action == "retrieve_more"
    assert decision.reason_codes == ["NO_EVIDENCE"]


def test_high_conflict_can_route_to_human_review() -> None:
    policy = RoutingPolicy(high_conflict_gte=0.5)
    decision = policy.route(
        label="supported", confidence=0.9, selective_score=0.9, features={"conflict_score": 0.6}
    )
    assert decision.action == "human_review"
