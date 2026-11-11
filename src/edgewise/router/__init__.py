"""Router: local-first answering with escalation, privacy rules and cost tracking."""

from edgewise.router.core import (
    EscalationFailed,
    RouteDecision,
    Router,
    RouteRequest,
    RouterStats,
    Tier,
)
from edgewise.router.evaluate import CascadeEvaluation, CascadePoint, evaluate_cascade
from edgewise.router.rules import Connectivity, detect_pii

__all__ = [
    "CascadeEvaluation",
    "CascadePoint",
    "Connectivity",
    "EscalationFailed",
    "RouteDecision",
    "RouteRequest",
    "Router",
    "RouterStats",
    "Tier",
    "detect_pii",
    "evaluate_cascade",
]
