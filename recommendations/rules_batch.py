"""
Axion AI — Batch reactor rule pack
====================================

Starter rule pack for the BATCH_PROFILE. Three rules covering the main
failure modes a process engineer would recognize on an exothermic batch
reactor with cooling jacket:

    B01_HighReactorTemp   reactor temperature exceeds upper spec
    B02_RunawayRisk       heat-release rate climbs while T_R is rising
    B03_LowConversion     batch reaches end without hitting conversion target

Rules use the same `DiagnosticRule` API as the pilot pack so the existing
`RecommendationEngine`, consensus controller, and outcome tracker work
unchanged. Tag references are namespaced under `batch.*` to avoid any
collision with the pilot pack.
"""

from __future__ import annotations

from typing import Optional

from analytics import EventSession, Severity
from .models import (
    Action, ActionType, ExpectedImpact, Recommendation, Urgency,
)
from .rules_base import DiagnosticRule, RuleContext


def _urgency_from_severity(sev: Severity) -> Urgency:
    mapping = {
        Severity.CRITICAL: Urgency.CRITICAL,
        Severity.HIGH:     Urgency.HIGH,
        Severity.MEDIUM:   Urgency.MEDIUM,
        Severity.LOW:      Urgency.LOW,
        Severity.INFO:     Urgency.LOW,
    }
    return mapping.get(sev, Urgency.MEDIUM)


def _priority_score(urgency: Urgency, confidence: float) -> float:
    base = {
        Urgency.CRITICAL: 90, Urgency.HIGH: 70,
        Urgency.MEDIUM: 50,   Urgency.LOW: 25,
    }[urgency]
    return base + 10 * confidence


# =============================================================================
# B01 — Reactor temperature exceeds spec
# =============================================================================

class B01_HighReactorTemp(DiagnosticRule):
    """Fires when batch.T_R_C trips its upper spec (default 95 °C)."""

    @property
    def rule_name(self) -> str: return "B01_HighReactorTemp"

    @property
    def description(self) -> str:
        return "Reactor temperature exceeds upper spec — open coolant valve."

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        return (
            session.tag == "batch.T_R_C"
            and session.peak_severity in (Severity.HIGH, Severity.CRITICAL)
        )

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        t_r = context.current_value("batch.T_R_C") or session.peak_alert.value
        f_c = context.current_value("batch.F_cool")
        if t_r is None or f_c is None:
            return None

        f_c_new = min(f_c * 1.30, 1.0)
        delta   = f_c_new - f_c
        urgency = _urgency_from_severity(session.peak_severity)
        confidence = 0.85

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(urgency, confidence),
            diagnosis=(
                f"Reactor temperature at {t_r:.1f} °C — above the safe "
                f"operating limit. Heat removal is currently insufficient."
            ),
            probable_cause=(
                "Cooling capacity not keeping up with the exothermic "
                "reaction. Coolant flow may be too low, or jacket UA "
                "may be degraded."
            ),
            action=Action(
                type=ActionType.ADJUST_SETPOINT,
                description=(
                    f"Increase coolant flow by ~30% to bring T_R back "
                    f"under the 95 °C limit."
                ),
                target_variable="batch.F_cool",
                current_value=f_c,
                proposed_value=f_c_new,
                adjustment=delta,
                units="m³/h",
            ),
            expected_impact=[
                ExpectedImpact(
                    variable="batch.T_R_C",
                    current_value=t_r,
                    predicted_value=max(85.0, t_r - 4.0),
                    time_to_effect_minutes=10,
                    description="Reactor temperature should fall below spec",
                ),
                ExpectedImpact(
                    variable="batch.F_cool",
                    current_value=f_c,
                    predicted_value=f_c_new,
                    time_to_effect_minutes=1,
                    description="Coolant flow ramped up",
                ),
            ],
            confidence=confidence,
            urgency=urgency,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=["batch.T_R_C", "batch.F_cool"],
        )


# =============================================================================
# B02 — Runaway risk (heat release climbing while T_R rising)
# =============================================================================

class B02_RunawayRisk(DiagnosticRule):
    """Fires when both batch.dHdt and batch.T_R_C are alerting at the same
    time — classical thermal runaway signature."""

    @property
    def rule_name(self) -> str: return "B02_RunawayRisk"

    @property
    def description(self) -> str:
        return (
            "Heat release rate and reactor temperature alerting together — "
            "runaway risk. Recommend emergency cooling or quench."
        )

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        if session.tag != "batch.dHdt":
            return False
        if session.peak_severity not in (Severity.HIGH, Severity.CRITICAL):
            return False
        # Co-occurring T_R session is what makes this "runaway risk"
        return context.has_sessions_from(
            ["SPC.Shewhart", "Trend"], tags=["batch.T_R_C"],
        )

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        dh   = session.peak_alert.value
        f_c  = context.current_value("batch.F_cool")
        t_r  = context.current_value("batch.T_R_C")
        if dh is None or f_c is None or t_r is None:
            return None

        urgency    = Urgency.CRITICAL
        confidence = 0.90
        f_c_new    = 1.0   # max coolant immediately

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(urgency, confidence),
            diagnosis=(
                f"Heat release rate at {dh:.0f} kW with reactor at "
                f"{t_r:.1f} °C and rising — thermal runaway pattern."
            ),
            probable_cause=(
                "Reaction kinetics accelerating (Arrhenius-driven). "
                "If not arrested, T_R will exceed the safety limit "
                "within minutes."
            ),
            action=Action(
                type=ActionType.ADJUST_SETPOINT,
                description=(
                    "Open coolant valve to maximum flow IMMEDIATELY. "
                    "If temperature continues to rise, initiate "
                    "emergency quench protocol."
                ),
                target_variable="batch.F_cool",
                current_value=f_c,
                proposed_value=f_c_new,
                adjustment=f_c_new - f_c,
                units="m³/h",
            ),
            expected_impact=[
                ExpectedImpact(
                    variable="batch.T_R_C",
                    current_value=t_r,
                    predicted_value=max(80.0, t_r - 8.0),
                    time_to_effect_minutes=8,
                    description="T_R should fall sharply",
                ),
            ],
            confidence=confidence,
            urgency=urgency,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=["batch.T_R_C", "batch.dHdt", "batch.F_cool"],
            extra={"requires_safety_review": True},
        )


# =============================================================================
# B03 — Low conversion (informational, end-of-batch)
# =============================================================================

class B03_LowConversion(DiagnosticRule):
    """Fires when conversion plateaus below the 0.85 target after the
    expected reaction window."""

    @property
    def rule_name(self) -> str: return "B03_LowConversion"

    @property
    def description(self) -> str:
        return "Batch conversion below target — investigate kinetics or feed."

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        if session.tag != "batch.conversion":
            return False
        return session.peak_severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        conv = context.current_value("batch.conversion") or session.peak_alert.value
        if conv is None:
            return None

        urgency    = Urgency.MEDIUM
        confidence = 0.70

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(urgency, confidence),
            diagnosis=(
                f"Batch conversion at {conv:.2f} — below the 0.85 target "
                f"required for product release."
            ),
            probable_cause=(
                "Possible causes: (a) catalyst activity below spec, "
                "(b) feed composition off, (c) reactor temperature too "
                "low to drive the kinetics, or (d) reaction time "
                "insufficient for this batch's characteristics."
            ),
            action=Action(
                type=ActionType.INVESTIGATE,
                description=(
                    "Take a sample for offline analysis. If kinetics are "
                    "the issue, consider extending the batch by 30 min or "
                    "increasing setpoint by 2-3 °C on the next batch."
                ),
                target_variable=None,
                current_value=conv,
                proposed_value=None,
                adjustment=None,
                units="",
            ),
            expected_impact=[],   # investigative — no numeric prediction
            confidence=confidence,
            urgency=urgency,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=["batch.conversion"],
        )


# =============================================================================
# Pack
# =============================================================================

BATCH_RULES = [
    B01_HighReactorTemp(),
    B02_RunawayRisk(),
    B03_LowConversion(),
]
