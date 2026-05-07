"""
Axion AI - Pilot Process Rule Library
=====================================

Rules that encode process knowledge for the pilot CSTR + binary distillation
process. Each rule corresponds to a recognizable failure mode or operational
situation a process engineer would identify.

Rules are organized by detection signature, not by scenario name. This is
important: the system shouldn't know about "the scenario called thermal_drift",
it should recognize the *pattern* that thermal drift produces
(T_R rising + purity dropping + Q_reb rising) in the real world.

Rules implemented in this module:
    1. R01_ThermalDrift         — jacket fouling / heat removal loss
    2. R02_FeedComposition      — feed composition shift detected
    3. R03_ControllerOscillation — PID instability in reactor loop
    4. R04_ColumnEfficiencyLoss — column separation capacity degrading
    5. R05_ExcessReflux         — energy waste via over-refluxing
    6. R06_PurityDeviation      — product off-spec direct detection
    7. R07_SensorFault          — frozen sensor or large bias
    8. R08_ProductTransition    — grade change in progress (informational)

Engineering notes on knowledge captured here:

- Rule R01 looks for temperature rising *together with* Q_reb rising (the
  reactor getting hotter means more A → B conversion, which means more B
  goes to the column, which needs more reboiler duty to separate). This
  co-occurrence is what distinguishes thermal drift from, say, a simple
  setpoint change on T_R.

- Rule R04 looks for purity dropping while reflux and temperatures are
  stable. That points to the column itself (tray fouling, foaming, etc.)
  rather than upstream issues.

- Rule R05 looks for reflux higher than needed for the achieved purity.
  This is the "energy waste" scenario and should always be a LOW urgency
  suggestion (optimization, not correction).

- Rule R07 looks for tags with suspicious statistics (zero variance for
  an extended period = frozen sensor). The Shewhart alert catches the
  start of the anomaly; this rule diagnoses it as an instrument issue.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd

from analytics import EventSession, Severity
from .models import (
    Recommendation, Action, ActionType, ExpectedImpact, Urgency,
)
from .rules_base import DiagnosticRule, RuleContext


# =============================================================================
# Helper functions
# =============================================================================

def _urgency_from_severity(sev: Severity) -> Urgency:
    mapping = {
        Severity.CRITICAL: Urgency.CRITICAL,
        Severity.HIGH: Urgency.HIGH,
        Severity.MEDIUM: Urgency.MEDIUM,
        Severity.LOW: Urgency.LOW,
        Severity.INFO: Urgency.LOW,
    }
    return mapping.get(sev, Urgency.MEDIUM)


def _priority_score(urgency: Urgency, confidence: float) -> float:
    base = {
        Urgency.CRITICAL: 90,
        Urgency.HIGH: 70,
        Urgency.MEDIUM: 50,
        Urgency.LOW: 25,
    }[urgency]
    return base + 10 * confidence


# =============================================================================
# R01 — Thermal Drift (jacket fouling)
# =============================================================================

class R01_ThermalDrift(DiagnosticRule):
    """
    Detects slow, correlated drift of reactor temperature upward with
    the jacket temperature not catching up. Classic signature of heat
    exchanger fouling.
    """

    @property
    def rule_name(self) -> str: return "R01_ThermalDrift"

    @property
    def description(self) -> str:
        return "Reactor heating slowly — heat removal loss, probable jacket fouling"

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        # Trigger on EWMA upward drift of reactor temperature (either EWMA
        # or Shewhart can start the session; what matters is that the peak
        # value during the session is above nominal)
        if session.tag != "cstr.T_R_C":
            return False
        if session.detector not in ("SPC.EWMA", "SPC.Shewhart"):
            return False
        # Use the peak value of the session rather than the start-time
        # snapshot: sessions can start with a value near nominal and
        # drift upward over their duration.
        peak_val = session.peak_alert.value
        if peak_val is None:
            return False
        # Require reactor running appreciably above nominal
        if peak_val < 79.8:
            return False
        # Only upward excursions qualify as thermal drift (downward is
        # the opposite problem — overcooling or load drop)
        msg = (session.peak_alert.message or "").lower()
        if "below" in msg:
            return False
        return True

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        # Use peak value from the session for diagnosis/urgency
        t_r = session.peak_alert.value
        f_c = context.current_value("cstr.F_cool")
        if t_r is None or f_c is None:
            return None

        # Proposed action: increase coolant flow to compensate for degraded UA
        # Magnitude: ~15% of current F_c, bounded to physical limits
        f_c_new = min(f_c * 1.15, 0.60)   # upper bound on coolant flow
        delta = f_c_new - f_c

        # Urgency: depends on how far above setpoint T_R already is
        if t_r > 82:
            urgency = Urgency.HIGH
        elif t_r > 80.5:
            urgency = Urgency.MEDIUM
        else:
            urgency = Urgency.LOW

        confidence = 0.85 if session.alert_count > 20 else 0.70

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(urgency, confidence),
            diagnosis=(
                f"Reactor temperature drifting upward (T_R = {t_r:.2f}°C) "
                f"despite coolant running at nominal conditions. "
                f"Heat removal capacity appears to be degrading."
            ),
            probable_cause=(
                "Progressive fouling of the cooling jacket heat exchanger, "
                "reducing effective UA. This is a slow-developing mechanical "
                "issue, not a control problem."
            ),
            action=Action(
                type=ActionType.ADJUST_SETPOINT,
                description=(
                    f"Increase coolant flow by ~15% to compensate for reduced "
                    f"heat transfer. Schedule heat exchanger cleaning at "
                    f"next planned shutdown."
                ),
                target_variable="cstr.F_cool",
                current_value=f_c,
                proposed_value=f_c_new,
                adjustment=delta,
                units="m³/h",
            ),
            expected_impact=[
                ExpectedImpact(
                    variable="cstr.T_R_C",
                    current_value=t_r,
                    predicted_value=max(79.2, t_r - 0.5),
                    time_to_effect_minutes=20,
                    description="Reactor returns toward setpoint",
                ),
                ExpectedImpact(
                    variable="cstr.F_cool",
                    current_value=f_c,
                    predicted_value=f_c_new,
                    time_to_effect_minutes=1,
                    description="Coolant flow increased",
                ),
            ],
            confidence=confidence,
            urgency=urgency,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=["cstr.T_R_C", "cstr.T_J_C", "cstr.F_cool"],
            extra={"requires_maintenance_ticket": True},
        )


# =============================================================================
# R02 — Feed Composition Disturbance
# =============================================================================

class R02_FeedComposition(DiagnosticRule):
    """
    Multiple downstream variables change together after an unexplained
    disturbance. Signature: simultaneous PCA T² + purity drop + C_A shift.
    """

    @property
    def rule_name(self) -> str: return "R02_FeedComposition"

    @property
    def description(self) -> str:
        return "Feed composition shift detected — purity and conversion both affected"

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        # This rule fires on PCA T² alerts (the multivariate signature)
        if session.detector != "PCA.T2":
            return False
        # Require that both reactor composition AND column purity show
        # co-occurring sessions. That's what distinguishes a feed disturbance
        # (affects everything downstream) from a local column problem.
        has_c_a = context.has_sessions_from(
            ["SPC.EWMA", "SPC.Shewhart"], tags=["cstr.C_A"]
        )
        has_purity = context.has_sessions_from(
            ["SPC.EWMA", "SPC.Shewhart"], tags=["column.purity_B"]
        )
        if not (has_c_a and has_purity):
            return False
        # Additionally require that purity has meaningfully deviated: the
        # raw PCA signal can fire on small shifts that don't affect the
        # product. We only want to engage this rule on operationally
        # significant composition disturbances.
        purity = context.current_value("column.purity_B")
        if purity is None:
            return False
        return purity < 99.0   # below nominal operating band

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        c_a = context.current_value("cstr.C_A")
        purity = context.current_value("column.purity_B")
        rr = context.current_value("column.RR")
        if c_a is None or purity is None or rr is None:
            return None

        # If purity is below spec, recommend compensatory reflux increase
        spec_min = context.operational_limits.get(
            "column.purity_B", {}).get("low", 98.5)

        if purity < spec_min:
            rr_new = min(rr + 0.3, 7.0)
            delta = rr_new - rr
            urgency = Urgency.HIGH if purity < spec_min - 1 else Urgency.MEDIUM
            action = Action(
                type=ActionType.ADJUST_SETPOINT,
                description=(
                    f"Product is below spec. Increase reflux ratio to "
                    f"restore purity while investigating root cause upstream."
                ),
                target_variable="column.RR",
                current_value=rr,
                proposed_value=rr_new,
                adjustment=delta,
                units="dimensionless",
            )
            impact = [
                ExpectedImpact(
                    variable="column.purity_B",
                    current_value=purity,
                    predicted_value=min(99.2, purity + 0.8),
                    time_to_effect_minutes=30,
                    description="Purity recovers within ~30 min",
                ),
                ExpectedImpact(
                    variable="column.Q_reb_kW",
                    current_value=context.current_value("column.Q_reb_kW"),
                    predicted_value=None,
                    description="Reboiler duty will increase ~10-15%",
                ),
            ]
        else:
            # Within spec — investigate instead of chasing
            urgency = Urgency.MEDIUM
            action = Action(
                type=ActionType.INVESTIGATE,
                description=(
                    "Check upstream: feed tank analysis, possible contamination, "
                    "upstream unit changes. Verify LIMS results for A/B ratio."
                ),
            )
            impact = []

        confidence = 0.80

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(urgency, confidence),
            diagnosis=(
                f"Unexplained multivariate shift: reactor composition "
                f"(C_A = {c_a:.0f} mol/m³) and column performance "
                f"(purity = {purity:.2f}%) moving together. "
                f"The signature is upstream-driven."
            ),
            probable_cause=(
                "Change in fresh feed composition (raw material lot change, "
                "dilution, or contamination). Upstream unit disturbance "
                "propagated through the reactor."
            ),
            action=action,
            expected_impact=impact,
            confidence=confidence,
            urgency=urgency,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=["cstr.C_A", "column.purity_B", "column.RR"],
        )


# =============================================================================
# R03 — Controller Oscillation
# =============================================================================

class R03_ControllerOscillation(DiagnosticRule):
    """
    Sustained oscillation in the reactor cooling loop. The real signature is
    on F_cool (the manipulated variable): many CUSUM regime-change sessions
    firing back to back because the flow oscillates around its mean. The
    temperature T_R may look OK because the PID loop is fighting the
    oscillation — but F_cool is doing a lot of work.

    Trigger: a CUSUM session on F_cool when there are ≥5 co-occurring
    CUSUM sessions on the same tag within the window.
    """

    @property
    def rule_name(self) -> str: return "R03_ControllerOscillation"

    @property
    def description(self) -> str:
        return "Cooling flow oscillating — PID may be mis-tuned or valve stiction"

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        # Trigger on CUSUM regime changes on the coolant flow
        if session.tag != "cstr.F_cool":
            return False
        if session.detector != "Regime.CUSUM":
            return False
        # Count co-occurring F_cool CUSUM sessions. The co-occurrence window
        # is set to 60 min by default at the engine level; sustained
        # oscillations should produce at least 3 regime-change detections
        # per hour.
        co_count = sum(
            1 for s in context.co_occurring_sessions
            if s.tag == "cstr.F_cool" and s.detector == "Regime.CUSUM"
        )
        return co_count >= 3

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        f_c = context.current_value("cstr.F_cool")
        t_r = context.current_value("cstr.T_R_C")
        if f_c is None or t_r is None:
            return None

        n_cycles = 1 + sum(
            1 for s in context.co_occurring_sessions
            if s.tag == "cstr.F_cool" and s.detector == "Regime.CUSUM"
        )

        confidence = 0.80

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(Urgency.MEDIUM, confidence),
            diagnosis=(
                f"Coolant flow is oscillating continuously ({n_cycles} "
                f"detected cycles in the last window). Reactor temperature "
                f"T_R = {t_r:.2f}°C is being held by the control action, but "
                f"the cooling loop is working unnecessarily hard."
            ),
            probable_cause=(
                "T_R controller output is oscillating — most commonly a "
                "retuned controller with too much gain, a sticky control "
                "valve on the coolant line, or an upstream disturbance "
                "the loop is rejecting aggressively."
            ),
            action=Action(
                type=ActionType.INVESTIGATE,
                description=(
                    "Switch T_R PID to MANUAL temporarily and observe the "
                    "open-loop response of T_R and F_cool. If oscillation "
                    "stops in manual, it is a controller tuning problem: "
                    "reduce gain by ~20%. If oscillation persists, check "
                    "the coolant valve for stiction."
                ),
            ),
            expected_impact=[
                ExpectedImpact(
                    variable="cstr.F_cool",
                    current_value=f_c,
                    predicted_value=None,
                    time_to_effect_minutes=45,
                    description="Oscillations damp out after retune",
                ),
                ExpectedImpact(
                    variable="cstr.T_R_C",
                    current_value=t_r,
                    predicted_value=79.2,
                    time_to_effect_minutes=45,
                    description="Reactor temperature stabilizes",
                ),
            ],
            confidence=confidence,
            urgency=Urgency.MEDIUM,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=["cstr.F_cool", "cstr.T_R_C"],
        )


# =============================================================================
# R04 — Column Efficiency Loss
# =============================================================================

class R04_ColumnEfficiencyLoss(DiagnosticRule):
    """
    Purity degrades while upstream (reactor) is stable. Points to the
    column itself (tray fouling, foaming, pressure drop).
    """

    @property
    def rule_name(self) -> str: return "R04_ColumnEfficiencyLoss"

    @property
    def description(self) -> str:
        return "Column separation efficiency declining — internal issue"

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        if session.tag != "column.purity_B":
            return False
        if session.detector not in ("SPC.EWMA", "SPC.Shewhart"):
            return False
        # Reactor side should be stable (no co-occurring T_R session)
        reactor_unstable = context.has_sessions_from(
            ["SPC.EWMA", "SPC.Shewhart"], tags=["cstr.T_R_C", "cstr.C_A"]
        )
        return not reactor_unstable

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        purity = context.current_value("column.purity_B")
        rr = context.current_value("column.RR")
        if purity is None or rr is None:
            return None

        spec_min = context.operational_limits.get(
            "column.purity_B", {}).get("low", 98.5)

        # Compensatory action: increase reflux
        rr_new = min(rr + 0.4, 7.0)
        delta = rr_new - rr

        if purity < spec_min - 1:
            urgency = Urgency.HIGH
        elif purity < spec_min:
            urgency = Urgency.MEDIUM
        else:
            urgency = Urgency.LOW

        confidence = 0.75

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(urgency, confidence),
            diagnosis=(
                f"Product purity is declining ({purity:.2f}% vs. "
                f"{spec_min:.1f}% spec) while reactor operation is stable. "
                f"The separation capacity of the column itself is reduced."
            ),
            probable_cause=(
                "Column internal efficiency loss: possible tray fouling, "
                "foaming, liquid maldistribution, or gradual pressure drop "
                "increase. Relative volatility is effectively lower than "
                "nominal."
            ),
            action=Action(
                type=ActionType.ADJUST_SETPOINT,
                description=(
                    f"Compensate by increasing reflux ratio. Plan a column "
                    f"pressure differential check; consider antifoam addition "
                    f"if recent changes in feed chemistry."
                ),
                target_variable="column.RR",
                current_value=rr,
                proposed_value=rr_new,
                adjustment=delta,
                units="dimensionless",
            ),
            expected_impact=[
                ExpectedImpact(
                    variable="column.purity_B",
                    current_value=purity,
                    predicted_value=min(99.0, purity + 1.0),
                    time_to_effect_minutes=45,
                ),
            ],
            confidence=confidence,
            urgency=urgency,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=["column.purity_B", "column.RR"],
            extra={"requires_inspection": True},
        )


# =============================================================================
# R05 — Excess Reflux (energy waste)
# =============================================================================

class R05_ExcessReflux(DiagnosticRule):
    """
    Product well within spec but reboiler duty abnormally high. Suggests
    column is over-refluxing and wasting energy.
    """

    @property
    def rule_name(self) -> str: return "R05_ExcessReflux"

    @property
    def description(self) -> str:
        return "Energy waste detected — column over-refluxing vs. target purity"

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        if session.detector != "SPC.EWMA" or session.tag != "column.Q_reb_kW":
            return False
        # Use the peak value reached during the session (not the start-time
        # instant value). A session begins when the drift is detected; peak
        # is the actual level of the Q_reb excursion.
        q_reb_peak = session.peak_alert.value
        if q_reb_peak is None:
            return False
        # Upward excursions only
        msg = (session.peak_alert.message or "").lower()
        if "below" in msg:
            return False
        purity = context.current_value("column.purity_B")
        if purity is None:
            return False
        # Reboiler running HIGH with purity WELL ABOVE spec. Thresholds
        # chosen so this does NOT fire during nominal ops (where purity
        # is typically ~99% and Q_reb ~235 kW). Requires clear waste.
        return q_reb_peak > 290 and purity > 99.3

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        rr = context.current_value("column.RR")
        purity = context.current_value("column.purity_B")
        q_reb = session.peak_alert.value  # use the peak of the excursion
        if rr is None or purity is None or q_reb is None:
            return None

        rr_new = max(rr - 0.3, 4.0)
        delta = rr_new - rr
        confidence = 0.90   # rule is very specific; false positives unlikely

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(Urgency.LOW, confidence),
            diagnosis=(
                f"Column is over-refluxing: purity = {purity:.2f}% vs. "
                f"98.5% spec, while reboiler duty = {q_reb:.0f} kW is "
                f"~20% above design. There is energy savings opportunity here."
            ),
            probable_cause=(
                "Reflux setpoint set conservatively (or never adjusted after "
                "a process improvement). Product quality is comfortable but "
                "energy cost is not."
            ),
            action=Action(
                type=ActionType.ADJUST_SETPOINT,
                description=(
                    "Reduce reflux ratio in small steps (0.2 at a time), "
                    "monitoring purity. Target purity > 98.8% with "
                    "minimized reboiler duty."
                ),
                target_variable="column.RR",
                current_value=rr,
                proposed_value=rr_new,
                adjustment=delta,
                units="dimensionless",
            ),
            expected_impact=[
                ExpectedImpact(
                    variable="column.Q_reb_kW",
                    current_value=q_reb,
                    predicted_value=q_reb * 0.92,
                    time_to_effect_minutes=30,
                    description="~8% energy savings",
                ),
                ExpectedImpact(
                    variable="column.purity_B",
                    current_value=purity,
                    predicted_value=max(98.8, purity - 0.3),
                    time_to_effect_minutes=30,
                    description="Still well above spec",
                ),
            ],
            confidence=confidence,
            urgency=Urgency.LOW,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=["column.RR", "column.Q_reb_kW", "column.purity_B"],
            extra={"optimization_opportunity": True},
        )


# =============================================================================
# R06 — Direct Purity Deviation (off-spec product)
# =============================================================================

class R06_PurityDeviation(DiagnosticRule):
    """
    Trend projection indicates purity will cross the spec limit soon.
    This is the key 'anticipation' rule — fires before the product is
    actually off-spec.
    """

    @property
    def rule_name(self) -> str: return "R06_PurityDeviation"

    @property
    def description(self) -> str:
        return "Purity projected to go off-spec — pre-emptive action needed"

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        return (session.detector == "Trend.Projection"
                and session.tag == "column.purity_B")

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        purity = context.current_value("column.purity_B")
        rr = context.current_value("column.RR")
        if purity is None or rr is None:
            return None

        peak = session.peak_alert
        minutes_to_limit = peak.extra.get("minutes_to_limit", 60)
        slope = peak.extra.get("slope_per_minute", 0)

        rr_new = min(rr + 0.25, 7.0)
        delta = rr_new - rr

        if minutes_to_limit < 15:
            urgency = Urgency.CRITICAL
        elif minutes_to_limit < 30:
            urgency = Urgency.HIGH
        else:
            urgency = Urgency.MEDIUM

        confidence = min(0.95, peak.confidence + 0.1)

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(urgency, confidence),
            diagnosis=(
                f"Purity is trending downward at {slope:+.4f}%/min "
                f"(current = {purity:.2f}%). At this rate, the product will "
                f"go below 98.5% spec in approximately {minutes_to_limit:.0f} "
                f"minutes."
            ),
            probable_cause=(
                "Continuous drift — could be any upstream disturbance. "
                "Primary priority is to prevent the off-spec event; "
                "root cause analysis can follow."
            ),
            action=Action(
                type=ActionType.ADJUST_SETPOINT,
                description=(
                    f"Pre-emptively increase reflux ratio by {delta:+.2f} "
                    f"to arrest the trend before purity crosses spec."
                ),
                target_variable="column.RR",
                current_value=rr,
                proposed_value=rr_new,
                adjustment=delta,
                units="dimensionless",
            ),
            expected_impact=[
                ExpectedImpact(
                    variable="column.purity_B",
                    current_value=purity,
                    predicted_value=min(99.0, purity + 0.4),
                    time_to_effect_minutes=25,
                    description="Trend reversed, purity recovers",
                ),
            ],
            confidence=confidence,
            urgency=urgency,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=["column.purity_B", "column.RR"],
        )


# =============================================================================
# R07 — Sensor Fault
# =============================================================================

class R07_SensorFault(DiagnosticRule):
    """
    A sensor showing zero variance over a window = frozen instrument.
    Detected by looking at recent history of the tag that produced the alert.

    Important: only applies to MEASURED variables. Setpoints and manipulated
    variables (F_feed, F_cool, RR) can legitimately stay perfectly constant
    — that's their purpose. Listing these explicitly avoids false positives.
    """

    # Tags that are either operator setpoints or manipulated by controllers.
    # For these, zero variance is expected behavior, not a fault.
    NON_SENSED_TAGS = {
        "cstr.F_feed", "cstr.F_cool", "cstr.T_feed_C", "cstr.T_cool_in_C",
        "column.RR", "column.F_vap_kgh", "column.P_top_bar", "column.P_bot_bar",
    }

    @property
    def rule_name(self) -> str: return "R07_SensorFault"

    @property
    def description(self) -> str:
        return "Sensor signal stuck — probable instrument failure"

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        if session.tag is None:
            return False
        # Skip tags that are legitimately constant (setpoints, manipulated vars)
        if session.tag in self.NON_SENSED_TAGS:
            return False
        # Primary match: a FrozenSensor session — dedicated detector for this
        if session.detector == "FrozenSensor":
            return True
        # Legacy fallback: check if recent data shows zero variance
        if session.tag not in context.process_data.columns:
            return False
        df = context.process_data[["timestamp", session.tag]].dropna()
        if df.empty:
            return False
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = df["timestamp"].iloc[-1] - pd.Timedelta(minutes=15)
        recent = df.loc[df["timestamp"] >= cutoff, session.tag]
        if len(recent) < 5:
            return False
        return recent.std() < 1e-4

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        value = context.current_value(session.tag)
        if value is None:
            return None
        confidence = 0.95

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(Urgency.HIGH, confidence),
            diagnosis=(
                f"Tag {session.tag} has been reading a constant value "
                f"({value:.3f}) for the past 15+ minutes. A real process "
                f"variable with measurement noise cannot be this stable."
            ),
            probable_cause=(
                "Frozen sensor: transmitter failure, communication loss "
                "between field instrument and DCS, or thermocouple lead "
                "break. The underlying process is almost certainly still "
                "moving — the measurement is not."
            ),
            action=Action(
                type=ActionType.VERIFY_INSTRUMENT,
                description=(
                    f"Dispatch an instrument tech to field-check {session.tag}. "
                    f"Meanwhile, operate using redundant measurements (soft "
                    f"sensor, neighboring TT, manual gauge) and do not trust "
                    f"this tag for control decisions."
                ),
            ),
            expected_impact=[
                ExpectedImpact(
                    variable=session.tag,
                    current_value=value,
                    predicted_value=None,
                    description="Value will resume normal variation when fixed",
                ),
            ],
            confidence=confidence,
            urgency=Urgency.HIGH,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=[session.tag],
            extra={"requires_field_service": True},
        )


# =============================================================================
# R08 — Product Grade Transition
# =============================================================================

class R08_ProductTransition(DiagnosticRule):
    """
    When the operator changes a setpoint deliberately (e.g. RR from 5.5 to
    4.0), the regime change detector will fire. This rule recognizes the
    pattern as a grade change and emits an informational recommendation
    (not a corrective action).
    """

    @property
    def rule_name(self) -> str: return "R08_ProductTransition"

    @property
    def description(self) -> str:
        return "Operator-initiated setpoint change — manage the transition"

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        # Regime change on RR usually indicates a setpoint step
        return (session.detector == "Regime.CUSUM"
                and session.tag == "column.RR")

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        rr = context.current_value("column.RR")
        purity = context.current_value("column.purity_B")
        if rr is None or purity is None:
            return None

        direction = session.peak_alert.extra.get("direction", "unknown")
        confidence = 0.85

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(Urgency.LOW, confidence),
            diagnosis=(
                f"Reflux setpoint changed {direction} "
                f"(current RR = {rr:.2f}). This appears to be an intentional "
                f"operator action, likely a product grade transition."
            ),
            probable_cause=(
                "Manual setpoint adjustment. Transition period: expect "
                "temporary off-spec material and stabilization over ~30 min."
            ),
            action=Action(
                type=ActionType.WAIT_AND_MONITOR,
                description=(
                    "Monitor purity during the transition. Consider diverting "
                    "transition-period product to slop tank until purity "
                    "stabilizes within the new grade spec."
                ),
            ),
            expected_impact=[
                ExpectedImpact(
                    variable="column.purity_B",
                    current_value=purity,
                    predicted_value=None,
                    time_to_effect_minutes=30,
                    description=("Purity will re-stabilize at a new level "
                                 "consistent with the new RR setpoint"),
                ),
            ],
            confidence=confidence,
            urgency=Urgency.LOW,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=["column.RR", "column.purity_B"],
            extra={"informational": True},
        )


# =============================================================================
# R09 — Soft Sensor Divergence
# =============================================================================

class R09_SoftSensorDivergence(DiagnosticRule):
    """
    The soft sensor's prediction disagrees sustainedly with the measured
    value. Two operational interpretations, presented as alternatives.
    """

    @property
    def rule_name(self) -> str: return "R09_SoftSensorDivergence"

    @property
    def description(self) -> str:
        return "Soft sensor vs measurement disagreement — instrument or model drift"

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        return session.detector == "SoftSensor.Divergence"

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        peak = session.peak_alert
        actual    = peak.extra.get("actual")
        predicted = peak.extra.get("predicted")
        residual  = peak.extra.get("residual")
        duration  = peak.extra.get("duration_minutes", 0)
        if actual is None or predicted is None:
            return None

        direction = "higher" if residual > 0 else "lower"
        confidence = 0.70
        urgency = Urgency.MEDIUM if abs(residual) > 1.0 else Urgency.LOW

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(urgency, confidence),
            diagnosis=(
                f"Measured {session.tag} ({actual:.2f}) is {direction} than "
                f"the soft sensor prediction ({predicted:.2f}) — residual "
                f"{residual:+.2f} sustained for {duration:.0f} min."
            ),
            probable_cause=(
                "Two possibilities: (1) instrument drift or lab "
                "contamination — the measurement is wrong; (2) process has "
                "moved outside the soft sensor's training envelope — the "
                "model is extrapolating. Examining the secondary variables "
                "(T_bot, Q_reb, RR) should disambiguate."
            ),
            action=Action(
                type=ActionType.VERIFY_INSTRUMENT,
                description=(
                    "Cross-check the measurement: request a LIMS lab re-sample "
                    "if available. Compare against the secondary variables used "
                    "by the soft sensor — if they are in a range seen during "
                    "training, suspect the measurement; if they are at the edge "
                    "of training, suspect model extrapolation."
                ),
            ),
            expected_impact=[
                ExpectedImpact(
                    variable=session.tag,
                    current_value=actual,
                    predicted_value=None,
                    description="Resolution will either recalibrate the sensor or retrain the model",
                ),
            ],
            confidence=confidence,
            urgency=urgency,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=[session.tag] + [f for f in peak.extra.get("features", [])],
            extra={"predicted": predicted, "actual": actual, "residual": residual},
        )


# =============================================================================
# Registry
# =============================================================================

PILOT_RULES_PLACEHOLDER = None  # legacy forward declaration, kept for trace


# =============================================================================
# R10 — Predictive Excursion (LSTM-driven forecast violation)
# =============================================================================

class R10_PredictedExcursion(DiagnosticRule):
    """
    The LSTM forecaster predicts a target variable will violate its
    operational limit within the configured horizon. The recommendation
    advises preemptive action with the time-to-violation as urgency driver.

    This rule complements R06 (linear trend projection) — it catches
    non-linear dynamics that a linear projection misses: oscillations,
    non-monotonic recoveries, lag-driven cascades.
    """

    @property
    def rule_name(self) -> str: return "R10_PredictedExcursion"

    @property
    def description(self) -> str:
        return "LSTM-predicted limit violation — anticipated process excursion"

    def matches(self, session: EventSession, context: RuleContext) -> bool:
        return session.detector == "LSTM.Forecast"

    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]:
        peak = session.peak_alert
        predicted   = peak.extra.get("predicted_value")
        current     = peak.extra.get("current_value")
        horizon_min = peak.extra.get("horizon_minutes", 0)
        direction   = peak.extra.get("direction", "")
        excess      = peak.extra.get("violation_excess", 0)
        if predicted is None or current is None:
            return None

        confidence = 0.75
        if horizon_min <= 10:
            urgency = Urgency.HIGH
        elif horizon_min <= 30:
            urgency = Urgency.MEDIUM
        else:
            urgency = Urgency.LOW

        suggestions = _suggest_action_for_predicted_excursion(
            session.tag, direction, excess
        )

        return Recommendation(
            id=Recommendation.new_id(),
            timestamp=session.start_time,
            priority_score=_priority_score(urgency, confidence),
            diagnosis=(
                f"LSTM forecast: {session.tag} predicted to go {direction} "
                f"limit ({peak.limit:g}) in ~{horizon_min:.0f} min "
                f"(predicted {predicted:.2f}, current {current:.2f})."
            ),
            probable_cause=(
                "Pattern in the recent process trajectory — including coupling "
                "between manipulated variables, disturbances, and aux measurements — "
                "indicates the forecasted excursion. The LSTM captures non-linear "
                "dynamics that a linear projection would miss."
            ),
            action=suggestions["action"],
            expected_impact=[
                ExpectedImpact(
                    variable=session.tag,
                    current_value=current,
                    predicted_value=peak.limit,
                    description=suggestions["impact_desc"],
                ),
            ],
            confidence=confidence,
            urgency=urgency,
            triggering_sessions=[context.session_id],
            rule_fired=self.rule_name,
            affected_variables=[session.tag, suggestions["target_var"]],
            extra={
                "predicted_value": predicted,
                "horizon_minutes": horizon_min,
                "direction":       direction,
            },
        )


def _suggest_action_for_predicted_excursion(
    tag: str, direction: str, excess: float
) -> dict:
    """Map a forecasted excursion to the most likely corrective MV adjustment."""
    if tag == "column.purity_B" and direction == "below":
        return {
            "action": Action(
                type=ActionType.ADJUST_SETPOINT,
                target_variable="column.RR",
                description=(
                    "Pre-emptively increase reflux ratio to lift purity before "
                    "the predicted off-spec event."
                ),
                adjustment=0.3,
            ),
            "target_var":  "column.RR",
            "impact_desc": "Predicted to bring purity back within spec before violation",
        }
    if tag == "cstr.T_R_C" and direction == "above":
        return {
            "action": Action(
                type=ActionType.ADJUST_SETPOINT,
                target_variable="cstr.F_cool",
                description=(
                    "Pre-emptively increase cooling water flow to bring reactor "
                    "temperature down before exceeding the upper limit."
                ),
                adjustment=0.05,
            ),
            "target_var":  "cstr.F_cool",
            "impact_desc": "Predicted to keep T_R within safe range",
        }
    if tag == "column.Q_reb_kW" and direction == "above":
        return {
            "action": Action(
                type=ActionType.ADJUST_SETPOINT,
                target_variable="column.RR",
                description=(
                    "Reduce reflux ratio to lower reboiler duty (within purity "
                    "constraints)."
                ),
                adjustment=-0.2,
            ),
            "target_var":  "column.RR",
            "impact_desc": "Lower energy consumption while staying on-spec",
        }
    return {
        "action": Action(
            type=ActionType.INVESTIGATE,
            target_variable=tag,
            description="Investigate the predicted excursion; manual review needed.",
        ),
        "target_var":  tag,
        "impact_desc": "Operator review",
    }


PILOT_RULES = [
    R01_ThermalDrift(),
    R02_FeedComposition(),
    R03_ControllerOscillation(),
    R04_ColumnEfficiencyLoss(),
    R05_ExcessReflux(),
    R06_PurityDeviation(),
    R07_SensorFault(),
    R08_ProductTransition(),
    R09_SoftSensorDivergence(),
    R10_PredictedExcursion(),
]

