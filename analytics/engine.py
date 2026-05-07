"""
Axion AI - Analytical Engine
============================

The orchestrator that runs all detectors on a dataset and returns a unified
alerts timeline. This is the single entry point for the analytics layer.

    engine = AnalyticalEngine(tags=[...])
    engine.fit(df)
    alerts = engine.run(df)
    sessions = engine.run_sessions(df)   # grouped version

Future Recommendation Engine (Task 4) consumes the session timeline.
"""

from __future__ import annotations
from typing import List, Optional, Dict
import pandas as pd

from .alerts import Alert, alerts_to_dataframe
from .sessions import EventSession, group_alerts_into_sessions, sessions_to_dataframe
from .spc import SPCDetector
from .pca import PCADetector
from .trend import TrendDetector, PILOT_OPERATIONAL_LIMITS
from .regime import RegimeChangeDetector
from .frozen import FrozenSensorDetector, DEFAULT_MEASURED_TAGS


# Default tag set for the pilot process. For a different process, the caller
# passes its own list of tags.
PILOT_TAGS = [
    "cstr.T_R_C", "cstr.T_J_C", "cstr.C_A", "cstr.conversion",
    "cstr.F_feed", "cstr.F_cool",
    "column.purity_B", "column.x_D", "column.T_top_C", "column.T_bot_C",
    "column.RR", "column.Q_reb_kW",
]


class AnalyticalEngine:
    """
    Runs SPC + PCA + Trend + Regime detectors on a dataset.

    Each detector can be disabled at construction. Operational limits used
    by the trend detector default to the pilot process limits but can be
    overridden.

    Two usage patterns:

    1. Single dataset (train + evaluate on the same data):
        engine.fit(df)
        alerts = engine.run(df, post_training_only=True)  # skips training window

    2. Separate training and evaluation datasets:
        engine.fit(df_train)
        alerts = engine.run(df_eval)  # whole eval set is monitored

    Extra options:
    - warmup_minutes: discard alerts in the first N minutes of the evaluation
      dataset. Use this when fitting on one simulation run and evaluating on
      another: the two runs have slightly different steady-state baselines
      due to stochastic warm-up, and the first few minutes produce spurious
      "startup transient" alerts. 10-30 minutes is typical.
    """

    def __init__(
        self,
        tags: Optional[List[str]] = None,
        enable_spc: bool = True,
        enable_pca: bool = True,
        enable_trend: bool = True,
        enable_regime: bool = True,
        enable_frozen: bool = True,
        operational_limits: Optional[Dict] = None,
        training_fraction: float = 0.25,
        warmup_minutes: float = 0.0,
        session_gap_minutes: float = 30.0,
        extra_detectors: Optional[List] = None,
    ):
        self.tags = list(tags) if tags is not None else list(PILOT_TAGS)
        self.training_fraction = training_fraction
        self.operational_limits = operational_limits or PILOT_OPERATIONAL_LIMITS
        self.warmup_minutes = warmup_minutes
        self.session_gap_minutes = session_gap_minutes

        self.spc = SPCDetector(self.tags, training_fraction=training_fraction) if enable_spc else None
        self.pca = PCADetector(self.tags, training_fraction=training_fraction) if enable_pca else None
        self.trend = TrendDetector(self.tags, self.operational_limits) if enable_trend else None
        self.regime = RegimeChangeDetector(self.tags, training_fraction=training_fraction) if enable_regime else None
        # Frozen sensor detector only monitors MEASURED tags (subset of self.tags)
        frozen_tags = [t for t in self.tags if t in DEFAULT_MEASURED_TAGS]
        self.frozen = FrozenSensorDetector(tags=frozen_tags) if enable_frozen else None
        # Pluggable extra detectors — anything implementing fit/run interface.
        # Used for SoftSensorDetector and future additions without modifying
        # the engine core.
        self.extra_detectors = list(extra_detectors) if extra_detectors else []

    def fit(self, df: pd.DataFrame) -> None:
        if self.spc:    self.spc.fit(df)
        if self.pca:    self.pca.fit(df)
        # Trend detector does not need a fit
        if self.regime: self.regime.fit(df)
        if self.frozen: self.frozen.fit(df)
        for det in self.extra_detectors:
            try:
                det.fit(df)
            except Exception:
                pass

    def run(self, df: pd.DataFrame, post_training_only: bool = False) -> List[Alert]:
        """
        Run all detectors. If post_training_only=True, restrict evaluation to
        samples beyond the training window. Alerts within the warmup period
        are filtered out.
        """
        eval_df = df
        if post_training_only:
            n_train = int(len(df) * self.training_fraction)
            eval_df = df.iloc[n_train:].reset_index(drop=True)

        all_alerts: List[Alert] = []
        if self.spc:    all_alerts.extend(self.spc.run(eval_df))
        if self.pca:    all_alerts.extend(self.pca.run(eval_df))
        if self.trend:  all_alerts.extend(self.trend.run(eval_df))
        if self.regime: all_alerts.extend(self.regime.run(eval_df))
        if self.frozen: all_alerts.extend(self.frozen.run(eval_df))
        for det in self.extra_detectors:
            try:
                all_alerts.extend(det.run(eval_df))
            except Exception:
                pass
        all_alerts.sort(key=lambda a: a.timestamp)

        # Filter warmup window
        if self.warmup_minutes > 0 and len(eval_df) > 0:
            t_start = pd.to_datetime(eval_df["timestamp"]).iloc[0]
            cutoff = t_start + pd.Timedelta(minutes=self.warmup_minutes)
            all_alerts = [a for a in all_alerts if a.timestamp >= cutoff]

        return all_alerts

    def run_to_dataframe(self, df: pd.DataFrame, post_training_only: bool = False) -> pd.DataFrame:
        return alerts_to_dataframe(self.run(df, post_training_only=post_training_only))

    def run_sessions(
        self,
        df: pd.DataFrame,
        post_training_only: bool = False,
    ) -> List[EventSession]:
        """
        Run all detectors and return EventSessions (grouped alerts).
        This is the canonical consumption interface for downstream modules.
        """
        alerts = self.run(df, post_training_only=post_training_only)
        return group_alerts_into_sessions(alerts, gap_minutes=self.session_gap_minutes)

    def run_sessions_to_dataframe(
        self,
        df: pd.DataFrame,
        post_training_only: bool = False,
    ) -> pd.DataFrame:
        return sessions_to_dataframe(self.run_sessions(df, post_training_only=post_training_only))
