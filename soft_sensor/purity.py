"""
Axion AI - Purity Soft Sensor
=============================

Production-oriented soft sensor for estimating `column.purity_B` from
secondary variables in the pilot process. Uses gradient boosting
regression with an ensemble-of-ensembles strategy for uncertainty
estimation.

Feature set (engineering-motivated)
-----------------------------------
The features are chosen based on distillation first principles, not by
blind feature engineering:

  - column.T_bot_C   — strongest predictor. By Antoine's law, the bottom
                       temperature at a given pressure is a direct
                       function of composition; for a binary separation
                       it's nearly 1:1 with purity in the nominal band.

  - column.T_top_C   — complements T_bot: together they pin down both ends
                       of the temperature profile, which is the fingerprint
                       of how well the column is separating.

  - column.RR        — reflux ratio is the primary manipulated variable;
                       at a given duty, higher RR = better separation.

  - column.Q_reb_kW  — reboiler duty. Combined with RR determines the
                       vapor traffic and therefore the separation achieved.

  - column.F_vap_kgh — vapor flow, tightly coupled with Q_reb (Q = λ·F) but
                       helps the model disambiguate when the liquid
                       heat of vaporization drifts.

  - column.P_top_bar — pressure affects relative volatility and thus the
                       temperature-composition relationship.

  - cstr.C_A         — upstream disturbance variable. A feed composition
                       change shifts the column's operating point well
                       before it shows in T_bot.

Deliberate exclusions
---------------------
  - column.purity_B itself (target leakage)
  - column.x_D, column.x_B_A (these are essentially the same measurement
    that we're trying to predict, just in a different form)

Algorithm
---------
sklearn's GradientBoostingRegressor with modest depth. We train an ensemble
of N identical models with different random seeds and bootstrap samples,
and report:
  - Prediction = mean of ensemble
  - Uncertainty = standard deviation across ensemble members

This gives operator-facing confidence intervals without requiring Bayesian
machinery. Width of the interval is meaningful: it grows when the operating
point is far from anything seen in training (extrapolation), which is
exactly when we want the operator to stop trusting the soft sensor.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base import SoftSensor, SoftSensorMetrics, compute_metrics


# Canonical feature set for the pilot process
PILOT_PURITY_FEATURES = [
    "column.T_bot_C",
    "column.T_top_C",
    "column.RR",
    "column.Q_reb_kW",
    "column.F_vap_kgh",
    "column.P_top_bar",
    "cstr.C_A",
]

TARGET_PURITY = "column.purity_B"


class PuritySoftSensor(SoftSensor):
    """
    Gradient-boosted ensemble soft sensor for product purity.

    Parameters
    ----------
    features : list of str
        Secondary variables to use as model inputs. Defaults to the pilot set.
    target : str
        Column name of the target variable. Defaults to column.purity_B.
    n_ensemble : int
        Number of models in the ensemble (each trained on a bootstrap sample).
    gb_params : dict
        Passed through to sklearn's GradientBoostingRegressor.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        target: str = TARGET_PURITY,
        n_ensemble: int = 5,
        gb_params: Optional[dict] = None,
    ):
        self.feature_names = list(features) if features is not None else list(PILOT_PURITY_FEATURES)
        self.target_name = target
        self.n_ensemble = n_ensemble
        self.gb_params = gb_params or dict(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=20,
            subsample=0.8,
        )
        self._models: List[GradientBoostingRegressor] = []
        self._scaler: Optional[StandardScaler] = None
        self.metrics: Optional[SoftSensorMetrics] = None

    # ---- fit ----

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> SoftSensorMetrics:
        """
        Train the ensemble. X must have all configured feature columns; y is
        the target array.
        """
        # Filter to the configured features, dropping any rows with NaNs
        X = X[self.feature_names].copy()
        y = pd.Series(y).copy()
        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

        if len(X) < 50:
            raise ValueError(f"Too few samples to fit soft sensor (got {len(X)})")

        # Time-aware split would be more realistic (don't leak future to past),
        # but sklearn's random split is fine for the MVP on simulator data
        # where scenarios are independent.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self._scaler = StandardScaler().fit(X_train.values)
        X_train_s = self._scaler.transform(X_train.values)
        X_test_s  = self._scaler.transform(X_test.values)

        # Train ensemble with different random seeds + bootstrap sampling
        rng = np.random.default_rng(random_state)
        self._models = []
        n = len(X_train_s)
        for i in range(self.n_ensemble):
            seed = int(rng.integers(0, 2**31 - 1))
            idx = rng.integers(0, n, size=n)    # bootstrap sample with replacement
            model = GradientBoostingRegressor(random_state=seed, **self.gb_params)
            model.fit(X_train_s[idx], y_train.values[idx])
            self._models.append(model)

        # Evaluate on held-out test set using the full ensemble
        y_pred_test = self._predict_ensemble(X_test_s)[0]
        self.metrics = compute_metrics(
            y_test.values, y_pred_test,
            feature_names=self.feature_names,
            target_name=self.target_name,
            n_train=len(X_train), n_test=len(X_test),
        )
        return self.metrics

    # ---- predict ----

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_with_confidence(X)[0]

    def predict_with_confidence(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (predictions, std) from the ensemble."""
        if not self._models:
            raise RuntimeError("Soft sensor not fitted")

        X_feat = X[self.feature_names].copy()
        # Forward-fill any NaNs; this matches what a real deployment sees
        # when a single tag briefly drops out.
        X_feat = X_feat.ffill().bfill()
        X_s = self._scaler.transform(X_feat.values)
        return self._predict_ensemble(X_s)

    def _predict_ensemble(
        self, X_scaled: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.column_stack([m.predict(X_scaled) for m in self._models])
        mean  = preds.mean(axis=1)
        std   = preds.std(axis=1, ddof=0)
        return mean, std

    # ---- feature importances (interpretability) ----

    def feature_importances(self) -> pd.DataFrame:
        """Average gain-based feature importance across the ensemble."""
        if not self._models:
            raise RuntimeError("Soft sensor not fitted")
        importances = np.array([m.feature_importances_ for m in self._models])
        mean_imp = importances.mean(axis=0)
        return pd.DataFrame({
            "feature":    self.feature_names,
            "importance": mean_imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
