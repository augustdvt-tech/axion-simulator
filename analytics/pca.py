"""
Axion AI - PCA Detector (Multivariate Statistical Process Control)
==================================================================

PCA-based multivariate process monitoring. The standard industrial approach
(see e.g. Kourti & MacGregor 1996, Joe Qin 2003) for detecting "abnormal
operation" in correlated process data.

Why PCA monitoring matters
--------------------------
Univariate SPC misses the kind of fault where each individual variable is still
within its 3-sigma limits, but the *relationship* between them has broken.
Example: in our distillation column under `quality_degradation`, the volatility
drops slowly. T_top, T_bot, RR, Q_reb each look fine in isolation, but their
joint configuration is now unusual. PCA catches that.

Two complementary statistics
----------------------------
1. HOTELLING T-squared: distance from the origin *within* the principal subspace.
   Detects unusually large variation along the modeled directions (e.g. a known
   variable went too far).

2. SPE (Squared Prediction Error, also called Q): residual after projecting onto
   the principal subspace. Detects deviations *outside* the modeled subspace —
   i.e. relationships between variables that the model has never seen before.
   This is what catches the most subtle faults.

Both statistics have analytical control limits (chi-squared for T2, Box's
approximation for SPE) computed from the training data.

Engineering decisions for the MVP
---------------------------------
- Standardize variables (z-score) before PCA: required because variables have
  different units (temperatures in C, flows in m3/h, etc.)
- Number of principal components: chosen to capture 90% of variance (configurable)
- Confidence level for control limits: 99% (configurable)
- Cooldown deduplication, same as SPC
- Top contributing variables are reported with each multivariate alert. This is
  what lets the Recommendation Engine (Task 4) say "T_R is contributing 60% to
  this anomaly" instead of just "something is wrong".
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_distribution

from .alerts import Alert, AlertType, Severity


# =============================================================================
# PCA model
# =============================================================================

@dataclass
class PCAModel:
    """A fitted PCA model with control limits."""
    tags: List[str]
    means: np.ndarray            # (p,)  feature means used for centering
    stds: np.ndarray             # (p,)  feature stds used for scaling
    components: np.ndarray       # (p, k) loadings matrix (kept k components)
    eigenvalues: np.ndarray      # (k,)  variance along each PC
    n_components: int
    variance_explained: float
    t2_limit: float              # control limit for T2 statistic
    spe_limit: float             # control limit for SPE statistic
    n_train: int                 # number of training samples


# =============================================================================
# PCA Detector
# =============================================================================

class PCADetector:
    """
    Multivariate process monitoring via PCA. Standard T2/SPE charts.

        detector = PCADetector(tags=[...])
        detector.fit(df_training)
        alerts = detector.run(df_full)
    """

    def __init__(
        self,
        tags: List[str],
        variance_threshold: float = 0.90,
        confidence_level: float = 0.99,
        cooldown_seconds: int = 600,
        training_fraction: float = 0.25,
        top_contributors: int = 3,
    ):
        self.tags = list(tags)
        self.variance_threshold = variance_threshold
        self.confidence_level = confidence_level
        self.cooldown_seconds = cooldown_seconds
        self.training_fraction = training_fraction
        self.top_contributors = top_contributors
        self.model: Optional[PCAModel] = None

    # ------ fit ------
    def fit(self, df: pd.DataFrame) -> None:
        n_train = int(len(df) * self.training_fraction)
        X = df.iloc[:n_train][self.tags].dropna().to_numpy(dtype=float)
        if X.shape[0] < 30:
            raise ValueError(f"Not enough training samples: {X.shape[0]} (need >= 30)")

        # Standardize
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd[sd < 1e-9] = 1e-9
        Z = (X - mu) / sd

        # SVD-based PCA. Components: (p,p) right singular vectors.
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        eigenvalues_all = (S ** 2) / (Z.shape[0] - 1)

        # Choose k components to reach variance_threshold
        cumvar = np.cumsum(eigenvalues_all) / np.sum(eigenvalues_all)
        k = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        k = max(1, min(k, Vt.shape[0] - 1))   # at least 1, leave room for residual space

        components = Vt[:k].T                # (p, k)
        eigvals = eigenvalues_all[:k]        # (k,)
        residual_eigvals = eigenvalues_all[k:]   # for SPE limit

        # Control limits
        t2_lim = self._t2_limit(k, n_train, self.confidence_level)
        spe_lim = self._spe_limit(residual_eigvals, self.confidence_level)

        self.model = PCAModel(
            tags=self.tags,
            means=mu, stds=sd,
            components=components,
            eigenvalues=eigvals,
            n_components=k,
            variance_explained=float(cumvar[k - 1]),
            t2_limit=t2_lim,
            spe_limit=spe_lim,
            n_train=n_train,
        )

    # ------ control limit formulas ------
    @staticmethod
    def _t2_limit(k: int, n: int, alpha: float) -> float:
        """Hotelling T2 control limit using F distribution (Tracy 1992)."""
        f_val = f_distribution.ppf(alpha, k, n - k)
        return k * (n - 1) * (n + 1) / (n * (n - k)) * f_val

    @staticmethod
    def _spe_limit(residual_eigvals: np.ndarray, alpha: float) -> float:
        """Box approximation for SPE control limit (Jackson & Mudholkar 1979)."""
        if len(residual_eigvals) == 0:
            return float("inf")
        theta1 = np.sum(residual_eigvals)
        theta2 = np.sum(residual_eigvals ** 2)
        theta3 = np.sum(residual_eigvals ** 3)
        if theta1 < 1e-12 or theta2 < 1e-12:
            return float("inf")
        h0 = 1 - 2 * theta1 * theta3 / (3 * theta2 ** 2)
        c_alpha = chi2.ppf(alpha, 1) ** 0.5
        term = (c_alpha * np.sqrt(2 * theta2 * h0 ** 2) / theta1
                + 1 + theta2 * h0 * (h0 - 1) / theta1 ** 2)
        return float(theta1 * term ** (1 / h0))

    # ------ scoring helpers ------
    def _project(self, x: np.ndarray):
        """Project a single observation: returns (T2, SPE, contributions_dict)."""
        m = self.model
        z = (x - m.means) / m.stds
        scores = z @ m.components                  # (k,)
        # T2
        t2 = float(np.sum(scores ** 2 / m.eigenvalues))
        # Reconstruction
        z_hat = scores @ m.components.T
        residual = z - z_hat
        spe = float(np.sum(residual ** 2))
        # Contribution per variable to SPE (squared residual contribution)
        contributions = {tag: float(residual[i] ** 2) for i, tag in enumerate(m.tags)}
        return t2, spe, contributions

    # ------ run ------
    def run(self, df: pd.DataFrame) -> List[Alert]:
        if self.model is None:
            raise RuntimeError("PCADetector must be fit() before run()")

        m = self.model
        alerts: List[Alert] = []
        cooldown = pd.Timedelta(seconds=self.cooldown_seconds)
        last_t2_alert: Optional[pd.Timestamp] = None
        last_spe_alert: Optional[pd.Timestamp] = None

        # Filter rows with all tags present
        sub = df[["timestamp"] + m.tags].dropna()
        timestamps = pd.to_datetime(sub["timestamp"]).reset_index(drop=True)
        X = sub[m.tags].to_numpy(dtype=float)

        for i in range(X.shape[0]):
            t2, spe, contribs = self._project(X[i])
            ts = timestamps.iloc[i]

            # T2 violation
            if t2 > m.t2_limit:
                if last_t2_alert is None or ts - last_t2_alert > cooldown:
                    severity = self._severity_from_ratio(t2 / m.t2_limit)
                    top_contrib = self._top_contributors(contribs)
                    alerts.append(Alert(
                        timestamp=ts,
                        detector="PCA.T2",
                        alert_type=AlertType.HOTELLING_T2,
                        severity=severity,
                        tag=None,
                        value=t2,
                        limit=m.t2_limit,
                        confidence=min(1.0, np.log1p(t2 / m.t2_limit) / np.log(10)),
                        contributors=top_contrib,
                        message=(
                            f"Hotelling T² = {t2:.2f} exceeds control limit "
                            f"({m.t2_limit:.2f}). Top contributors: "
                            f"{', '.join(f'{k} ({v:.2f})' for k, v in top_contrib.items())}"
                        ),
                        extra={"n_components": m.n_components},
                    ))
                    last_t2_alert = ts

            # SPE violation
            if spe > m.spe_limit:
                if last_spe_alert is None or ts - last_spe_alert > cooldown:
                    severity = self._severity_from_ratio(spe / m.spe_limit)
                    top_contrib = self._top_contributors(contribs)
                    alerts.append(Alert(
                        timestamp=ts,
                        detector="PCA.SPE",
                        alert_type=AlertType.SPE_Q,
                        severity=severity,
                        tag=None,
                        value=spe,
                        limit=m.spe_limit,
                        confidence=min(1.0, np.log1p(spe / m.spe_limit) / np.log(10)),
                        contributors=top_contrib,
                        message=(
                            f"SPE (Q) = {spe:.2f} exceeds control limit "
                            f"({m.spe_limit:.2f}). Variable correlations have changed. "
                            f"Top contributors: {', '.join(f'{k} ({v:.2f})' for k, v in top_contrib.items())}"
                        ),
                    ))
                    last_spe_alert = ts

        return alerts

    def _top_contributors(self, contribs: Dict[str, float]) -> Dict[str, float]:
        sorted_items = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
        return dict(sorted_items[: self.top_contributors])

    @staticmethod
    def _severity_from_ratio(ratio: float) -> Severity:
        """Map (statistic / limit) ratio to severity."""
        if ratio > 10:
            return Severity.CRITICAL
        if ratio > 4:
            return Severity.HIGH
        if ratio > 2:
            return Severity.MEDIUM
        return Severity.LOW
