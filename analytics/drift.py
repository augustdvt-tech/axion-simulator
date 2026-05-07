"""
Axion AI — Distributional drift detection
==========================================

Detects when the live process stream has shifted away from the distribution
the soft sensor was trained on. Uses Population Stability Index (PSI), the
standard tool for monitoring feature drift in deployed ML models.

PSI per feature:
    PSI = Σ (p_live[i] − p_ref[i]) · ln(p_live[i] / p_ref[i])

over discrete bins fitted on the reference distribution. Conventional bands:
    PSI < 0.10        → no significant shift
    0.10 ≤ PSI < 0.25 → moderate shift (investigate)
    PSI ≥ 0.25        → significant shift (model likely out of domain)

The detector is intentionally simple — quantile bins, equal-width over the
quantile range. It deliberately does *not* try to detect concept drift
(p(y|x) shift); only covariate shift (p(x) shift) which is what causes the
soft sensor to extrapolate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Default PSI thresholds — match industry practice (Siddiqi 2006).
PSI_THRESHOLD_MODERATE     = 0.10
PSI_THRESHOLD_SIGNIFICANT  = 0.25

DEFAULT_N_BINS = 10
_EPSILON       = 1e-6   # avoid log(0) when a bin is empty in the live window


def classify_psi(psi: float) -> str:
    """Map a PSI value to one of: none / moderate / significant."""
    if psi >= PSI_THRESHOLD_SIGNIFICANT:
        return "significant"
    if psi >= PSI_THRESHOLD_MODERATE:
        return "moderate"
    return "none"


def quantile_bin_edges(values: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    """Compute n_bins+1 edges spanning the values via equal-frequency quantiles.

    Returns a strictly increasing array. If the reference is constant or has
    too few unique values, the returned edges may collapse to fewer bins —
    `compute_psi` handles that gracefully via _EPSILON.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        # degenerate — return two edges around the single value
        v = float(arr[0]) if arr.size == 1 else 0.0
        return np.array([v - 0.5, v + 0.5])
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(arr, quantiles)
    # Make strictly increasing — adjacent ties collapse a bin
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + _EPSILON
    return edges


def compute_psi(
    reference: np.ndarray,
    live: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    """Compute PSI between two 1D distributions using fixed bin edges.

    Either input may contain NaNs — they are dropped before binning.
    """
    ref = np.asarray(reference, dtype=float); ref = ref[~np.isnan(ref)]
    liv = np.asarray(live,      dtype=float); liv = liv[~np.isnan(liv)]
    if ref.size == 0 or liv.size == 0:
        return 0.0

    # Open-ended on both sides so out-of-range live values still bin
    edges = np.array(bin_edges, dtype=float).copy()
    edges[0]  = -np.inf
    edges[-1] =  np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    liv_counts, _ = np.histogram(liv, bins=edges)

    ref_pct = np.clip(ref_counts / max(1, ref.size), _EPSILON, None)
    liv_pct = np.clip(liv_counts / max(1, liv.size), _EPSILON, None)

    psi = float(np.sum((liv_pct - ref_pct) * np.log(liv_pct / ref_pct)))
    return max(0.0, psi)


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    status: str               # "none" | "moderate" | "significant"
    n_ref: int
    n_live: int


@dataclass
class DriftReport:
    overall_status: str       # worst feature's status
    max_psi: float            # worst PSI across features
    worst_feature: Optional[str]
    n_live: int
    by_feature: List[FeatureDrift] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "overall_status": self.overall_status,
            "max_psi":        self.max_psi,
            "worst_feature":  self.worst_feature,
            "n_live":         self.n_live,
            "by_feature": [
                {
                    "feature": f.feature,
                    "psi":     f.psi,
                    "status":  f.status,
                    "n_ref":   f.n_ref,
                    "n_live":  f.n_live,
                }
                for f in self.by_feature
            ],
        }


class DriftDetector:
    """PSI-based drift detector fitted on a reference distribution.

    Usage:
        detector = DriftDetector(features=PILOT_PURITY_FEATURES)
        detector.fit(reference_df)        # once, on training data
        report   = detector.score(live_df) # repeatedly, on a recent window
    """

    def __init__(self, features: List[str], n_bins: int = DEFAULT_N_BINS):
        self.features = list(features)
        self.n_bins   = n_bins
        self._edges:  Dict[str, np.ndarray] = {}
        self._ref_n:  Dict[str, int] = {}
        self._ref_values: Dict[str, np.ndarray] = {}

    @property
    def fitted(self) -> bool:
        return bool(self._edges)

    def fit(self, reference_df: pd.DataFrame) -> "DriftDetector":
        for f in self.features:
            if f not in reference_df.columns:
                continue
            arr = pd.to_numeric(reference_df[f], errors="coerce").to_numpy()
            arr = arr[~np.isnan(arr)]
            self._edges[f]      = quantile_bin_edges(arr, self.n_bins)
            self._ref_n[f]      = int(arr.size)
            self._ref_values[f] = arr
        return self

    def score(self, live_df: pd.DataFrame) -> DriftReport:
        if not self.fitted:
            raise RuntimeError("DriftDetector not fitted")

        rows: List[FeatureDrift] = []
        n_live = len(live_df)
        for f in self.features:
            if f not in self._edges:
                continue
            edges = self._edges[f]
            ref_n = self._ref_n[f]
            if f not in live_df.columns:
                rows.append(FeatureDrift(
                    feature=f, psi=0.0, status="none",
                    n_ref=ref_n, n_live=0,
                ))
                continue
            live_arr = pd.to_numeric(live_df[f], errors="coerce").to_numpy()
            live_arr = live_arr[~np.isnan(live_arr)]
            psi = compute_psi(self._ref_values[f], live_arr, edges)
            rows.append(FeatureDrift(
                feature=f, psi=psi, status=classify_psi(psi),
                n_ref=ref_n, n_live=int(live_arr.size),
            ))

        if not rows:
            return DriftReport(
                overall_status="none", max_psi=0.0,
                worst_feature=None, n_live=n_live,
            )

        worst = max(rows, key=lambda r: r.psi)
        return DriftReport(
            overall_status=worst.status,
            max_psi=float(worst.psi),
            worst_feature=worst.feature,
            n_live=n_live,
            by_feature=rows,
        )
