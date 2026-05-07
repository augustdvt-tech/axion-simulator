"""
Axion AI - Instrumentation and Data Logging
===========================================

- SensorModel: applies realistic sensor behavior to measurements
  (gaussian noise, bias, drift, and injected failures).
- DataLogger: buffers measurements and writes them to CSV.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Optional
import numpy as np


# =============================================================================
# Sensor Model
# =============================================================================

class SensorModel:
    """
    Applies realistic sensor behavior to measurements. Each tag can have its
    own configuration. Also supports injected failures (frozen, drift, bias).
    """

    # Default noise levels as fraction of reading (1% unless overridden)
    DEFAULT_NOISE = {
        "T_R_C":     {"sigma_abs": 0.15, "bias": 0.0},   # temp: ~0.15 degC noise
        "T_J_C":     {"sigma_abs": 0.15, "bias": 0.0},
        "T_feed_C":  {"sigma_abs": 0.10, "bias": 0.0},
        "T_cool_in_C": {"sigma_abs": 0.10, "bias": 0.0},
        "T_top_C":   {"sigma_abs": 0.20, "bias": 0.0},
        "T_bot_C":   {"sigma_abs": 0.20, "bias": 0.0},
        "F_feed":    {"sigma_rel": 0.01, "bias": 0.0},
        "F_cool":    {"sigma_rel": 0.015, "bias": 0.0},
        "F_vap_kgh": {"sigma_rel": 0.02, "bias": 0.0},
        "P_R":       {"sigma_abs": 0.02, "bias": 0.0},
        "C_A":       {"sigma_rel": 0.02, "bias": 0.0},
        "x_D":       {"sigma_abs": 0.003, "bias": 0.0},
        "x_B_A":     {"sigma_abs": 0.001, "bias": 0.0},
        "purity_B":  {"sigma_abs": 0.05, "bias": 0.0},
        "RR":        {"sigma_abs": 0.0, "bias": 0.0},     # setpoint, no noise
        "Q_reb_kW":  {"sigma_rel": 0.015, "bias": 0.0},
    }

    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)
        self.failures: Dict[str, Dict] = {}        # active failures by tag
        self._frozen_values: Dict[str, float] = {} # last value for frozen sensors

    def apply(self, tag: str, value: float, t: float) -> float:
        """Apply noise, bias, and injected failures to a measurement."""
        # Check for active failure on this tag (match by suffix)
        for failed_tag, fail_info in self.failures.items():
            if tag.endswith(failed_tag) or failed_tag.endswith(tag.split(".")[-1]):
                return self._apply_failure(tag, value, fail_info, t)

        # Normal noise
        key = tag.split(".")[-1]   # e.g. "cstr.T_R_C" -> "T_R_C"
        config = self.DEFAULT_NOISE.get(key, {"sigma_rel": 0.01, "bias": 0.0})
        bias = config.get("bias", 0.0)
        sigma_abs = config.get("sigma_abs", 0.0)
        sigma_rel = config.get("sigma_rel", 0.0)
        noise = self.rng.normal(0, sigma_abs + sigma_rel * abs(value))
        return value + bias + noise

    def inject_failure(self, tag: str, failure_type: str) -> None:
        """
        Inject a failure on a tag.
        Types: 'frozen' (stuck at last value), 'drift' (slow linear drift),
               'bias' (sudden offset), 'noise' (excessive noise).
        """
        self.failures[tag] = {"type": failure_type, "injected_at": None, "offset": 0.0}

    def clear_failure(self, tag: str) -> None:
        """Remove an active failure."""
        if tag in self.failures:
            del self.failures[tag]
        if tag in self._frozen_values:
            del self._frozen_values[tag]

    def _apply_failure(self, tag: str, value: float, info: Dict, t: float) -> float:
        if info["injected_at"] is None:
            info["injected_at"] = t
            self._frozen_values[tag] = value

        elapsed = t - info["injected_at"]
        ftype = info["type"]

        if ftype == "frozen":
            return self._frozen_values.get(tag, value)
        elif ftype == "drift":
            # 1%/hour drift
            drift = value * 0.01 * (elapsed / 3600.0)
            return value + drift
        elif ftype == "bias":
            if info["offset"] == 0.0:
                info["offset"] = value * 0.05   # +5% bias
            return value + info["offset"]
        elif ftype == "noise":
            return value + self.rng.normal(0, value * 0.05)
        return value


# =============================================================================
# Data Logger
# =============================================================================

class DataLogger:
    """Buffers samples and writes them to CSV."""

    def __init__(self, output_path: str, flush_every: int = 1000):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.flush_every = flush_every
        self.headers: Optional[list] = None
        self._file = None
        self._writer = None

    def log(self, row: Dict) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        if self._writer is None:
            # Initialize headers from first row
            self.headers = list(self.buffer[0].keys())
            self._file = open(self.output_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.headers,
                                          extrasaction="ignore")
            self._writer.writeheader()
        # Ensure consistent headers: collect any new keys from later rows
        for row in self.buffer:
            # Fill missing columns with empty
            out = {k: row.get(k, "") for k in self.headers}
            self._writer.writerow(out)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
