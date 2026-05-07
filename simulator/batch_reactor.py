"""
Axion AI — Exothermic batch reactor simulator
==============================================

A second concrete process for the platform, alongside the pilot CSTR + column.
Models a single-vessel exothermic batch reactor with a cooling jacket and a
single first-order reaction A → P.

Equations
---------
Mass balance (constant volume, no inflow/outflow during batch):
    dC_A/dt = -k(T) · C_A
    dC_P/dt = +k(T) · C_A
    conversion = (C_A0 - C_A) / C_A0

Energy balance on the reactor:
    ρ·V·Cp · dT_R/dt = (-ΔH)·V·k(T)·C_A − UA·(T_R − T_J)

Energy balance on the jacket (perfectly mixed):
    ρ_c·V_c·Cp_c · dT_J/dt = ρ_c·F_c·Cp_c·(T_cool_in − T_J) + UA·(T_R − T_J)

Arrhenius kinetics:
    k(T) = k0 · exp(−Ea/(R·T_K))

The default parameters yield a moderately exothermic batch (peak ΔT_ad ≈ 35 °C
under adiabatic conditions). The integrator uses scipy.solve_ivp with RK45 —
no external dependencies beyond numpy/scipy that the project already uses.

Reusing the simulator
---------------------
The output column names live under the `batch.` prefix as declared in
`BATCH_PROFILE`. Output sample period is configurable (default 60 s, matching
the pilot CSV cadence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# ─────────────────────────────────────────────────────────────────────────────
# Default parameters (tunable per scenario)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchParams:
    # Reactor geometry
    V:        float = 1.0          # m³, reactor volume
    UA:       float = 8000.0       # W/K, overall heat-transfer coefficient × area
    rho:      float = 950.0        # kg/m³, reaction mixture density
    Cp:       float = 3500.0       # J/(kg·K)

    # Kinetics
    k0:       float = 7.0e7        # s⁻¹, Arrhenius pre-exponential
    Ea_R:     float = 8000.0       # K (Ea/R)
    dH_rxn:   float = -7.0e4       # J/mol (negative = exothermic)

    # Initial conditions
    C_A0:     float = 1500.0       # mol/m³
    T_R0:     float = 25.0         # °C
    T_J0:     float = 20.0         # °C

    # Cooling jacket
    V_c:      float = 0.15         # m³, jacket volume
    rho_c:    float = 1000.0       # kg/m³ (water-like)
    Cp_c:     float = 4180.0       # J/(kg·K)
    F_cool_default: float = 0.4    # m³/h, baseline coolant flow
    T_cool_in_default: float = 18.0  # °C

    # Run length
    duration_min: float = 240.0    # 4 hours
    sample_period_s: float = 60.0  # 1 sample/minute (matches pilot CSV cadence)

    # Disturbances (optional, set per scenario)
    F_cool_schedule: Optional[Dict[float, float]] = field(default=None)
    T_cool_in_schedule: Optional[Dict[float, float]] = field(default=None)
    """Both schedules are minute → value step functions."""


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────

def _step_value(schedule: Optional[Dict[float, float]],
                t_min: float, default: float) -> float:
    """Evaluate a step-function schedule {minute: value} at t_min."""
    if not schedule:
        return default
    keys = sorted(schedule.keys())
    val = default
    for k in keys:
        if t_min >= k:
            val = schedule[k]
        else:
            break
    return val


def simulate_batch(params: Optional[BatchParams] = None,
                   start_time: pd.Timestamp = pd.Timestamp("2026-01-01")) -> pd.DataFrame:
    """Run the batch simulator and return a DataFrame in canonical shape.

    Columns: timestamp, batch.T_R_C, batch.T_J_C, batch.T_cool_in_C,
             batch.C_A, batch.C_P, batch.conversion, batch.F_cool, batch.dHdt
    """
    p = params or BatchParams()

    # State: [C_A (mol/m3), T_R (K), T_J (K)]
    y0 = np.array([p.C_A0, p.T_R0 + 273.15, p.T_J0 + 273.15])
    t_end_s = p.duration_min * 60.0

    def rhs(t, y, F_cool, T_cool_in_K):
        C_A, T_R, T_J = y
        if C_A < 0: C_A = 0.0
        k_T = p.k0 * np.exp(-p.Ea_R / max(T_R, 1.0))
        r   = k_T * C_A                                              # mol/(m3·s)
        # Mass
        dC_A = -r
        # Energy — reactor
        Q_gen  = (-p.dH_rxn) * p.V * r                               # W
        Q_cool = p.UA * (T_R - T_J)                                  # W
        dT_R = (Q_gen - Q_cool) / (p.rho * p.V * p.Cp)
        # Energy — jacket
        F_c_m3s = F_cool / 3600.0
        m_dot = p.rho_c * F_c_m3s                                    # kg/s
        dT_J  = (m_dot * p.Cp_c * (T_cool_in_K - T_J) + Q_cool) / (
                p.rho_c * p.V_c * p.Cp_c)
        return [dC_A, dT_R, dT_J]

    # Solve in 1-minute chunks so disturbances can change between steps
    sample_period_s = p.sample_period_s
    n_steps = int(t_end_s // sample_period_s) + 1

    rows = []
    t = 0.0
    y = y0.copy()
    for i in range(n_steps):
        t_min = t / 60.0
        F_cool     = _step_value(p.F_cool_schedule,     t_min, p.F_cool_default)
        T_cool_in  = _step_value(p.T_cool_in_schedule,  t_min, p.T_cool_in_default)
        T_cool_in_K = T_cool_in + 273.15

        # Record current state
        C_A, T_R_K, T_J_K = y
        T_R_C = T_R_K - 273.15
        T_J_C = T_J_K - 273.15
        C_P   = max(0.0, p.C_A0 - C_A)
        conv  = C_P / p.C_A0 if p.C_A0 > 0 else 0.0
        k_T   = p.k0 * np.exp(-p.Ea_R / max(T_R_K, 1.0))
        dHdt  = (-p.dH_rxn) * p.V * k_T * max(C_A, 0.0) / 1000.0   # kW

        rows.append({
            "timestamp":         start_time + pd.Timedelta(seconds=t),
            "batch.T_R_C":       float(T_R_C),
            "batch.T_J_C":       float(T_J_C),
            "batch.T_cool_in_C": float(T_cool_in),
            "batch.C_A":         float(max(0.0, C_A)),
            "batch.C_P":         float(C_P),
            "batch.conversion":  float(conv),
            "batch.F_cool":      float(F_cool),
            "batch.dHdt":        float(dHdt),
        })

        # Advance one step
        if i == n_steps - 1:
            break
        sol = solve_ivp(
            rhs, (t, t + sample_period_s), y,
            args=(F_cool, T_cool_in_K),
            method="RK45", rtol=1e-6, atol=1e-8,
            dense_output=False, max_step=10.0,
        )
        y = sol.y[:, -1]
        t = t + sample_period_s

    return pd.DataFrame(rows)
