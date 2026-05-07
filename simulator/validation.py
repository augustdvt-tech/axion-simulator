"""
Axion AI - Simulator Validation: Analytical Correlations
=========================================================

Pure functions that encode published first-principles correlations for a
CSTR-column process. They return scalar quantities that can be compared
against simulation output to verify model correctness.

All functions are side-effect-free (no I/O, no simulation calls).

References
----------
CSTR:
  Fogler, H.S. "Elements of Chemical Reaction Engineering", 5th ed. §5.3
  van Heerden, C. Ind. Eng. Chem. 45:1242 (1953) — CSTR multiplicity

Distillation:
  Fenske, M.R. Ind. Eng. Chem. 24(9):482 (1932) — minimum stages
  Luyben, W.L. "Process Modeling, Simulation and Control for Chemical
      Engineers", 2nd ed. §8 — CMO tray model
  Smith, B.D. "Design of Equilibrium Stage Processes" — VLE, Kremser
"""

from __future__ import annotations

import numpy as np
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
# CSTR first-principles correlations
# ─────────────────────────────────────────────────────────────────────────────

def damkohler_number(
    k0: float,
    Ea_R: float,
    T_R_K: float,
    V: float,
    F_m3h: float,
) -> float:
    """Damköhler number for a 1st-order CSTR: Da = k(T) * tau.

    Parameters
    ----------
    k0    : pre-exponential factor [1/min]
    Ea_R  : activation energy / R [K]
    T_R_K : reactor temperature [K]
    V     : reactor volume [m³]
    F_m3h : volumetric feed flow [m³/h]

    Returns
    -------
    Da : dimensionless Damköhler number
    """
    tau_min = V / (F_m3h / 60.0)     # residence time [min]
    k = k0 * np.exp(-Ea_R / T_R_K)  # rate constant [1/min]
    return float(k * tau_min)


def cstr_da_conversion(da: float) -> float:
    """Theoretical CSTR steady-state conversion for 1st-order irreversible reaction.

    From the CSTR design equation at steady state:
        X_ss = Da / (1 + Da)

    Valid for: A → B, first-order, isothermal CSTR.
    For non-isothermal CSTRs this overestimates X if T differs from design T.
    """
    return float(da / (1.0 + da))


def adiabatic_temperature_rise(
    dH_J_mol: float,
    C_A0_mol_m3: float,
    rho_kg_m3: float,
    Cp_J_kgK: float,
) -> float:
    """Maximum possible temperature rise for complete adiabatic reaction.

    ΔT_ad = (−ΔH) · C_A0 / (ρ · Cp)   [K]

    With cooling active, the actual ΔT_R = T_R − T_feed must satisfy:
        ΔT_R < ΔT_ad  (cooling removes some of the heat of reaction)

    Parameters
    ----------
    dH_J_mol    : heat of reaction [J/mol], negative for exothermic
    C_A0_mol_m3 : inlet concentration [mol/m³]
    rho_kg_m3   : reactor liquid density [kg/m³]
    Cp_J_kgK    : reactor heat capacity [J/(kg·K)]
    """
    return float((-dH_J_mol) * C_A0_mol_m3 / (rho_kg_m3 * Cp_J_kgK))


def cstr_energy_balance_residual(
    C_A: float,
    T_R_K: float,
    T_J_K: float,
    params: Dict[str, float],
) -> float:
    """Fractional energy balance residual at a given operating point.

    At true steady state, Q_generated = Q_removed, so the residual → 0.
    For dynamic simulation data (not exactly at SS), values ≲ 0.15 are normal.

    Returns
    -------
    (Q_gen − Q_removed) / Q_gen — dimensionless.
    Positive → more heat generated than removed (still heating up).
    Negative → more heat removed than generated (cooling down).
    """
    k = params["k0"] * np.exp(-params["Ea_R"] / T_R_K)
    r = k * C_A                                       # mol/(m³·min)

    Q_gen = (-params["dH"]) * r * params["V"]         # J/min

    F_m3min = params["F"] / 60.0
    Q_flow   = F_m3min * params["rho"] * params["Cp"] * (T_R_K - params["T_feed"])
    Q_jacket = params["UA"] * (T_R_K - T_J_K)
    Q_removed = Q_flow + Q_jacket

    if abs(Q_gen) < 1e-6:
        return 0.0
    return float((Q_gen - Q_removed) / Q_gen)


def arrhenius_ratio(k0: float, Ea_R: float, T1_K: float, T2_K: float) -> float:
    """Rate constant ratio predicted by Arrhenius: k(T2) / k(T1).

    k(T2)/k(T1) = exp(−Ea/R · (1/T2 − 1/T1))
    """
    return float(np.exp(-Ea_R * (1.0 / T2_K - 1.0 / T1_K)))


# ─────────────────────────────────────────────────────────────────────────────
# Distillation column first-principles correlations
# ─────────────────────────────────────────────────────────────────────────────

def vle_y_from_x(x: float, alpha: float) -> float:
    """Binary VLE with constant relative volatility (modified Raoult's law).

    y* = α · x / (1 + (α − 1) · x)

    This is the exact equilibrium relationship assumed by the column model.
    Any stage composition (x_i, y_i) in the simulator must satisfy this.
    """
    return float(alpha * x / (1.0 + (alpha - 1.0) * x))


def fenske_minimum_stages(x_D: float, x_B_A: float, alpha: float) -> float:
    """Fenske equation: minimum theoretical stages at total reflux (1932).

    N_min = log[(x_D / (1−x_D)) · ((1−x_B_A) / x_B_A)] / log(α)

    Parameters
    ----------
    x_D   : light-component mole fraction in distillate
    x_B_A : light-component mole fraction in bottoms
    alpha : relative volatility (light / heavy)

    A feasible column requires N_actual > N_min. The larger the safety
    margin, the more resilient the column is to feed composition changes.
    """
    if x_D <= 0 or x_D >= 1 or x_B_A <= 0 or x_B_A >= 1:
        raise ValueError("Compositions must be strictly between 0 and 1.")
    sep = (x_D / (1.0 - x_D)) * ((1.0 - x_B_A) / x_B_A)
    return float(np.log(sep) / np.log(alpha))


def column_material_balance_error(
    z_F: float,
    x_D: float,
    x_B_A: float,
    D_over_F: float,
) -> float:
    """Absolute error in the overall light-component material balance.

    z_F · F = D · x_D + B · x_B
    Dividing by F: z_F = (D/F) · x_D + (1 − D/F) · x_B_A

    At steady state the CMO model enforces this internally, so the error
    should be near machine precision. In noisy simulation data ≲ 5e-3
    is expected due to the fluctuating D/F calculation.

    Parameters
    ----------
    z_F      : feed mole fraction of light component (= C_A / C_A0 from CSTR)
    x_D      : distillate light-component fraction (column.x_D)
    x_B_A    : bottoms light-component fraction (column.x_B_A)
    D_over_F : distillate / feed molar flow ratio
    """
    rhs = D_over_F * x_D + (1.0 - D_over_F) * x_B_A
    return float(abs(z_F - rhs))


def distillate_fraction(z_F: float, x_D: float, x_B_A: float) -> float:
    """Overall material balance solved for D/F.

    D/F = (z_F − x_B_A) / (x_D − x_B_A)
    """
    if abs(x_D - x_B_A) < 1e-6:
        return 0.5
    return float(np.clip((z_F - x_B_A) / (x_D - x_B_A), 0.05, 0.95))
