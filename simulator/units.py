"""
Axion AI - Process Units
========================

Implementations of the process units for the MVP pilot process:
- CSTR: Continuous Stirred Tank Reactor with cooling jacket (first-order
  exothermic reaction A -> B). First-principles ODE model.
- DistillationColumn: simplified dynamic tray-by-tray binary column model
  with constant molar overflow. Separates unreacted A (volatile) from
  product B (heavy) in the reactor effluent.

Physical constants and typical parameter ranges are documented inline so that
engineers from other industries can adapt the parameters to their own process.

NOTE (MVP): the distillation column uses a simplified equilibrium model (constant
relative volatility). For the production version this will be replaced with a
more rigorous thermodynamic model (e.g., Antoine + activity coefficients).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List
from .core import ProcessUnit, Stream


# =============================================================================
# CSTR: Continuous Stirred Tank Reactor
# =============================================================================

class CSTR(ProcessUnit):
    """
    Continuous Stirred Tank Reactor with cooling jacket.

    Reaction:  A -> B  (first-order, exothermic)
    Kinetics:  r = k0 * exp(-Ea/(R*T)) * C_A

    State variables (x):
        x[0] = C_A   concentration of A in reactor     [mol/m3]
        x[1] = T_R   reactor temperature                [K]
        x[2] = T_J   jacket temperature                 [K]

    The reactor volume is assumed constant (level control is perfect).
    Level dynamics can be added later by including V as a state variable.
    """

    DEFAULT_PARAMETERS = {
        # Reactor geometry
        "V":     10.0,         # reactor volume [m3]
        "V_j":   2.0,          # jacket volume [m3]

        # Reaction kinetics (calibrated for ~85% conversion at 353 K, tau=5h)
        "k0":    1.1e9,        # pre-exponential factor [1/min]
        "Ea_R":  8750.0,       # activation energy / R [K]
        "dH":   -50000.0,      # heat of reaction [J/mol]  (negative = exothermic)

        # Thermodynamic properties — organic solvent (typical for petrochemical
        # or fine chemistry processes). Water-like coolant in jacket.
        "rho":   800.0,        # reactor density [kg/m3]
        "Cp":    2000.0,       # reactor heat capacity [J/(kg*K)]
        "rho_c": 1000.0,       # coolant density (water)
        "Cp_c":  4184.0,       # coolant heat capacity (water)

        # Heat transfer (calibrated from SS energy balance)
        "UA":    4.5e4,        # overall heat transfer coeff * area [J/(min*K)]

        # Feed conditions (nominal). Feed preheated upstream, common in industry.
        "C_A0":  1000.0,       # inlet A concentration [mol/m3]
        "T_feed": 343.15,      # feed temperature [K] — preheated to 70 degC
        "T_c_in": 288.15,      # coolant inlet temperature [K] — 15 degC

        # Manipulated variables (initial setpoints, calibrated for design point)
        "F":     2.0,          # feed flow [m3/h]
        "F_c":   0.30,         # coolant flow [m3/h]
    }

    @property
    def state_variables(self) -> List[str]:
        return ["C_A", "T_R", "T_J"]

    def initial_state(self) -> np.ndarray:
        # Reasonable steady-state guess (will be corrected by warm-up)
        return np.array([150.0, 353.15, 313.15])

    def derivatives(self, t: float, x: np.ndarray, inlet: Stream) -> np.ndarray:
        p = self.parameters
        C_A, T_R, T_J = x

        # Convert volumetric flows from m3/h to m3/min (we work in /min internally)
        F   = p["F"] / 60.0
        F_c = p["F_c"] / 60.0

        # Reaction rate
        k = p["k0"] * np.exp(-p["Ea_R"] / T_R)
        r = k * C_A                          # mol/(m3*min)

        # --- Mass balance on A ---
        dC_A_dt = (F / p["V"]) * (p["C_A0"] - C_A) - r

        # --- Energy balance on reactor ---
        # dT_R/dt = F/V * (T_feed - T_R) + (-dH)*r / (rho*Cp) - UA*(T_R - T_J)/(V*rho*Cp)
        dT_R_dt = (
            (F / p["V"]) * (p["T_feed"] - T_R)
            + (-p["dH"]) * r / (p["rho"] * p["Cp"])
            - p["UA"] * (T_R - T_J) / (p["V"] * p["rho"] * p["Cp"])
        )

        # --- Energy balance on jacket ---
        # dT_J/dt = F_c/V_j * (T_c_in - T_J) + UA*(T_R - T_J)/(V_j*rho_c*Cp_c)
        dT_J_dt = (
            (F_c / p["V_j"]) * (p["T_c_in"] - T_J)
            + p["UA"] * (T_R - T_J) / (p["V_j"] * p["rho_c"] * p["Cp_c"])
        )

        # Convert from /min to /s for the integrator
        return np.array([dC_A_dt, dT_R_dt, dT_J_dt]) / 60.0

    def compute_outlet(self, x: np.ndarray, inlet: Stream, outlet: Stream) -> None:
        """Propagate CSTR state to the outlet stream."""
        p = self.parameters
        C_A, T_R, T_J = x

        # Total molar concentration assumed constant for binary A+B
        C_total = p["C_A0"]
        C_B = max(0.0, C_total - C_A)
        x_A = C_A / C_total
        x_B = 1.0 - x_A

        outlet.flow = p["F"]
        outlet.temperature = T_R
        outlet.composition = {"A": x_A, "B": x_B}
        outlet.pressure = 2.0   # assumed constant for MVP

    def measured_variables(self, x: np.ndarray, inlet: Stream) -> Dict[str, float]:
        p = self.parameters
        C_A, T_R, T_J = x
        return {
            "T_R_C":     T_R - 273.15,        # reactor temp in degC
            "T_J_C":     T_J - 273.15,        # jacket temp in degC
            "C_A":       C_A,                  # mol/m3
            "F_feed":    p["F"],               # m3/h
            "F_cool":    p["F_c"],             # m3/h
            "T_feed_C":  p["T_feed"] - 273.15,
            "T_cool_in_C": p["T_c_in"] - 273.15,
            "P_R":       2.0,                  # bar (constant for MVP)
            "conversion": 1.0 - C_A / p["C_A0"],
        }


# =============================================================================
# DistillationColumn: binary column with CMO assumption
# =============================================================================

class DistillationColumn(ProcessUnit):
    """
    Binary distillation column with constant molar overflow (CMO) assumption.

    Configuration:
        Stage 0         = total condenser
        Stages 1..N_r   = rectifying section (above feed)
        Stage N_r+1     = feed stage
        Stages N_r+2..N = stripping section (below feed)
        Stage N+1       = partial reboiler

    State variable per stage: liquid mole fraction of light component A.
    Vapor composition via relative volatility: y = alpha*x / (1 + (alpha-1)*x).

    Component balances per stage (CMO, no energy balance):
        dM*dx_i/dt = L*(x_{i-1} - x_i) + V*(y_{i+1} - y_i)
    with adjustments at feed / reboiler / condenser stages.

    This is a standard simplified model used in process control education
    (e.g. Luyben textbook). Adequate for dynamic simulation and control studies.
    """

    DEFAULT_PARAMETERS = {
        "N_stages":   15,      # total theoretical stages (excluding reboiler)
        "feed_stage": 8,       # feed location (0-indexed from top condenser)
        "alpha":      3.2,     # relative volatility A/B (A = lighter)
        # Industrial-scale holdups: ~30-60s residence per tray, 5min drums
        "M_tray":     500.0,   # liquid holdup per tray [mol]
        "M_cond":     3000.0,  # condenser holdup [mol]
        "M_reb":      3000.0,  # reboiler holdup [mol]

        # Manipulated variables
        "RR":         5.5,     # reflux ratio L/D (nominal, for 98.5% purity target)
        "B_frac":     None,    # bottoms fraction of feed (auto if None)

        # Operating pressures
        "P_top":      1.02,    # bar
        "P_bot":      1.15,    # bar

        # Energy estimates (for consumption tracking)
        "H_vap":      35000.0, # heat of vaporization [J/mol]
    }

    @property
    def state_variables(self) -> List[str]:
        n = self.parameters["N_stages"] + 1  # stages + reboiler
        return [f"x_{i}" for i in range(n + 1)]   # +1 for condenser drum

    def initial_state(self) -> np.ndarray:
        """Linear profile as initial guess: pure A at top, pure B at bottom."""
        n = self.parameters["N_stages"] + 2  # condenser + stages + reboiler
        return np.linspace(0.95, 0.02, n)

    def derivatives(self, t: float, x: np.ndarray, inlet: Stream) -> np.ndarray:
        p = self.parameters
        N = p["N_stages"]
        fs = p["feed_stage"]
        alpha = p["alpha"]

        # Clip compositions to physical bounds for numerical robustness
        x = np.clip(x, 1e-6, 1.0 - 1e-6)

        # Feed properties from inlet stream (from CSTR)
        F_mol = self._volumetric_to_molar_flow(inlet.flow)   # mol/min
        z_F   = inlet.composition.get("A", 0.15)             # mole fraction A in feed
        z_F   = float(np.clip(z_F, 1e-6, 1.0 - 1e-6))

        # Internal flows (CMO assumption, feed as saturated liquid)
        RR = p["RR"]
        # Component balance on overall column:
        # Material balance: F = D + B
        # Key component balance: F*z_F = D*x_D + B*x_B
        # We don't know D and B a priori. For CMO with saturated liquid feed:
        # L_rect = RR * D, V_rect = (RR+1) * D
        # L_strip = L_rect + F, V_strip = V_rect
        # Need to determine D. Use current state: assume D is fraction of feed that is "light"
        # For MVP: use xD and xB from state to compute D consistently
        x_D = x[0]                # top composition
        x_B = x[-1]               # bottom composition
        if abs(x_D - x_B) < 1e-3:
            D_mol = 0.5 * F_mol
        else:
            D_mol = F_mol * (z_F - x_B) / (x_D - x_B)
            D_mol = np.clip(D_mol, 0.05 * F_mol, 0.95 * F_mol)
        B_mol = F_mol - D_mol

        L_rect = RR * D_mol
        V_col  = (RR + 1.0) * D_mol     # vapor up the whole column
        L_strip = L_rect + F_mol        # adds feed liquid

        # Vapor compositions from equilibrium
        def y_eq(xi):
            return alpha * xi / (1.0 + (alpha - 1.0) * xi)

        y = y_eq(x)

        n_total = N + 2    # condenser(0) + N stages + reboiler(N+1)
        dx = np.zeros(n_total)

        # --- Condenser (total, stage 0) ---
        # dM*dx0/dt = V*(y1 - x0)   (total condenser: y_in condenses to x_D = x_0)
        dx[0] = V_col * (y[1] - x[0]) / p["M_cond"]

        # --- Trays 1..N ---
        for i in range(1, N + 1):
            # Determine L and V above/below this stage
            if i <= fs:
                L_above = L_rect
            else:
                L_above = L_strip
            if (i + 1) <= fs:
                L_below = L_rect
            else:
                L_below = L_strip
            # Feed enters at stage fs
            feed_in = F_mol * z_F if i == fs else 0.0
            feed_flow = F_mol if i == fs else 0.0

            inflow_liq_above = L_above * x[i - 1]
            outflow_liq_below = L_below * x[i]
            inflow_vap_below = V_col * y[i + 1]
            outflow_vap_up   = V_col * y[i]

            dx[i] = (inflow_liq_above - outflow_liq_below +
                     inflow_vap_below - outflow_vap_up +
                     feed_in) / p["M_tray"]

        # --- Reboiler (partial, stage N+1) ---
        # Liquid in from tray N, vapor out, bottoms out
        inflow_liq = L_strip * x[N]
        vap_out    = V_col * y[N + 1]
        bot_out    = B_mol * x[N + 1]
        dx[N + 1] = (inflow_liq - vap_out - bot_out) / p["M_reb"]

        # Convert /min -> /s
        return dx / 60.0

    def compute_outlet(self, x: np.ndarray, inlet: Stream, outlet: Stream) -> None:
        """The column has two products; outlet stream carries the bottoms (product B)."""
        p = self.parameters
        x_D = x[0]
        x_B = x[-1]
        F_mol = self._volumetric_to_molar_flow(inlet.flow)
        z_F = inlet.composition.get("A", 0.15)

        if abs(x_D - x_B) < 1e-3:
            D_mol = 0.5 * F_mol
        else:
            D_mol = F_mol * (z_F - x_B) / (x_D - x_B)
            D_mol = np.clip(D_mol, 0.05 * F_mol, 0.95 * F_mol)
        B_mol = F_mol - D_mol

        outlet.flow = self._molar_to_volumetric_flow(B_mol)
        outlet.temperature = 390.0  # bottoms boiling point approximation
        outlet.composition = {"A": float(x_B), "B": float(1 - x_B)}
        outlet.pressure = p["P_bot"]

    def measured_variables(self, x: np.ndarray, inlet: Stream) -> Dict[str, float]:
        p = self.parameters
        x_D = float(x[0])
        x_B = float(x[-1])
        RR = p["RR"]

        # Compute flows for KPIs
        F_mol = self._volumetric_to_molar_flow(inlet.flow)
        z_F = inlet.composition.get("A", 0.15)
        if abs(x_D - x_B) < 1e-3:
            D_mol = 0.5 * F_mol
        else:
            D_mol = F_mol * (z_F - x_B) / (x_D - x_B)
            D_mol = np.clip(D_mol, 0.05 * F_mol, 0.95 * F_mol)
        V_col = (RR + 1.0) * D_mol

        # Reboiler duty: Q_reb = V * H_vap
        Q_reb_W = V_col * p["H_vap"] / 60.0   # J/s = W
        F_vap_kg_h = V_col * 0.080 * 60.0     # approx MW ~ 80 g/mol, kg/h

        # Temperatures (very simplified from composition)
        T_top_C = 70.0 + (1 - x_D) * 40.0
        T_bot_C = 118.0 - x_B * 40.0

        return {
            "x_D":       x_D,                        # top composition (A fraction)
            "x_B_A":     x_B,                        # bottom composition (A fraction)
            "purity_B":  (1.0 - x_B) * 100.0,        # product purity [%]
            "T_top_C":   T_top_C,
            "T_bot_C":   T_bot_C,
            "RR":        RR,
            "F_vap_kgh": F_vap_kg_h,
            "Q_reb_kW":  Q_reb_W / 1000.0,
            "P_top_bar": p["P_top"],
            "P_bot_bar": p["P_bot"],
        }

    # --- helpers ---
    @staticmethod
    def _volumetric_to_molar_flow(flow_m3_h: float) -> float:
        """Approximate conversion: ~1000 kg/m3 density, ~80 g/mol average MW."""
        # kg/h -> mol/h: (flow * rho) / MW
        rho = 1000.0    # kg/m3
        MW  = 80.0      # g/mol ~ average
        mol_per_h = flow_m3_h * rho * 1000.0 / MW   # mol/h
        return mol_per_h / 60.0   # mol/min

    @staticmethod
    def _molar_to_volumetric_flow(mol_per_min: float) -> float:
        rho = 1000.0
        MW  = 80.0
        kg_per_min = mol_per_min * MW / 1000.0
        m3_per_min = kg_per_min / rho
        return m3_per_min * 60.0   # m3/h
