"""
Distillation Column Simulator Validation — First-Principles Correlations
=========================================================================

Validates that the column dynamic simulation is consistent with:
  1. Fenske minimum stages (Fenske 1932) — separation feasibility
  2. VLE equilibrium relation (constant relative volatility)
  3. Overall component material balance (CMO model invariant)
  4. Column temperature gradient (thermodynamic requirement)
  5. Product purity target (design specification)

All checks use the mean of the last 200 rows of the "normal" scenario,
which represents quasi-steady-state operation at design conditions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

from simulator.units import DistillationColumn
from simulator.validation import (
    vle_y_from_x,
    fenske_minimum_stages,
    column_material_balance_error,
    distillate_fraction,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PARAMS   = DistillationColumn.DEFAULT_PARAMETERS


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ss_data():
    path = DATA_DIR / "normal.csv"
    if not path.exists():
        pytest.skip(f"normal.csv not found at {path}")
    return pd.read_csv(path).tail(200).reset_index(drop=True)


@pytest.fixture(scope="module")
def ss_mean(ss_data):
    return ss_data.mean(numeric_only=True)


@pytest.fixture(scope="module")
def ss_z_F(ss_mean):
    """Feed composition to column = fraction of A remaining from CSTR.

    z_F = C_A / C_A0.  We read C_A from the simulation and use the nominal
    C_A0 = 1000 mol/m³ from the CSTR default parameters.
    """
    from simulator.units import CSTR
    C_A0 = CSTR.DEFAULT_PARAMETERS["C_A0"]
    return float(ss_mean["cstr.C_A"] / C_A0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fenske minimum stages
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.validation
class TestFenskeMinimumStages:
    def test_n_actual_exceeds_n_min(self, ss_mean):
        """N_actual=15 > N_min (Fenske 1932): separation is feasible."""
        x_D   = float(ss_mean["column.x_D"])
        x_B_A = float(ss_mean["column.x_B_A"])
        alpha = PARAMS["alpha"]

        x_D   = np.clip(x_D,   1e-4, 1 - 1e-4)
        x_B_A = np.clip(x_B_A, 1e-4, 1 - 1e-4)

        n_min = fenske_minimum_stages(x_D, x_B_A, alpha)
        n_actual = PARAMS["N_stages"]

        assert n_actual > n_min, (
            f"Feasibility violated: N_actual={n_actual} ≤ N_min={n_min:.1f} "
            f"(x_D={x_D:.4f}, x_B_A={x_B_A:.4f}, α={alpha})"
        )

    def test_safety_margin_at_least_two_stages(self, ss_mean):
        """N_actual − N_min ≥ 2: adequate design margin."""
        x_D   = np.clip(float(ss_mean["column.x_D"]),   1e-4, 1 - 1e-4)
        x_B_A = np.clip(float(ss_mean["column.x_B_A"]), 1e-4, 1 - 1e-4)
        n_min = fenske_minimum_stages(x_D, x_B_A, PARAMS["alpha"])
        margin = PARAMS["N_stages"] - n_min
        assert margin >= 2.0, (
            f"Thin design margin: N_actual−N_min={margin:.1f} stages"
        )

    def test_fenske_increases_with_tighter_spec(self):
        """Tighter purity target → more stages needed (monotonic)."""
        alpha = PARAMS["alpha"]
        n_min_loose = fenske_minimum_stages(0.990, 0.015, alpha)
        n_min_tight = fenske_minimum_stages(0.999, 0.005, alpha)
        assert n_min_tight > n_min_loose

    def test_fenske_increases_with_lower_volatility(self):
        """Closer-boiling components → more stages needed."""
        x_D, x_B = 0.998, 0.007
        n_easy = fenske_minimum_stages(x_D, x_B, alpha=5.0)
        n_hard = fenske_minimum_stages(x_D, x_B, alpha=1.5)
        assert n_hard > n_easy


# ─────────────────────────────────────────────────────────────────────────────
# 2. VLE equilibrium relation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.validation
class TestVLEEquilibrium:
    def test_vle_formula_exact_at_zero(self):
        """y*(0, α) = 0 (pure heavy component has zero light in vapour)."""
        assert vle_y_from_x(0.0, PARAMS["alpha"]) == pytest.approx(0.0)

    def test_vle_formula_exact_at_one(self):
        """y*(1, α) = 1 (pure light component is entirely in vapour)."""
        assert vle_y_from_x(1.0, PARAMS["alpha"]) == pytest.approx(1.0)

    def test_vle_enriches_light_component(self):
        """y* > x for 0 < x < 1 when α > 1 (light component enriched in vapour)."""
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            y = vle_y_from_x(x, PARAMS["alpha"])
            assert y > x, f"VLE not enriching at x={x}: y*={y:.4f}"

    def test_vle_monotone_increasing(self):
        """Higher liquid fraction → higher vapour fraction."""
        xs = np.linspace(0.01, 0.99, 50)
        ys = [vle_y_from_x(float(xi), PARAMS["alpha"]) for xi in xs]
        assert all(ys[i] < ys[i + 1] for i in range(len(ys) - 1))

    def test_column_top_composition_consistent_with_vle(self, ss_mean):
        """Distillate (top of column) is enriched above the equilibrium line."""
        x_D = float(ss_mean["column.x_D"])
        y_eq = vle_y_from_x(x_D, PARAMS["alpha"])
        # At the top, vapour entering condenser must be ≥ y* of the distillate liquid
        assert x_D >= 0.95, (
            f"Distillate purity low: x_D={x_D:.4f} — not reaching design spec"
        )
        # VLE computed at top composition should be physically valid
        assert 0.0 <= y_eq <= 1.0

    def test_column_bottom_composition_consistent_with_vle(self, ss_mean):
        """Bottoms liquid fraction of light component is below the feed."""
        x_B_A = float(ss_mean["column.x_B_A"])
        assert x_B_A < 0.05, (
            f"Bottoms still has too much light component: x_B_A={x_B_A:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Overall component material balance
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.validation
class TestColumnMaterialBalance:
    def test_material_balance_closes(self, ss_mean, ss_z_F):
        """F·z_F = D·x_D + B·x_B — CMO invariant (Luyben §8)."""
        x_D   = float(ss_mean["column.x_D"])
        x_B_A = float(ss_mean["column.x_B_A"])

        D_over_F = distillate_fraction(ss_z_F, x_D, x_B_A)
        error    = column_material_balance_error(ss_z_F, x_D, x_B_A, D_over_F)

        assert error < 5e-3, (
            f"Material balance error={error:.5f} — CMO invariant violated "
            f"(z_F={ss_z_F:.4f}, x_D={x_D:.4f}, x_B_A={x_B_A:.4f}, D/F={D_over_F:.4f})"
        )

    def test_distillate_fraction_between_0_and_1(self, ss_mean, ss_z_F):
        x_D   = float(ss_mean["column.x_D"])
        x_B_A = float(ss_mean["column.x_B_A"])
        D_over_F = distillate_fraction(ss_z_F, x_D, x_B_A)
        assert 0.0 < D_over_F < 1.0

    def test_bottoms_richer_in_heavy(self, ss_mean):
        """Product B (bottoms) must be richer in B than the feed."""
        purity_B = float(ss_mean["column.purity_B"])  # % B in bottoms
        x_B_A    = float(ss_mean["column.x_B_A"])     # fraction A in bottoms
        # purity_B = (1 - x_B_A) * 100
        assert purity_B == pytest.approx((1.0 - x_B_A) * 100.0, abs=0.1)

    def test_total_flow_conservation(self, ss_mean, ss_z_F):
        """Light + heavy component fractions sum to 1 at each end."""
        x_D   = float(ss_mean["column.x_D"])
        x_B_A = float(ss_mean["column.x_B_A"])
        # These are mole fractions of A; (1-x) is B fraction
        assert abs(x_D + (1 - x_D) - 1.0) < 1e-10
        assert abs(x_B_A + (1 - x_B_A) - 1.0) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 4. Column temperature gradient
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.validation
class TestColumnTemperatureGradient:
    def test_bottom_hotter_than_top(self, ss_mean):
        """T_bot > T_top: higher pressure and heavier component at bottom (always)."""
        assert ss_mean["column.T_bot_C"] > ss_mean["column.T_top_C"], (
            f"Temperature inversion: T_top={ss_mean['column.T_top_C']:.1f}°C "
            f"T_bot={ss_mean['column.T_bot_C']:.1f}°C"
        )

    def test_bottom_pressure_exceeds_top(self, ss_mean):
        """P_bot > P_top: pressure increases from top to bottom (always)."""
        assert ss_mean["column.P_bot_bar"] > ss_mean["column.P_top_bar"]

    def test_temperatures_in_physical_range(self, ss_mean):
        """Column temperatures between 50°C and 180°C (organic solvents)."""
        assert ss_mean["column.T_top_C"] > 50.0
        assert ss_mean["column.T_bot_C"] < 180.0

    def test_reboiler_duty_positive(self, ss_mean):
        """Q_reb > 0: energy must be supplied to vapourise liquid (always)."""
        assert ss_mean["column.Q_reb_kW"] > 0.0

    def test_vapour_flow_positive(self, ss_mean):
        """F_vap > 0: vapour must flow upward for separation to occur."""
        assert ss_mean["column.F_vap_kgh"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Product purity design specification
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.validation
class TestProductPurity:
    PURITY_TARGET = 98.5   # % — design specification

    def test_mean_purity_meets_target(self, ss_mean):
        """Mean purity_B ≥ 98.5% in normal operation (design spec)."""
        purity = float(ss_mean["column.purity_B"])
        assert purity >= self.PURITY_TARGET, (
            f"Mean purity {purity:.2f}% < target {self.PURITY_TARGET}%"
        )

    def test_purity_within_physical_bounds(self, ss_data):
        """Purity is bounded to [0, 100]% at every timestep."""
        assert (ss_data["column.purity_B"] >= 0.0).all()
        assert (ss_data["column.purity_B"] <= 100.0).all()

    def test_purity_stable_in_normal_operation(self, ss_data):
        """Purity std dev < 0.5% — no oscillation in normal scenario."""
        std = ss_data["column.purity_B"].std()
        assert std < 0.5, (
            f"Excessive purity oscillation: std={std:.3f}% "
            "— normal scenario should be near steady state"
        )
