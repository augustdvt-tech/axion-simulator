"""Tests for simulator/units.py: CSTR and DistillationColumn."""

import numpy as np
import pytest
from simulator import CSTR, DistillationColumn, Stream


@pytest.fixture
def cstr():
    return CSTR(name="cstr", parameters=CSTR.DEFAULT_PARAMETERS)


@pytest.fixture
def column():
    return DistillationColumn(name="column",
                              parameters=DistillationColumn.DEFAULT_PARAMETERS)


@pytest.fixture
def cstr_outlet(cstr):
    feed = Stream("feed", flow=2.0, temperature=343.15,
                  composition={"A": 1.0, "B": 0.0}, pressure=2.0)
    outlet = Stream("cstr_out")
    cstr.compute_outlet(cstr.state, feed, outlet)
    return outlet


# ---------------------------------------------------------------------------
# CSTR
# ---------------------------------------------------------------------------

class TestCSTR:
    def test_state_variables(self, cstr):
        assert cstr.state_variables == ["C_A", "T_R", "T_J"]

    def test_initial_state_shape(self, cstr):
        assert cstr.state.shape == (3,)

    def test_initial_state_physical_range(self, cstr):
        C_A, T_R, T_J = cstr.state
        assert 0 < C_A < 2000          # mol/m3
        assert 300 < T_R < 420         # K  (27–147 °C)
        assert 280 < T_J < 380         # K

    def test_derivatives_returns_correct_shape(self, cstr):
        feed = Stream("f", flow=2.0, temperature=343.15,
                      composition={"A": 1.0}, pressure=2.0)
        dxdt = cstr.derivatives(0.0, cstr.state, feed)
        assert dxdt.shape == (3,)

    def test_rk4_step_changes_state(self, cstr):
        feed = Stream("f", flow=2.0, temperature=343.15,
                      composition={"A": 1.0}, pressure=2.0)
        state_before = cstr.state.copy()
        cstr.step_rk4(0.0, 10.0, feed)
        assert not np.allclose(cstr.state, state_before)

    def test_reset_parameters(self, cstr):
        cstr.parameters["F"] = 999.0
        cstr.reset_parameters()
        assert cstr.parameters["F"] == CSTR.DEFAULT_PARAMETERS["F"]

    def test_measured_variables_keys(self, cstr):
        feed = Stream("f", flow=2.0, temperature=343.15,
                      composition={"A": 1.0}, pressure=2.0)
        mv = cstr.measured_variables(cstr.state, feed)
        for key in ("T_R_C", "T_J_C", "C_A", "F_feed", "F_cool", "conversion"):
            assert key in mv

    def test_conversion_physical_range(self, cstr):
        feed = Stream("f", flow=2.0, temperature=343.15,
                      composition={"A": 1.0}, pressure=2.0)
        mv = cstr.measured_variables(cstr.state, feed)
        assert 0.0 <= mv["conversion"] <= 1.0

    def test_temperature_in_celsius(self, cstr):
        feed = Stream("f", flow=2.0, temperature=343.15,
                      composition={"A": 1.0}, pressure=2.0)
        mv = cstr.measured_variables(cstr.state, feed)
        # Kelvin initial state → Celsius should be around 70–120 °C at start
        assert 0 < mv["T_R_C"] < 200

    def test_compute_outlet_sets_composition(self, cstr):
        feed = Stream("f", flow=2.0, temperature=343.15,
                      composition={"A": 1.0, "B": 0.0}, pressure=2.0)
        out = Stream("out")
        cstr.compute_outlet(cstr.state, feed, out)
        assert "A" in out.composition
        assert 0.0 <= out.composition.get("A", 0) <= 1.0

    def test_higher_cooling_reduces_jacket_temp(self):
        """Increasing F_cool should keep T_J lower under same reaction conditions."""
        def final_T_J(F_c):
            p = {**CSTR.DEFAULT_PARAMETERS, "F_c": F_c}
            c = CSTR(name="c", parameters=p)
            feed = Stream("f", flow=2.0, temperature=343.15,
                          composition={"A": 1.0}, pressure=2.0)
            for _ in range(600):   # 6000 s warm-up
                c.step_rk4(0.0, 10.0, feed)
            return c.state[2]   # T_J

        T_J_low_cool  = final_T_J(0.15)
        T_J_high_cool = final_T_J(0.50)
        assert T_J_high_cool < T_J_low_cool


# ---------------------------------------------------------------------------
# DistillationColumn
# ---------------------------------------------------------------------------

class TestDistillationColumn:
    def test_initial_state_shape(self, column):
        n = column.parameters["N_stages"] + 2
        assert column.state.shape == (n,)

    def test_initial_state_range(self, column):
        # Mole fractions of A: descending from top (~0.95) to bottom (~0.02)
        assert column.state[0] > column.state[-1]
        assert 0.0 <= column.state.min()
        assert column.state.max() <= 1.0

    def test_state_variables_length(self, column):
        assert len(column.state_variables) == len(column.state)

    def test_derivatives_shape(self, column, cstr_outlet):
        dxdt = column.derivatives(0.0, column.state, cstr_outlet)
        assert dxdt.shape == column.state.shape

    def test_rk4_step_changes_state(self, column, cstr_outlet):
        before = column.state.copy()
        column.step_rk4(0.0, 10.0, cstr_outlet)
        assert not np.allclose(column.state, before)

    def test_measured_variables_keys(self, column, cstr_outlet):
        mv = column.measured_variables(column.state, cstr_outlet)
        for key in ("purity_B", "T_top_C", "T_bot_C", "RR", "Q_reb_kW"):
            assert key in mv

    def test_purity_physical_range(self, column, cstr_outlet):
        mv = column.measured_variables(column.state, cstr_outlet)
        assert 0.0 <= mv["purity_B"] <= 100.0

    def test_reset_parameters(self, column):
        original_rr = column.parameters["RR"]
        column.parameters["RR"] = 99.0
        column.reset_parameters()
        assert column.parameters["RR"] == original_rr

    def test_higher_rr_increases_purity(self):
        """Increasing RR should increase purity_B (law of distillation)."""
        def purity_at_rr(rr):
            p = {**DistillationColumn.DEFAULT_PARAMETERS, "RR": rr}
            col = DistillationColumn(name="col", parameters=p)
            feed = Stream("f", flow=2.0, temperature=343.15,
                          composition={"A": 0.15, "B": 0.85}, pressure=2.0)
            for _ in range(3000):
                col.step_rk4(0.0, 10.0, feed)
            mv = col.measured_variables(col.state, feed)
            return mv["purity_B"]

        assert purity_at_rr(6.0) > purity_at_rr(4.0)
