"""Tests for optimizer/surrogate.py: ProcessSurrogate."""

import pytest
import pandas as pd
from optimizer import ProcessSurrogate, SURROGATE_INPUTS, SURROGATE_OUTPUTS


@pytest.fixture
def surrogate():
    return ProcessSurrogate()


class TestProcessSurrogateInit:
    def test_inputs_defined(self, surrogate):
        assert len(surrogate.inputs) > 0

    def test_outputs_defined(self, surrogate):
        assert len(surrogate.outputs) > 0

    def test_known_inputs(self, surrogate):
        for inp in ("column.RR", "cstr.F_cool", "cstr.F_feed"):
            assert inp in surrogate.inputs

    def test_known_outputs(self, surrogate):
        for out in ("column.purity_B", "column.Q_reb_kW"):
            assert out in surrogate.outputs


class TestProcessSurrogatePredictOne:
    def test_predict_one_returns_kpis(self, surrogate):
        kpis = surrogate.predict_one(**{
            "column.RR": 5.5,
            "cstr.F_cool": 0.30,
            "cstr.F_feed": 2.0,
            "cstr.C_A": 172.0,
            "cstr.T_feed_C": 70.0,
        })
        assert "column.purity_B" in kpis
        assert "column.Q_reb_kW" in kpis

    def test_purity_physical_range(self, surrogate):
        kpis = surrogate.predict_one(**{
            "column.RR": 5.5,
            "cstr.F_cool": 0.30,
            "cstr.F_feed": 2.0,
            "cstr.C_A": 172.0,
            "cstr.T_feed_C": 70.0,
        })
        assert 80.0 <= kpis["column.purity_B"] <= 100.0

    def test_higher_rr_higher_purity(self, surrogate):
        base = {"cstr.F_cool": 0.30, "cstr.F_feed": 2.0,
                "cstr.C_A": 172.0, "cstr.T_feed_C": 70.0}
        kpis_low  = surrogate.predict_one(**{"column.RR": 4.0, **base})
        kpis_high = surrogate.predict_one(**{"column.RR": 7.0, **base})
        assert kpis_high["column.purity_B"] > kpis_low["column.purity_B"]

    def test_higher_rr_higher_energy(self, surrogate):
        base = {"cstr.F_cool": 0.30, "cstr.F_feed": 2.0,
                "cstr.C_A": 172.0, "cstr.T_feed_C": 70.0}
        kpis_low  = surrogate.predict_one(**{"column.RR": 4.0, **base})
        kpis_high = surrogate.predict_one(**{"column.RR": 7.0, **base})
        assert kpis_high["column.Q_reb_kW"] > kpis_low["column.Q_reb_kW"]


class TestProcessSurrogatePredict:
    def test_predict_batch(self, surrogate):
        df = pd.DataFrame([
            {"column.RR": rr, "cstr.F_cool": 0.30, "cstr.F_feed": 2.0,
             "cstr.C_A": 172.0, "cstr.T_feed_C": 70.0}
            for rr in [4.0, 5.5, 7.0]
        ])
        result = surrogate.predict(df)
        assert len(result) == 3
        assert "column.purity_B" in result.columns

    def test_predict_monotone_rr(self, surrogate):
        rr_values = [3.5, 4.5, 5.5, 6.5, 7.5]
        df = pd.DataFrame([
            {"column.RR": rr, "cstr.F_cool": 0.30, "cstr.F_feed": 2.0,
             "cstr.C_A": 172.0, "cstr.T_feed_C": 70.0}
            for rr in rr_values
        ])
        result = surrogate.predict(df)
        purities = result["column.purity_B"].tolist()
        # purity should increase monotonically with RR
        assert all(purities[i] <= purities[i+1] for i in range(len(purities)-1))
