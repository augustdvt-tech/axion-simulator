"""Tests for soft_sensor/purity.py: PuritySoftSensor (slow — trains GBR ensemble)."""

import numpy as np
import pandas as pd
import pytest
from soft_sensor import PuritySoftSensor, PILOT_PURITY_FEATURES, TARGET_PURITY


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained_sensor(df_normal_csv):
    sensor = PuritySoftSensor(n_estimators=50, n_ensemble=3, seed=42)
    sensor.fit(df_normal_csv)
    return sensor


class TestPuritySoftSensorFit:
    def test_fit_completes(self, df_normal_csv):
        sensor = PuritySoftSensor(n_estimators=50, n_ensemble=3, seed=0)
        sensor.fit(df_normal_csv)
        assert sensor.is_fitted

    def test_feature_names_set(self, trained_sensor):
        assert trained_sensor.feature_names == PILOT_PURITY_FEATURES

    def test_target_name_set(self, trained_sensor):
        assert trained_sensor.target_name == TARGET_PURITY


class TestPuritySoftSensorPredict:
    def test_predict_shape(self, trained_sensor, df_normal_csv):
        preds = trained_sensor.predict(df_normal_csv)
        assert preds.shape == (len(df_normal_csv),)

    def test_predict_physical_range(self, trained_sensor, df_normal_csv):
        preds = trained_sensor.predict(df_normal_csv)
        assert preds.min() > 80.0
        assert preds.max() < 100.0

    def test_predict_with_confidence(self, trained_sensor, df_normal_csv):
        preds, uncertainty = trained_sensor.predict_with_confidence(df_normal_csv)
        assert preds.shape == (len(df_normal_csv),)
        assert uncertainty.shape == (len(df_normal_csv),)
        assert (uncertainty >= 0).all()

    def test_r2_above_threshold(self, trained_sensor, df_normal_csv):
        from soft_sensor import compute_metrics
        preds = trained_sensor.predict(df_normal_csv)
        actuals = df_normal_csv[TARGET_PURITY].values
        metrics = compute_metrics(actuals, preds)
        assert metrics.r2 > 0.90
