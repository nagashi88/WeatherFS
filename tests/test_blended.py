"""Tests for Enhancement 5: Blended probability with HRRR priority weighting."""

import pytest

from weatheredge.probability.blended import (
    blended_probability,
    WEIGHTS_US,
    WEIGHTS_NON_US,
    _get_lead_bucket,
)


class TestLeadBucket:
    def test_short(self):
        assert _get_lead_bucket(6) == "short"
        assert _get_lead_bucket(11.9) == "short"

    def test_medium(self):
        assert _get_lead_bucket(12) == "medium"
        assert _get_lead_bucket(24) == "medium"

    def test_long(self):
        assert _get_lead_bucket(25) == "long"
        assert _get_lead_bucket(72) == "long"


class TestBlendedProbability:
    def test_metar_override_takes_precedence(self):
        prob, model, details = blended_probability(
            threshold=75.0,
            ensemble_temps=[80.0] * 31,
            gfs_forecast_temp=80.0,
            metar_override=0.99,
        )
        assert prob == 0.99
        assert model == "metar_override"

    def test_us_short_range_hrrr_dominant(self):
        """For US with <12h lead, HRRR should have 60% weight."""
        prob, model, details = blended_probability(
            threshold=70.0,
            direction="above",
            is_us_station=True,
            lead_time_hours=6,
            ensemble_temps=[75.0] * 31,
            gfs_forecast_temp=75.0,
            ecmwf_forecast_temp=75.0,
            hrrr_forecast_temp=75.0,
        )
        assert details["weight_bucket"] == "short"
        assert details["weights"]["hrrr"] == 0.60
        assert prob > 0.5  # 75°F forecast > 70°F threshold
        assert "hrrr" in model

    def test_us_medium_range(self):
        prob, model, details = blended_probability(
            threshold=70.0,
            is_us_station=True,
            lead_time_hours=18,
            ensemble_temps=[75.0] * 31,
            gfs_forecast_temp=75.0,
            ecmwf_forecast_temp=75.0,
            hrrr_forecast_temp=75.0,
        )
        assert details["weight_bucket"] == "medium"
        assert details["weights"]["hrrr"] == 0.40

    def test_us_long_range(self):
        prob, model, details = blended_probability(
            threshold=70.0,
            is_us_station=True,
            lead_time_hours=48,
            ensemble_temps=[75.0] * 31,
            gfs_forecast_temp=75.0,
            ecmwf_forecast_temp=75.0,
            hrrr_forecast_temp=75.0,
        )
        assert details["weight_bucket"] == "long"
        assert details["weights"]["hrrr"] == 0.20
        assert details["weights"]["gfs_ensemble"] == 0.40

    def test_non_us_no_hrrr(self):
        """Non-US cities should get 50/50 GFS/ECMWF, no HRRR."""
        prob, model, details = blended_probability(
            threshold=20.0,
            is_us_station=False,
            lead_time_hours=12,
            ensemble_temps=[25.0] * 31,
            ecmwf_forecast_temp=25.0,
            hrrr_forecast_temp=25.0,  # should be ignored
        )
        assert details["weight_bucket"] == "non_us"
        assert "hrrr" not in model  # HRRR not used for non-US

    def test_ensemble_only_fallback(self):
        """If only ensemble data available, should still produce a probability."""
        # Use a spread of ensemble members so probability is between 0 and 1
        temps = [68.0] * 15 + [72.0] * 16  # ~half above, half below threshold
        prob, model, details = blended_probability(
            threshold=70.0,
            is_us_station=True,
            lead_time_hours=24,
            ensemble_temps=temps,
        )
        assert 0 < prob < 1
        assert "gfs_ensemble" in model

    def test_no_data_fallback(self):
        """No forecast data at all should return 0.5."""
        prob, model, details = blended_probability(
            threshold=70.0,
            is_us_station=True,
            lead_time_hours=24,
        )
        assert prob == 0.50
        assert model == "no_data"

    def test_gfs_point_fallback(self):
        """If only GFS point forecast, should use Gaussian fallback."""
        prob, model, details = blended_probability(
            threshold=70.0,
            is_us_station=False,
            lead_time_hours=24,
            gfs_forecast_temp=75.0,
        )
        assert 0 < prob < 1
        assert model == "gaussian_fallback"

    def test_probability_bounded(self):
        """All probabilities should be between 0 and 1."""
        for threshold in [50, 70, 90]:
            for temp in [40, 60, 80, 100]:
                prob, _, _ = blended_probability(
                    threshold=threshold,
                    ensemble_temps=[temp] * 31,
                    gfs_forecast_temp=temp,
                    ecmwf_forecast_temp=temp,
                    hrrr_forecast_temp=temp,
                    is_us_station=True,
                    lead_time_hours=6,
                )
                assert 0.0 <= prob <= 1.0

    def test_hrrr_tighter_rmse(self):
        """HRRR should use tighter RMSE (0.85x) than base."""
        _, _, details = blended_probability(
            threshold=70.0,
            rmse=5.0,
            is_us_station=True,
            lead_time_hours=6,
            hrrr_forecast_temp=72.0,
        )
        assert details["hrrr_rmse"] == pytest.approx(4.25, abs=0.01)


class TestHrrrRmseFloor:
    """G4 regression: 0.85× multiplier must not push hrrr_rmse below the HRRR floor
    once main.py has already floored the input rmse upstream via rmse_floor_dynamic.
    """

    def test_hrrr_short_lead_rmse_not_driven_below_floor(self):
        import weatheredge.config as cfg
        orig = cfg.MODULES.get("rmse_floor_dynamic", False)
        cfg.MODULES["rmse_floor_dynamic"] = True
        try:
            _, _, details = blended_probability(
                threshold=70.0,
                rmse=1.5,  # already floored upstream at HRRR <12h floor (1.5°F)
                is_us_station=True,
                lead_time_hours=6,
                hrrr_forecast_temp=72.0,
            )
            assert details["hrrr_rmse"] == pytest.approx(1.5, abs=0.01)
        finally:
            cfg.MODULES["rmse_floor_dynamic"] = orig

    def test_hrrr_long_lead_rmse_not_driven_below_floor(self):
        import weatheredge.config as cfg
        orig = cfg.MODULES.get("rmse_floor_dynamic", False)
        cfg.MODULES["rmse_floor_dynamic"] = True
        try:
            _, _, details = blended_probability(
                threshold=70.0,
                rmse=3.0,  # GFS-floored (long lead primary is GFS)
                is_us_station=True,
                lead_time_hours=48,
                hrrr_forecast_temp=72.0,
            )
            assert details["hrrr_rmse"] == pytest.approx(3.0, abs=0.01)
        finally:
            cfg.MODULES["rmse_floor_dynamic"] = orig

    def test_floor_inactive_preserves_multiplier(self):
        import weatheredge.config as cfg
        orig = cfg.MODULES.get("rmse_floor_dynamic", False)
        cfg.MODULES["rmse_floor_dynamic"] = False
        try:
            _, _, details = blended_probability(
                threshold=70.0,
                rmse=5.0,
                is_us_station=True,
                lead_time_hours=6,
                hrrr_forecast_temp=72.0,
            )
            assert details["hrrr_rmse"] == pytest.approx(4.25, abs=0.01)
        finally:
            cfg.MODULES["rmse_floor_dynamic"] = orig

    def test_multiplier_still_applies_above_floor(self):
        """When rmse is well above HRRR floor, the 0.85× tightening still takes effect."""
        import weatheredge.config as cfg
        orig = cfg.MODULES.get("rmse_floor_dynamic", False)
        cfg.MODULES["rmse_floor_dynamic"] = True
        try:
            _, _, details = blended_probability(
                threshold=70.0,
                rmse=5.0,  # 5.0 * 0.85 = 4.25, well above 1.5°F HRRR floor
                is_us_station=True,
                lead_time_hours=6,
                hrrr_forecast_temp=72.0,
            )
            assert details["hrrr_rmse"] == pytest.approx(4.25, abs=0.01)
        finally:
            cfg.MODULES["rmse_floor_dynamic"] = orig


class TestWeightTables:
    def test_us_weights_sum_to_one(self):
        for bucket, weights in WEIGHTS_US.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0, abs=0.001), f"{bucket} weights sum to {total}"

    def test_non_us_weights_sum_to_one(self):
        total = sum(WEIGHTS_NON_US.values())
        assert total == pytest.approx(1.0, abs=0.001)
