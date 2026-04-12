"""Tests for the deterministic RulesQualitativeEngine.

Covers all 4 rule categories:
1. Microclimate: static per-station bias lookup
2. Unusual weather: METAR wind / precipitation / pressure
3. Model bias: RMSE damping + calibration tracker
4. Timing: lead time and time-of-day relative to daily peak
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from weatheredge.agents.base import MarketOpportunity, Signal
from weatheredge.agents.rules_qualitative import (
    RulesQualitativeEngine,
    STATION_MICROCLIMATE,
    STATION_UTC_OFFSET,
)
from weatheredge.data.metar import CurrentConditionsTracker, DailyTracker, StationObservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_opp(**kwargs) -> MarketOpportunity:
    defaults = dict(
        market_id="MKT-1",
        platform="kalshi",
        city="NYC",
        station="KNYC",
        threshold=85.0,
        direction="above",
        market_price=0.40,
        model_probability=0.65,
        net_edge=0.20,
        forecast_temp=87.0,
        rmse=2.5,
        lead_time_days=1,
    )
    defaults.update(kwargs)
    return MarketOpportunity(**defaults)


def make_tracker_with_obs(
    station: str = "KNYC",
    temp_f: float = 78.0,
    wind_kt: float = 5.0,
    altim_hpa: float = 1013.25,
    raw_metar: str = "",
    daily_high_f: float | None = None,
) -> CurrentConditionsTracker:
    """Build a CurrentConditionsTracker pre-loaded with a single station observation."""
    obs = StationObservation(
        station=station,
        temperature_f=temp_f,
        wind_speed_kt=wind_kt,
        altimeter_hpa=altim_hpa,
        raw_metar=raw_metar,
    )
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily = DailyTracker(station=station, date=today)
    daily.last_observation = obs
    daily.daily_high_f = daily_high_f if daily_high_f is not None else temp_f
    daily.daily_low_f = temp_f
    daily.observation_count = 1

    tracker = CurrentConditionsTracker()
    tracker._trackers[station] = daily
    return tracker


engine = RulesQualitativeEngine()


# ===========================================================================
# Rule 1: Microclimate
# ===========================================================================

class TestRuleMicroclimate:

    def test_phoenix_positive_adjustment(self):
        """KPHX has +0.03 microclimate bias → should lean buy_yes."""
        opp = make_opp(station="KPHX", city="Phoenix")
        sig = engine.analyze(opp)
        # Net: +0.03 (microclimate only, rmse=2.5 → damping≈1.0); confidence = min(0.03*10, 1.0) = 0.30
        assert sig.side == "buy_yes"
        assert sig.confidence == pytest.approx(0.30, abs=0.05)
        assert "microclimate" in sig.reasoning.lower()

    def test_lax_negative_adjustment(self):
        """KLAX has -0.02 microclimate bias → should lean buy_no."""
        opp = make_opp(station="KLAX", city="LAX")
        sig = engine.analyze(opp)
        assert sig.side == "buy_no"
        assert sig.confidence == pytest.approx(0.20, abs=0.05)

    def test_neutral_station_no_microclimate(self):
        """KMDW has 0.00 → no microclimate contribution."""
        opp = make_opp(station="KMDW", city="Chicago", rmse=2.0)
        contribs = engine._rule_microclimate(opp)
        assert contribs == []

    def test_unknown_station_no_microclimate(self):
        """Station not in table → no contribution."""
        opp = make_opp(station="KUNKNOWN")
        contribs = engine._rule_microclimate(opp)
        assert contribs == []

    def test_all_known_stations_have_entries(self):
        """Every station in STATION_UTC_OFFSET should have a microclimate entry."""
        for station in STATION_UTC_OFFSET:
            assert station in STATION_MICROCLIMATE, f"{station} missing from STATION_MICROCLIMATE"

    def test_den_positive(self):
        opp = make_opp(station="KDEN", city="Denver")
        contribs = engine._rule_microclimate(opp)
        assert len(contribs) == 1
        assert contribs[0].adjustment == pytest.approx(0.02)

    def test_sea_negative(self):
        opp = make_opp(station="KSEA", city="Seattle")
        contribs = engine._rule_microclimate(opp)
        assert len(contribs) == 1
        assert contribs[0].adjustment == pytest.approx(-0.02)


# ===========================================================================
# Rule 2: Unusual weather (METAR)
# ===========================================================================

class TestRuleUnusualWeather:

    def test_high_winds_above_market(self):
        """Wind ≥25kt on above-threshold market → negative adjustment."""
        tracker = make_tracker_with_obs(wind_kt=30.0)
        opp = make_opp(station="KNYC", direction="above")
        contribs = engine._rule_unusual_weather(opp, tracker)
        wind_contribs = [c for c in contribs if c.rule == "high_winds"]
        assert len(wind_contribs) == 1
        assert wind_contribs[0].adjustment == pytest.approx(-0.02)

    def test_high_winds_below_market(self):
        """Wind ≥25kt on below-threshold market → positive adjustment."""
        tracker = make_tracker_with_obs(wind_kt=28.0)
        opp = make_opp(station="KNYC", direction="below")
        contribs = engine._rule_unusual_weather(opp, tracker)
        wind_contribs = [c for c in contribs if c.rule == "high_winds"]
        assert wind_contribs[0].adjustment == pytest.approx(+0.01)

    def test_calm_winds_no_contribution(self):
        """Wind <25kt → no high_winds contribution."""
        tracker = make_tracker_with_obs(wind_kt=10.0)
        opp = make_opp(station="KNYC", direction="above")
        contribs = engine._rule_unusual_weather(opp, tracker)
        assert not any(c.rule == "high_winds" for c in contribs)

    def test_precipitation_above_market(self):
        """RA in METAR on above-threshold market → -0.03."""
        tracker = make_tracker_with_obs(raw_metar="KNYC 301354Z 15007KT 5SM RA OVC010 18/16")
        opp = make_opp(station="KNYC", direction="above")
        contribs = engine._rule_unusual_weather(opp, tracker)
        precip = [c for c in contribs if c.rule == "precipitation"]
        assert len(precip) == 1
        assert precip[0].adjustment == pytest.approx(-0.03)

    def test_precipitation_below_market(self):
        """SN in METAR on below-threshold market → +0.02."""
        tracker = make_tracker_with_obs(station="KDEN", raw_metar="KDEN 010000Z 30010KT 1SM SN OVC005 M02/M05")
        opp = make_opp(station="KDEN", direction="below")
        contribs = engine._rule_unusual_weather(opp, tracker)
        precip = [c for c in contribs if c.rule == "precipitation"]
        assert precip[0].adjustment == pytest.approx(+0.02)

    def test_no_precipitation_no_contribution(self):
        """Clear METAR → no precipitation contribution."""
        tracker = make_tracker_with_obs(raw_metar="KNYC 301354Z 15007KT 10SM SKC 22/10 A2992")
        opp = make_opp(station="KNYC", direction="above")
        contribs = engine._rule_unusual_weather(opp, tracker)
        assert not any(c.rule == "precipitation" for c in contribs)

    def test_low_pressure_above_market(self):
        """Pressure <1005.25 hPa on above-threshold market → -0.01."""
        tracker = make_tracker_with_obs(altim_hpa=1000.0)  # dev = -13.25
        opp = make_opp(station="KNYC", direction="above")
        contribs = engine._rule_unusual_weather(opp, tracker)
        lp = [c for c in contribs if c.rule == "low_pressure"]
        assert len(lp) == 1
        assert lp[0].adjustment == pytest.approx(-0.01)

    def test_high_pressure_above_market(self):
        """Pressure >1025.25 hPa on above-threshold market → +0.01."""
        tracker = make_tracker_with_obs(altim_hpa=1030.0)  # dev = +16.75
        opp = make_opp(station="KNYC", direction="above")
        contribs = engine._rule_unusual_weather(opp, tracker)
        hp = [c for c in contribs if c.rule == "high_pressure"]
        assert len(hp) == 1
        assert hp[0].adjustment == pytest.approx(+0.01)

    def test_normal_pressure_no_contribution(self):
        """Pressure near 1013 hPa → no pressure contribution."""
        tracker = make_tracker_with_obs(altim_hpa=1013.0)
        opp = make_opp(station="KNYC", direction="above")
        contribs = engine._rule_unusual_weather(opp, tracker)
        assert not any(c.rule in ("low_pressure", "high_pressure") for c in contribs)

    def test_no_tracker_returns_empty(self):
        """None tracker → no contributions."""
        opp = make_opp(station="KNYC")
        assert engine._rule_unusual_weather(opp, None) == []

    def test_tracker_no_data_for_station(self):
        """Tracker present but no data for requested station → empty."""
        tracker = CurrentConditionsTracker()  # empty
        opp = make_opp(station="KNYC")
        assert engine._rule_unusual_weather(opp, tracker) == []

    def test_multiple_factors_combine(self):
        """High winds + precipitation → two negative contributions."""
        tracker = make_tracker_with_obs(
            wind_kt=30.0,
            raw_metar="KNYC 301354Z 30030KT 2SM TSRA OVC020 20/18",
            altim_hpa=1000.0,
        )
        opp = make_opp(station="KNYC", direction="above")
        contribs = engine._rule_unusual_weather(opp, tracker)
        rules = {c.rule for c in contribs}
        assert "high_winds" in rules
        assert "precipitation" in rules
        assert "low_pressure" in rules
        total = sum(c.adjustment for c in contribs)
        assert total < -0.04  # all three push negative


# ===========================================================================
# Rule 3: Model bias / RMSE damping
# ===========================================================================

class TestRuleModelBias:

    def test_low_rmse_full_damping(self):
        """RMSE=2.0 → damping=1.0 (no reduction)."""
        opp = make_opp(rmse=2.0)
        damping, _ = engine._rule_model_bias(opp, None)
        assert damping == pytest.approx(1.0)

    def test_high_rmse_reduced_damping(self):
        """RMSE=6.0 → damping=0.5 (floor)."""
        opp = make_opp(rmse=6.0)
        damping, _ = engine._rule_model_bias(opp, None)
        assert damping == pytest.approx(0.5)

    def test_moderate_rmse_intermediate_damping(self):
        """RMSE=4.0 → damping = 1 - (4-2)/8 = 0.75."""
        opp = make_opp(rmse=4.0)
        damping, _ = engine._rule_model_bias(opp, None)
        assert damping == pytest.approx(0.75)

    def test_very_high_rmse_capped_at_floor(self):
        """RMSE=20.0 → damping capped at 0.5."""
        opp = make_opp(rmse=20.0)
        damping, _ = engine._rule_model_bias(opp, None)
        assert damping == pytest.approx(0.5)

    def test_high_rmse_note_in_reasoning(self):
        """RMSE>3.5 → reasoning note mentions high RMSE."""
        opp = make_opp(rmse=4.5)
        _, note = engine._rule_model_bias(opp, None)
        assert "rmse" in note.lower()

    def test_low_rmse_no_note(self):
        """RMSE=2.5 → no note."""
        opp = make_opp(rmse=2.5)
        _, note = engine._rule_model_bias(opp, None)
        assert note == ""

    def test_poor_calibration_further_dampens(self):
        """Poor Brier score (>0.20) with ≥20 samples → additional 0.8× damping."""
        cal = MagicMock()
        cal.get_calibration_stats.return_value = {"avg_brier": 0.25, "count": 30}
        opp = make_opp(rmse=2.0)  # baseline damping = 1.0
        damping, note = engine._rule_model_bias(opp, cal)
        # 1.0 * 0.8 = 0.8
        assert damping == pytest.approx(0.8)
        assert "calibration" in note.lower()

    def test_insufficient_calibration_samples_ignored(self):
        """<20 calibration samples → not applied."""
        cal = MagicMock()
        cal.get_calibration_stats.return_value = {"avg_brier": 0.30, "count": 5}
        opp = make_opp(rmse=2.0)
        damping, _ = engine._rule_model_bias(opp, cal)
        assert damping == pytest.approx(1.0)

    def test_calibration_error_ignored_gracefully(self):
        """CalibrationTracker raising exception → damping unchanged."""
        cal = MagicMock()
        cal.get_calibration_stats.side_effect = RuntimeError("DB error")
        opp = make_opp(rmse=2.0)
        damping, _ = engine._rule_model_bias(opp, cal)
        assert damping == pytest.approx(1.0)

    def test_high_rmse_dampens_microclimate(self):
        """High RMSE should reduce the net signal from microclimate."""
        opp_low_rmse = make_opp(station="KPHX", rmse=2.0)
        opp_high_rmse = make_opp(station="KPHX", rmse=6.0)
        sig_low = engine.analyze(opp_low_rmse)
        sig_high = engine.analyze(opp_high_rmse)
        assert sig_high.confidence < sig_low.confidence


# ===========================================================================
# Rule 4: Timing
# ===========================================================================

class TestRuleTiming:

    def _mock_utc_hour(self, utc_hour: int):
        """Patch datetime.now to return a fixed UTC hour."""
        fixed = datetime(2026, 3, 30, utc_hour, 0, 0, tzinfo=timezone.utc)
        return patch(
            "weatheredge.agents.rules_qualitative.datetime",
            wraps=datetime,
        )

    def test_lead_time_gt_1_no_timing(self):
        """lead_time_days>1 → no timing contributions regardless of hour."""
        opp = make_opp(lead_time_days=2, direction="above")
        contribs = engine._rule_timing(opp, None)
        assert contribs == []

    def test_early_morning_above_positive(self):
        """Same-day above-threshold at local 06:00 → +0.01 early_morning_rising."""
        # KNYC UTC-4: UTC hour 10 → local 06:00
        with patch("weatheredge.agents.rules_qualitative.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            opp = make_opp(station="KNYC", direction="above", lead_time_days=0)
            contribs = engine._rule_timing(opp, None)

        early = [c for c in contribs if c.rule == "early_morning_rising"]
        assert len(early) == 1
        assert early[0].adjustment == pytest.approx(+0.01)

    def test_late_day_below_threshold_with_metar(self):
        """Same-day above-threshold at local 19:00, daily high 3°F below threshold → negative adj."""
        # KNYC UTC-4: UTC hour 23 → local 19:00
        tracker = make_tracker_with_obs(
            station="KNYC", temp_f=82.0, daily_high_f=82.0
        )
        opp = make_opp(
            station="KNYC", direction="above", lead_time_days=0, threshold=85.0
        )
        with patch("weatheredge.agents.rules_qualitative.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 30, 23, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            contribs = engine._rule_timing(opp, tracker)

        late = [c for c in contribs if c.rule == "late_day_below_threshold"]
        assert len(late) == 1
        # gap = 85 - 82 = 3 → adj = max(-0.05, -0.01*3) = -0.03
        assert late[0].adjustment == pytest.approx(-0.03)

    def test_late_day_large_gap_capped(self):
        """Gap of 8°F → adj capped at -0.05."""
        tracker = make_tracker_with_obs(
            station="KNYC", temp_f=70.0, daily_high_f=70.0
        )
        opp = make_opp(
            station="KNYC", direction="above", lead_time_days=0, threshold=78.0
        )
        with patch("weatheredge.agents.rules_qualitative.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 30, 23, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            contribs = engine._rule_timing(opp, tracker)

        late = [c for c in contribs if c.rule == "late_day_below_threshold"]
        assert len(late) == 1
        assert late[0].adjustment == pytest.approx(-0.05)

    def test_late_day_already_exceeded_threshold(self):
        """Daily high already exceeds threshold → no late_day contribution (handled elsewhere)."""
        tracker = make_tracker_with_obs(
            station="KNYC", temp_f=90.0, daily_high_f=90.0
        )
        opp = make_opp(
            station="KNYC", direction="above", lead_time_days=0, threshold=85.0
        )
        with patch("weatheredge.agents.rules_qualitative.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 30, 23, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            contribs = engine._rule_timing(opp, tracker)

        late = [c for c in contribs if c.rule == "late_day_below_threshold"]
        assert late == []  # threshold already exceeded, no downward signal

    def test_late_day_no_metar_fallback(self):
        """Past peak time, no METAR → small -0.01 fallback adjustment."""
        opp = make_opp(station="KNYC", direction="above", lead_time_days=0)
        with patch("weatheredge.agents.rules_qualitative.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 30, 23, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            contribs = engine._rule_timing(opp, None)

        fallback = [c for c in contribs if c.rule == "late_day_no_metar"]
        assert len(fallback) == 1
        assert fallback[0].adjustment == pytest.approx(-0.01)

    def test_direction_below_no_timing(self):
        """direction='below' → no timing contributions (timing only applies to 'above')."""
        opp = make_opp(station="KNYC", direction="below", lead_time_days=0)
        with patch("weatheredge.agents.rules_qualitative.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            contribs = engine._rule_timing(opp, None)

        assert contribs == []

    def test_midday_no_timing_contribution(self):
        """Local midday (12:00) → no early/late timing."""
        # KNYC UTC-4: UTC 16 → local 12:00
        opp = make_opp(station="KNYC", direction="above", lead_time_days=0)
        with patch("weatheredge.agents.rules_qualitative.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            contribs = engine._rule_timing(opp, None)

        assert contribs == []


# ===========================================================================
# Integration: analyze() end-to-end
# ===========================================================================

class TestAnalyzeIntegration:

    def test_returns_signal_type(self):
        opp = make_opp()
        sig = engine.analyze(opp)
        assert isinstance(sig, Signal)
        assert sig.agent_name == "Qualitative"

    def test_signal_side_values(self):
        """Side must be one of the three valid values."""
        opp = make_opp(station="KPHX")
        sig = engine.analyze(opp)
        assert sig.side in ("buy_yes", "buy_no", "no_trade")

    def test_confidence_clamped_to_010(self):
        """Confidence (scaled = |adjustment| * 10) must be in [0, 1.0]."""
        for station in ["KNYC", "KPHX", "KLAX", "KSEA"]:
            tracker = make_tracker_with_obs(
                station=station, wind_kt=30.0,
                raw_metar="TSRA", altim_hpa=998.0,
            )
            opp = make_opp(station=station, direction="above")
            sig = engine.analyze(opp, conditions_tracker=tracker)
            assert 0.0 <= sig.confidence <= 1.0, (
                f"{station}: confidence {sig.confidence} out of [0, 1.0]"
            )

    def test_no_trade_when_adjustment_tiny(self):
        """KMDW (0.00 microclimate) with no other factors → no_trade."""
        opp = make_opp(station="KMDW", rmse=2.0, lead_time_days=1)
        sig = engine.analyze(opp)
        assert sig.side == "no_trade"
        assert sig.confidence == 0.0

    def test_reasoning_non_empty_when_trading(self):
        """When a trade signal fires, reasoning must be non-empty."""
        opp = make_opp(station="KPHX")
        sig = engine.analyze(opp)
        assert sig.side != "no_trade"
        assert len(sig.reasoning) > 0

    def test_deterministic(self):
        """Same inputs always produce same output."""
        opp = make_opp(station="KNYC")
        results = [engine.analyze(opp) for _ in range(5)]
        sides = {r.side for r in results}
        confs = {r.confidence for r in results}
        assert len(sides) == 1
        assert len(confs) == 1

    def test_all_factors_combine(self):
        """Stacking microclimate + weather + timing should compound adjustments."""
        tracker = make_tracker_with_obs(
            station="KPHX",
            wind_kt=5.0,
            altim_hpa=1030.0,  # high pressure → +0.01
            raw_metar="CLR",
        )
        opp = make_opp(station="KPHX", direction="above", lead_time_days=0, rmse=2.0)
        with patch("weatheredge.agents.rules_qualitative.datetime") as mock_dt:
            # KPHX UTC-7: UTC 14 → local 07:00 (early morning)
            mock_dt.now.return_value = datetime(2026, 3, 30, 14, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            sig = engine.analyze(opp, conditions_tracker=tracker)

        # KPHX: +0.03 micro + 0.01 high_pressure + 0.01 early_morning = +0.05; confidence = min(0.05*10, 1.0) = 0.50
        assert sig.side == "buy_yes"
        assert sig.confidence == pytest.approx(0.50, abs=0.10)
