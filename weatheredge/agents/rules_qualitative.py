"""Deterministic rules-based qualitative engine.

Replaces the LLM qualitative agent in the hot trading loop.
Runs in microseconds, is fully deterministic and testable.

The 4 rule categories:
1. Microclimate: static per-station lookup table
2. Unusual weather: METAR wind / pressure / precipitation
3. Model bias: RMSE-based damping + calibration history
4. Timing: lead time and time-of-day relative to daily temperature peak
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from weatheredge.agents.base import MarketOpportunity, Signal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule 1: Microclimate lookup table (static, does not change)
# ---------------------------------------------------------------------------
# Positive adjustment = model tends to *under*estimate actual temp → lean YES
# Negative adjustment = model tends to *over*estimate actual temp → lean NO
STATION_MICROCLIMATE: dict[str, float] = {
    "KNYC": +0.02,   # Central Park urban heat island; GFS/ECMWF run slightly cool
    "KMDW":  0.00,   # Midway airport; near-average model skill
    "KMIA": -0.01,   # Coastal maritime moderating effect; models slightly warm
    "KDEN": +0.02,   # 5280 ft altitude, large diurnal range; models underestimate peak
    "KLAX": -0.02,   # Marine layer suppresses afternoon highs; models often too warm
    "EGLC": +0.01,   # London City urban Thames corridor; slight warm bias vs. model
    "KAUS": +0.01,   # Austin urban heat island; models run cool
    "KSEA": -0.02,   # Marine influence; models consistently run warm vs. observed
    "SAEZ":  0.00,   # Buenos Aires; insufficient domestic bias history
    "LTAC":  0.00,   # Ankara; insufficient bias history
    "RKSS":  0.00,   # Seoul; insufficient bias history
    "KATL": +0.01,   # Atlanta urban heat island
    "KDFW": +0.01,   # Dallas urban surface heating
    "KPHX": +0.03,   # Phoenix desert UHI; models consistently underestimate afternoon highs
    "KIAH": -0.01,   # Houston Gulf Coast humidity; models run warm
    "KSFO": -0.03,   # San Francisco marine layer + fog; models consistently too warm
    "KBOS": +0.01,   # Boston urban heat island; slight warm bias vs. model
}

# ---------------------------------------------------------------------------
# Rule 4: Station UTC offsets (DST-adjusted, approximate for current season)
# ---------------------------------------------------------------------------
# BugFix M4: Use zoneinfo for correct DST-aware UTC offsets instead of hardcoded values
# This ensures metar.py and rules_qualitative.py agree on local time
def _station_utc_offset(station: str) -> int:
    """Get current UTC offset for a station, respecting DST."""
    _STATION_TZ: dict[str, str] = {
        "KNYC": "America/New_York", "KMDW": "America/Chicago",
        "KMIA": "America/New_York", "KDEN": "America/Denver",
        "KLAX": "America/Los_Angeles", "KAUS": "America/Chicago",
        "KSEA": "America/Los_Angeles", "KATL": "America/New_York",
        "KDFW": "America/Chicago", "KPHX": "America/Phoenix",
        "KIAH": "America/Chicago", "KSFO": "America/Los_Angeles",
        "KBOS": "America/New_York",
        "EGLC": "Europe/London", "SAEZ": "America/Argentina/Buenos_Aires",
        "LTAC": "Europe/Istanbul", "RKSS": "Asia/Seoul",
    }
    tz_name = _STATION_TZ.get(station)
    if tz_name:
        try:
            from datetime import datetime, timezone
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo(tz_name))
            return int(now.utcoffset().total_seconds() // 3600)
        except Exception:
            pass
    # Fallback to static table
    _FALLBACK: dict[str, int] = {
        "KNYC": -5, "KMDW": -6, "KMIA": -5, "KDEN": -7,
        "KLAX": -8, "KAUS": -6, "KSEA": -8, "KATL": -5,
        "KDFW": -6, "KPHX": -7, "KIAH": -6, "KSFO": -8, "KBOS": -5,
        "EGLC": 0, "SAEZ": -3, "LTAC": +3, "RKSS": +9,
    }
    return _FALLBACK.get(station, 0)

# Keep dict interface for backward compat — populated dynamically
STATION_UTC_OFFSET: dict[str, int] = {
    station: _station_utc_offset(station)
    for station in [
        "KNYC", "KMDW", "KMIA", "KDEN", "KLAX", "EGLC", "KAUS", "KSEA",
        "SAEZ", "LTAC", "RKSS", "KATL", "KDFW", "KPHX", "KIAH",
    ]
}

# METAR precipitation codes to check in raw METAR string
_PRECIP_CODES = frozenset([
    "RA", "SN", "DZ", "GR", "GS", "PL",
    "TSRA", "RASN", "FZRA", "FZDZ", "SHRA", "SHRA",
])

# RMSE damping: at rmse=2.0 → factor=1.0; at rmse=6.0 → factor=0.5
_RMSE_DAMP_BASELINE = 2.0
_RMSE_DAMP_SLOPE = 1 / 8.0   # lose 0.125 factor per °F above baseline
_RMSE_DAMP_FLOOR = 0.5


@dataclass
class _Contribution:
    rule: str
    adjustment: float
    reasoning: str


class RulesQualitativeEngine:
    """Deterministic drop-in replacement for QualitativeAgent in the hot loop.

    Same call signature as QualitativeAgent.analyze(); same Signal output format.
    No network calls, no LLM, runs in < 1 ms.
    """

    def analyze(
        self,
        opportunity: MarketOpportunity,
        conditions_tracker=None,    # CurrentConditionsTracker | None
        calibration_tracker=None,   # CalibrationTracker | None
    ) -> Signal:
        """Evaluate qualitative factors via deterministic rules.

        Returns Signal with:
          - side: 'buy_yes' / 'buy_no' / 'no_trade'
          - confidence: magnitude of net adjustment clamped to [0, 0.10]
          - agent_name: 'RulesQualitative'
          - reasoning: human-readable summary of which rules fired
        """
        contribs: list[_Contribution] = []

        # Rule 1: microclimate
        contribs += self._rule_microclimate(opportunity)

        # Rule 2: unusual weather (METAR)
        contribs += self._rule_unusual_weather(opportunity, conditions_tracker)

        # Rule 4: timing (computed before Rule 3 damping)
        contribs += self._rule_timing(opportunity, conditions_tracker)

        raw_total = sum(c.adjustment for c in contribs)

        # Rule 3: RMSE + calibration damping applied to combined signal
        damping, bias_note = self._rule_model_bias(opportunity, calibration_tracker)
        total = max(-0.10, min(0.10, raw_total * damping))

        parts = [c.reasoning for c in contribs if c.adjustment != 0.0]
        if bias_note:
            parts.append(bias_note)
        reasoning = "; ".join(parts) if parts else "no significant factors"

        if abs(total) < 0.01:
            return Signal(
                market_id=opportunity.market_id,
                side="no_trade",
                confidence=0.0,
                agent_name="Qualitative",
                reasoning=reasoning,
            )

        side = "buy_yes" if total > 0 else "buy_no"
        return Signal(
            market_id=opportunity.market_id,
            side=side,
            confidence=min(abs(total) * 10, 1.0),
            agent_name="Qualitative",
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Rule 1: Microclimate
    # ------------------------------------------------------------------

    def _rule_microclimate(self, opp: MarketOpportunity) -> list[_Contribution]:
        base = STATION_MICROCLIMATE.get(opp.station, 0.0)
        if base == 0.0:
            return []
        direction = "warmer" if base > 0 else "cooler"
        return [_Contribution(
            rule="microclimate",
            adjustment=base,
            reasoning=(
                f"{opp.station} microclimate: historically {direction} than model "
                f"forecast ({base:+.2f})"
            ),
        )]

    # ------------------------------------------------------------------
    # Rule 2: Unusual weather from METAR
    # ------------------------------------------------------------------

    def _rule_unusual_weather(
        self, opp: MarketOpportunity, tracker
    ) -> list[_Contribution]:
        if tracker is None:
            return []
        daily = tracker.get_tracker(opp.station)
        if daily is None or daily.last_observation is None:
            return []

        obs = daily.last_observation
        contribs: list[_Contribution] = []

        # --- High winds (>25 kt): mixing prevents extreme surface highs ---
        wind_kt = obs.wind_speed_kt or 0.0
        if wind_kt >= 25:
            adj = -0.02 if opp.direction == "above" else +0.01
            contribs.append(_Contribution(
                rule="high_winds",
                adjustment=adj,
                reasoning=(
                    f"high winds {wind_kt:.0f}kt at {opp.station}: "
                    f"mixing prevents extreme temp ({adj:+.2f})"
                ),
            ))

        # --- Precipitation: cooling suppresses daily high ---
        raw = obs.raw_metar.upper()
        has_precip = any(code in raw for code in _PRECIP_CODES)
        if has_precip:
            adj = -0.03 if opp.direction == "above" else +0.02
            matched = next((c for c in _PRECIP_CODES if c in raw), "precip")
            contribs.append(_Contribution(
                rule="precipitation",
                adjustment=adj,
                reasoning=(
                    f"precipitation ({matched}) at {opp.station}: "
                    f"cooling suppresses high ({adj:+.2f})"
                ),
            ))

        # --- Pressure anomaly (vs. standard 1013.25 hPa) ---
        altim = obs.altimeter_hpa
        if altim is not None:
            deviation = altim - 1013.25
            if deviation < -8:   # Low pressure: cloudiness / cooling
                adj = -0.01 if opp.direction == "above" else +0.01
                contribs.append(_Contribution(
                    rule="low_pressure",
                    adjustment=adj,
                    reasoning=(
                        f"low pressure {altim:.1f} hPa at {opp.station}: "
                        f"associated cloudiness/cooling ({adj:+.2f})"
                    ),
                ))
            elif deviation > 12:  # High pressure: clear skies / warming
                adj = +0.01 if opp.direction == "above" else -0.01
                contribs.append(_Contribution(
                    rule="high_pressure",
                    adjustment=adj,
                    reasoning=(
                        f"high pressure {altim:.1f} hPa at {opp.station}: "
                        f"clear skies warm-bias ({adj:+.2f})"
                    ),
                ))

        return contribs

    # ------------------------------------------------------------------
    # Rule 3: Model bias / accuracy (returns damping factor + note)
    # ------------------------------------------------------------------

    def _rule_model_bias(
        self, opp: MarketOpportunity, calibration_tracker
    ) -> tuple[float, str]:
        """Return (damping_factor, reasoning_note).

        damping_factor 1.0 = no change; 0.5 = signal magnitude halved.
        Derived from opportunity RMSE (already computed upstream) and optional
        CalibrationTracker history.
        """
        rmse = opp.rmse
        # Linear damping: full signal at rmse=2.0, half signal at rmse=6.0
        damping = max(
            _RMSE_DAMP_FLOOR,
            1.0 - max(0.0, rmse - _RMSE_DAMP_BASELINE) * _RMSE_DAMP_SLOPE,
        )

        notes: list[str] = []
        if rmse > 3.5:
            notes.append(f"high RMSE {rmse:.1f}°F dampens signal (×{damping:.2f})")

        # CalibrationTracker: if we have ≥20 resolved trades for this station,
        # check whether the model is systematically poorly calibrated.
        if calibration_tracker is not None:
            try:
                stats = calibration_tracker.get_calibration_stats(station=opp.station)
                count = stats.get("count", 0)
                avg_brier = stats.get("avg_brier")
                if count >= 20 and avg_brier is not None:
                    if avg_brier > 0.20:
                        damping *= 0.8
                        notes.append(
                            f"poor calibration Brier={avg_brier:.3f} (n={count}): "
                            f"signal dampened further"
                        )
            except Exception as exc:
                log.debug("calibration_tracker query failed in rules engine: %s", exc)

        return damping, "; ".join(notes)

    # ------------------------------------------------------------------
    # Rule 4: Timing (lead time + time-of-day vs. daily peak)
    # ------------------------------------------------------------------

    def _rule_timing(
        self, opp: MarketOpportunity, tracker
    ) -> list[_Contribution]:
        # Only meaningful for same-day or next-day markets
        if opp.lead_time_days > 1:
            return []

        utc_offset = STATION_UTC_OFFSET.get(opp.station, 0)
        local_hour = (datetime.now(timezone.utc).hour + utc_offset) % 24

        contribs: list[_Contribution] = []

        if opp.direction == "above" and opp.lead_time_days == 0:
            if local_hour >= 17:
                # Past typical daily high (peaks 14–16 local)
                if tracker is not None:
                    daily = tracker.get_tracker(opp.station)
                    if (
                        daily is not None
                        and daily.daily_high_f > -999.0
                        and daily.daily_high_f < opp.threshold
                    ):
                        gap = opp.threshold - daily.daily_high_f
                        adj = max(-0.05, -0.01 * gap)
                        contribs.append(_Contribution(
                            rule="late_day_below_threshold",
                            adjustment=adj,
                            reasoning=(
                                f"past peak hour (local {local_hour:02d}:00), "
                                f"daily high {daily.daily_high_f:.1f}°F still "
                                f"{gap:.1f}°F below {opp.threshold:.1f}°F ({adj:+.2f})"
                            ),
                        ))
                else:
                    # No METAR — small pessimistic bias for above-threshold late day
                    contribs.append(_Contribution(
                        rule="late_day_no_metar",
                        adjustment=-0.01,
                        reasoning=(
                            f"past peak hour (local {local_hour:02d}:00), "
                            f"no METAR: slight downward bias (-0.01)"
                        ),
                    ))

            elif local_hour <= 9:
                # Early morning: temps still climbing toward daily peak
                contribs.append(_Contribution(
                    rule="early_morning_rising",
                    adjustment=+0.01,
                    reasoning=(
                        f"early morning (local {local_hour:02d}:00): "
                        f"temps still climbing toward daily peak (+0.01)"
                    ),
                ))

        return contribs
