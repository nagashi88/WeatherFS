"""Station-specific bias correction — STEP 5.

Computes systematic forecast errors by (station, month, wind_regime, cloud_cover)
and applies corrections to improve probability estimates.

Two levels:
1. Simple lookup table: BIAS_TABLE[station, month, wind, cloud] = mean_error
2. BiasAnalyzer: builds the table from historical DailyRecord data
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

log = logging.getLogger(__name__)


# Station-specific RMSE lookup (PRD Section 3.1.3, lines 269-283)
# BugFix H3: Populate STATION_RMSE for all traded stations (not just KNYC/KMDW)
# Default RMSE values based on station type/climate variability
# RMSE values represent daily-max temperature forecast error (°F), not instantaneous 2m temp.
# Daily-max has ~35-50% higher RMSE than hourly 2m temperature that NWP models are typically
# validated against. Values raised based on Phase 0 Brier=0.50 evidence of overconfidence.
_DEFAULT_US_RMSE = {
    "gfs": {0: 4.0, 1: 5.0, 2: 7.0, 3: 9.0},
    "ecmwf": {0: 3.5, 1: 4.5, 2: 6.5, 3: 8.5},
    "hrrr": {0: 3.0, 1: 4.0, 2: 6.0},
}
_DEFAULT_INTL_RMSE = {
    "gfs": {0: 4.5, 1: 5.5, 2: 7.5, 3: 9.5},
    "ecmwf": {0: 3.5, 1: 4.5, 2: 6.5, 3: 8.5},
}
STATION_RMSE: dict[str, dict[str, dict[int, float]]] = {
    "KNYC": {  # Central Park, NYC — maritime influence reduces extremes slightly
        "gfs": {0: 3.5, 1: 4.5, 2: 6.5, 3: 8.5},
        "ecmwf": {0: 3.0, 1: 4.0, 2: 5.5, 3: 7.5},
        "hrrr": {0: 2.5, 1: 3.5, 2: 5.5},
    },
    "KMDW": {  # Midway Airport, Chicago — continental, high variability
        "gfs": {0: 4.0, 1: 5.0, 2: 7.0, 3: 9.0},
        "ecmwf": {0: 3.5, 1: 4.5, 2: 6.5, 3: 8.5},
        "hrrr": {0: 3.0, 1: 4.0, 2: 6.0},
    },
    # US stations — default US RMSE (will be refined by RMSE updater)
    "KMIA": {**_DEFAULT_US_RMSE},
    "KDEN": {**_DEFAULT_US_RMSE},
    "KLAX": {**_DEFAULT_US_RMSE},
    "KAUS": {**_DEFAULT_US_RMSE},
    "KSEA": {**_DEFAULT_US_RMSE},
    "KATL": {**_DEFAULT_US_RMSE},
    "KDFW": {**_DEFAULT_US_RMSE},
    "KPHX": {"gfs": {0: 3.0, 1: 4.0, 2: 6.0, 3: 8.0}, "ecmwf": {0: 2.5, 1: 3.5, 2: 5.5, 3: 7.5}, "hrrr": {0: 2.5, 1: 3.0, 2: 5.0}},  # Desert: lower cloud/precip variance but heat extremes
    "KIAH": {**_DEFAULT_US_RMSE},
    "KSFO": {**_DEFAULT_US_RMSE},
    "KBOS": {**_DEFAULT_US_RMSE},
    # International stations — no HRRR
    "EGLC": {**_DEFAULT_INTL_RMSE},
    "SAEZ": {**_DEFAULT_INTL_RMSE},
    "LTAC": {**_DEFAULT_INTL_RMSE},
    "RKSS": {**_DEFAULT_INTL_RMSE},
}

# Wind direction buckets for regime classification
WIND_REGIMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "calm"]

# Cloud cover categories
CLOUD_CATEGORIES = ["clear", "scattered", "broken", "overcast", "unknown"]

# Bias table: (station, month, wind_regime, cloud_cover) -> mean_error_fahrenheit
# Positive = model too warm, Negative = model too cold
BIAS_TABLE: dict[tuple[str, str, str, str], float] = {}

# Standard deviation table for uncertainty estimation
BIAS_STD_TABLE: dict[tuple[str, str, str, str], float] = {}

# Sample count table (for confidence weighting)
BIAS_COUNT_TABLE: dict[tuple[str, str, str, str], int] = {}


def wind_direction_to_regime(wind_dir_degrees: int | float | None) -> str:
    """Convert wind direction in degrees to 8-point compass regime."""
    if wind_dir_degrees is None or wind_dir_degrees < 0:
        return "calm"
    d = wind_dir_degrees % 360
    if d < 22.5 or d >= 337.5:
        return "N"
    elif d < 67.5:
        return "NE"
    elif d < 112.5:
        return "E"
    elif d < 157.5:
        return "SE"
    elif d < 202.5:
        return "S"
    elif d < 247.5:
        return "SW"
    elif d < 292.5:
        return "W"
    else:
        return "NW"


def month_to_name(month: int) -> str:
    """Convert month number to 3-letter abbreviation."""
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month - 1]


def apply_bias_correction(
    forecast_temp: float,
    station: str,
    month: str | int,
    wind_regime: str = "calm",
    cloud_cover: str = "unknown",
    min_samples: int = 10,
) -> float:
    """Correct for systematic forecast bias.

    Returns bias-corrected temperature. Falls back to no correction
    if the specific bin has too few samples.
    """
    if isinstance(month, int):
        month = month_to_name(month)

    key = (station, month, wind_regime, cloud_cover)
    count = BIAS_COUNT_TABLE.get(key, 0)

    if count >= min_samples:
        bias = BIAS_TABLE.get(key, 0.0)
        return forecast_temp - bias

    # Fallback: try without cloud cover
    for fallback_cloud in CLOUD_CATEGORIES:
        fallback_key = (station, month, wind_regime, fallback_cloud)
        if BIAS_COUNT_TABLE.get(fallback_key, 0) >= min_samples:
            bias = BIAS_TABLE.get(fallback_key, 0.0)
            return forecast_temp - bias

    # Fallback: try station + month only (any wind/cloud)
    for wr in WIND_REGIMES:
        for cc in CLOUD_CATEGORIES:
            fallback_key = (station, month, wr, cc)
            if BIAS_COUNT_TABLE.get(fallback_key, 0) >= min_samples:
                bias = BIAS_TABLE.get(fallback_key, 0.0)
                return forecast_temp - bias

    return forecast_temp  # No correction available


def get_rmse(station: str, model: str, lead_time_days: int) -> float:
    """Lookup station/model/lead-time RMSE from STATION_RMSE table.

    BugFix H3: Returns sensible defaults instead of KeyError for unknown stations.
    CCG-bug-9: lead-time miss now falls back to nearest known bucket (typically
    lead=1) instead of a blind 7.0°F, which over-inflated σ on same-day markets
    and produced phantom buy_no edges on Polymarket binaries.
    """
    station_data = STATION_RMSE.get(station)
    if station_data is None:
        # Unknown station — use conservative defaults
        log.warning("get_rmse: no RMSE data for station %s, using defaults", station)
        station_data = _DEFAULT_INTL_RMSE
    model_data = station_data.get(model)
    if model_data is None:
        return 5.0  # safe middle-ground default
    value = model_data.get(lead_time_days)
    if value is None:
        # CCG-bug-9: fall back to the nearest known lead bucket rather than 7.0.
        # Prefer lead=1 (best-calibrated short-lead), then the minimum lead in table.
        fallback_lead = 1 if 1 in model_data else min(model_data.keys())
        value = model_data[fallback_lead]
        log.warning(
            "get_rmse: fallback for %s/%s/lead=%d → using lead=%d value %.1f",
            station, model, lead_time_days, fallback_lead, value,
        )
    return value


@dataclass
class BiasRecord:
    """A single data point for bias analysis."""
    station: str
    record_date: date
    forecast_high_f: float
    actual_high_f: float
    wind_regime: str = "calm"
    cloud_cover: str = "unknown"

    @property
    def error(self) -> float:
        """Forecast error: positive = model too warm."""
        return self.forecast_high_f - self.actual_high_f

    @property
    def month(self) -> str:
        return month_to_name(self.record_date.month)


class BiasAnalyzer:
    """Builds bias correction tables from historical data.

    Groups records by (station, month, wind_regime, cloud_cover) and
    computes mean error, std, and sample count for each bin.
    """

    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples

    def analyze(self, records: list[BiasRecord]) -> dict[str, dict]:
        """Compute bias statistics and populate global BIAS_TABLE.

        Returns a summary dict keyed by station with per-bin stats.
        """
        # Group by (station, month, wind_regime, cloud_cover)
        bins: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)

        for rec in records:
            key = (rec.station, rec.month, rec.wind_regime, rec.cloud_cover)
            bins[key].append(rec.error)

        # Compute stats and populate global tables
        summary: dict[str, list[dict]] = defaultdict(list)

        for key, errors in bins.items():
            station, month, wind, cloud = key
            n = len(errors)
            mean_err = sum(errors) / n
            std_err = math.sqrt(sum((e - mean_err) ** 2 for e in errors) / max(n - 1, 1))

            BIAS_TABLE[key] = mean_err
            BIAS_STD_TABLE[key] = std_err
            BIAS_COUNT_TABLE[key] = n

            summary[station].append({
                "month": month,
                "wind_regime": wind,
                "cloud_cover": cloud,
                "mean_error": round(mean_err, 2),
                "std_error": round(std_err, 2),
                "sample_size": n,
            })

        log.info(
            "BiasAnalyzer: computed %d bias bins from %d records across %d stations",
            len(bins), len(records), len(summary),
        )
        return dict(summary)

    def build_from_daily_records(
        self,
        daily_records: list,  # list[DailyRecord] from backtester
        wind_data: dict[tuple[str, date], int | None] | None = None,
        cloud_data: dict[tuple[str, date], str] | None = None,
    ) -> dict[str, dict]:
        """Convert backtester DailyRecords to BiasRecords and analyze.

        Args:
            daily_records: List of DailyRecord from TradingBacktester
            wind_data: Optional (station, date) -> wind_dir_degrees
            cloud_data: Optional (station, date) -> cloud_category
        """
        bias_records = []
        for dr in daily_records:
            wind_dir = wind_data.get((dr.station, dr.date)) if wind_data else None
            cloud = cloud_data.get((dr.station, dr.date), "unknown") if cloud_data else "unknown"

            bias_records.append(BiasRecord(
                station=dr.station,
                record_date=dr.date,
                forecast_high_f=dr.forecast_high_f,
                actual_high_f=dr.actual_high_f,
                wind_regime=wind_direction_to_regime(wind_dir),
                cloud_cover=cloud,
            ))
        return self.analyze(bias_records)

    def print_table(self, summary: dict[str, list[dict]]) -> str:
        """Print a formatted bias correction table."""
        lines = [
            "=" * 80,
            "STATION-SPECIFIC BIAS CORRECTION TABLE",
            "=" * 80,
            f"{'Station':<8} {'Month':<5} {'Wind':<6} {'Cloud':<10} {'MeanErr':>8} {'StdErr':>8} {'N':>5}",
            "-" * 80,
        ]
        for station in sorted(summary.keys()):
            for entry in sorted(summary[station], key=lambda x: (x["month"], x["wind_regime"])):
                if entry["sample_size"] >= self.min_samples:
                    lines.append(
                        f"{station:<8} {entry['month']:<5} {entry['wind_regime']:<6} "
                        f"{entry['cloud_cover']:<10} {entry['mean_error']:>+8.2f} "
                        f"{entry['std_error']:>8.2f} {entry['sample_size']:>5}"
                    )
        lines.append("=" * 80)
        report = "\n".join(lines)
        print(report)
        return report

    @staticmethod
    def clear_tables() -> None:
        """Clear all global bias tables (useful for testing)."""
        BIAS_TABLE.clear()
        BIAS_STD_TABLE.clear()
        BIAS_COUNT_TABLE.clear()
