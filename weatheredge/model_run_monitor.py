"""Phase 2 burst mode trigger — detects new NWP model runs.

Compares the most recent forecast timestamp per model against last-seen
timestamps to identify when a fresh GFS, HRRR, or ECMWF run has arrived.
Used by the burst mode scheduler to fire opportunistic trades in the
BURST_WINDOW_MINUTES window immediately after a new model run.
"""

from __future__ import annotations

import logging
from datetime import datetime

log = logging.getLogger(__name__)

# Synoptic hours at which major NWP models initialise (UTC)
SYNOPTIC_TIMES: list[str] = ["00Z", "06Z", "12Z", "18Z"]


class ModelRunMonitor:
    """Detects new GFS / HRRR / ECMWF model runs by comparing forecast timestamps.

    Maintains the last-seen ``forecast_timestamp`` for each model and returns
    the names of models whose timestamps have advanced since the previous call.

    Usage::

        monitor = ModelRunMonitor()
        while True:
            forecasts = fetch_current_forecasts()   # dict keyed by model name
            new_runs = monitor.check_for_new_run(forecasts)
            if new_runs:
                trigger_burst_mode(new_runs)
    """

    def __init__(self) -> None:
        # Last observed forecast_timestamp per model; None = never seen
        self._last_seen: dict[str, datetime | None] = {
            "gfs": None,
            "hrrr": None,
            "ecmwf": None,
        }
        log.info(
            "ModelRunMonitor initialised — tracking models: %s",
            list(self._last_seen),
        )

    def check_for_new_run(self, current_forecasts: dict) -> list[str] | None:
        """Compare current forecast timestamps against last-seen values.

        Args:
            current_forecasts: Mapping of model name to a forecast object or
                dict.  Each entry must expose a ``forecast_timestamp`` attribute
                or key (``datetime``).  Models not present in the dict are
                silently skipped.

        Returns:
            A list of model names whose ``forecast_timestamp`` has advanced
            beyond the previously recorded value, or ``None`` if no model has
            a new run.
        """
        new_runs: list[str] = []

        for model in self._last_seen:
            entry = current_forecasts.get(model)
            if entry is None:
                continue

            # Support both object attributes and dict keys
            if isinstance(entry, dict):
                ts: datetime | None = entry.get("forecast_timestamp")
            else:
                ts = getattr(entry, "forecast_timestamp", None)

            if ts is None:
                log.debug("Model %s: no forecast_timestamp present, skipping", model)
                continue

            last = self._last_seen[model]
            if last is None or ts > last:
                log.info(
                    "New model run detected: %s (prev=%s, now=%s)",
                    model,
                    last,
                    ts,
                )
                self._last_seen[model] = ts
                new_runs.append(model)

        return new_runs if new_runs else None
