"""Position tracker for WeatherEdge.

Wraps open position tracking with city-grouped exposure queries.
Addendum Gap 11: same-city correlation grouping informs exposure limits.
"""

from __future__ import annotations


class PositionTracker:
    """Tracks open positions grouped by city for exposure management."""

    def __init__(self) -> None:
        self._positions: list[dict] = []

    def add_position(self, position: dict) -> None:
        """Add a filled position to the tracker."""
        self._positions.append(position)

    def remove_position(self, market_id: str) -> None:
        """Remove all positions for a resolved market."""
        self._positions = [p for p in self._positions if p.get("market_id") != market_id]

    def get_open_positions(self) -> list[dict]:
        """Return all open positions."""
        return [p for p in self._positions if p.get("status") == "open"]

    def get_positions_by_city(self) -> dict[str, list[dict]]:
        """Return open positions grouped by city."""
        result: dict[str, list[dict]] = {}
        for pos in self.get_open_positions():
            city = pos.get("city", "unknown")
            result.setdefault(city, []).append(pos)
        return result

    def get_city_exposure(self, city: str) -> float:
        """Total cost of open positions in a city."""
        return sum(
            p.get("cost", 0.0)
            for p in self.get_open_positions()
            if p.get("city") == city
        )
