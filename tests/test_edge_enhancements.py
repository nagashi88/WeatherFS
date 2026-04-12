"""Tests for Enhancement 1: Expanded Polymarket market parser."""

import pytest
from weatheredge.data.market_parser import (
    parse_polymarket_title,
    CITY_TO_STATION,
    US_STATIONS,
    ParsedMarket,
)


class TestExpandedCityMapping:
    """Enhancement 1: All city variants map correctly."""

    def test_nyc_variants(self):
        assert CITY_TO_STATION["nyc"] == "KNYC"
        assert CITY_TO_STATION["new york city"] == "KNYC"
        assert CITY_TO_STATION["new york"] == "KNYC"

    def test_new_cities(self):
        assert CITY_TO_STATION["buenos aires"] == "SAEZ"
        assert CITY_TO_STATION["ankara"] == "LTAC"
        assert CITY_TO_STATION["seoul"] == "RKSS"
        assert CITY_TO_STATION["atlanta"] == "KATL"
        assert CITY_TO_STATION["dallas"] == "KDFW"
        assert CITY_TO_STATION["phoenix"] == "KPHX"
        assert CITY_TO_STATION["houston"] == "KIAH"

    def test_us_stations_set(self):
        assert "KNYC" in US_STATIONS
        assert "KMDW" in US_STATIONS
        assert "KATL" in US_STATIONS
        assert "KDFW" in US_STATIONS
        assert "KPHX" in US_STATIONS
        assert "KIAH" in US_STATIONS
        # Non-US should NOT be in the set
        assert "EGLC" not in US_STATIONS
        assert "SAEZ" not in US_STATIONS
        assert "LTAC" not in US_STATIONS
        assert "RKSS" not in US_STATIONS


class TestPolymarketTitleParsing:
    """Enhancement 1: All specified title patterns parse correctly."""

    def test_bucket_pattern(self):
        title = "Highest temperature in NYC on April 4? - between 50-51°F"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KNYC"
        assert result.date == "2026-04-04"
        assert result.threshold_low == 50.0
        assert result.threshold_high == 51.0

    def test_exceed_pattern(self):
        title = "Will the high temperature in Chicago exceed 60°F on April 3?"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KMDW"
        assert result.date == "2026-04-03"
        assert result.threshold == 60.0
        assert result.direction == "above"

    def test_or_above_pattern(self):
        title = "Highest temperature in Denver on April 4? - 55°F or above"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KDEN"
        assert result.date == "2026-04-04"
        assert result.threshold == 55.0
        assert result.direction == "above"

    def test_or_below_pattern(self):
        title = "Highest temperature in Miami on March 30? - 43°F or below"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KMIA"
        assert result.threshold == 43.0
        assert result.direction == "below"

    def test_new_york_city_variant(self):
        title = "Highest temperature in New York City on April 4? - between 70-71°F"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KNYC"

    def test_new_york_variant(self):
        title = "Highest temperature in New York on April 4? - 65°F or above"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KNYC"

    def test_new_city_atlanta(self):
        title = "Highest temperature in Atlanta on April 5? - between 72-73°F"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KATL"

    def test_new_city_houston(self):
        title = "Will the high temperature in Houston exceed 85°F on April 6?"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KIAH"

    def test_new_city_dallas(self):
        title = "Highest temperature in Dallas on April 4? - 78°F or above"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KDFW"

    def test_new_city_phoenix(self):
        title = "Highest temperature in Phoenix on April 4? - 90°F or above"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KPHX"

    def test_new_city_buenos_aires(self):
        title = "Highest temperature in Buenos Aires on April 4? - between 60-61°F"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "SAEZ"

    def test_new_city_seoul(self):
        title = "Highest temperature in Seoul on April 4? - 55°F or above"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "RKSS"

    def test_unknown_city_returns_none(self):
        title = "Highest temperature in Timbuktu on April 4? - between 60-61°F"
        result = parse_polymarket_title(title)
        assert result is None

    def test_no_date_returns_none(self):
        title = "Highest temperature in NYC? - between 50-51°F"
        result = parse_polymarket_title(title)
        assert result is None

    def test_daily_low_detection(self):
        title = "Low temperature in NYC on April 4? - 43°F or below"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.market_type == "daily_low"
