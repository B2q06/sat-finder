"""Tests for satelite.geo — bearing calculation, city slug, Nominatim lookup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from satelite.geo import calculate_bearing, lookup_city_bbox, slugify_city
from satelite.models import BBox


# ---------------------------------------------------------------------------
# calculate_bearing
# ---------------------------------------------------------------------------

class TestCalculateBearing:
    def test_due_north(self):
        # Moving north: same lng, increasing lat
        bearing = calculate_bearing(42.0, -83.0, 43.0, -83.0)
        assert abs(bearing - 0.0) < 1.0

    def test_due_east(self):
        bearing = calculate_bearing(0.0, 0.0, 0.0, 1.0)
        assert abs(bearing - 90.0) < 1.0

    def test_due_south(self):
        bearing = calculate_bearing(43.0, -83.0, 42.0, -83.0)
        assert abs(bearing - 180.0) < 1.0

    def test_due_west(self):
        bearing = calculate_bearing(0.0, 1.0, 0.0, 0.0)
        assert abs(bearing - 270.0) < 1.0

    def test_northeast(self):
        bearing = calculate_bearing(0.0, 0.0, 1.0, 1.0)
        assert 0.0 < bearing < 90.0

    def test_result_always_0_to_360(self):
        bearing = calculate_bearing(42.0, -83.0, 42.0, -84.0)  # west
        assert 0.0 <= bearing < 360.0

    def test_same_point_returns_zero(self):
        bearing = calculate_bearing(42.0, -83.0, 42.0, -83.0)
        assert bearing == 0.0


# ---------------------------------------------------------------------------
# slugify_city
# ---------------------------------------------------------------------------

class TestSlugifyCity:
    def test_basic(self):
        assert slugify_city("Detroit, MI") == "detroit-mi"

    def test_no_special_chars(self):
        assert slugify_city("Detroit") == "detroit"

    def test_spaces_and_punctuation(self):
        assert slugify_city("New York City, NY") == "new-york-city-ny"

    def test_strips_trailing_hyphens(self):
        assert slugify_city("  Detroit!!!  ") == "detroit"

    def test_multiple_spaces(self):
        assert slugify_city("San   Francisco") == "san-francisco"

    def test_already_slug(self):
        assert slugify_city("detroit-mi") == "detroit-mi"


# ---------------------------------------------------------------------------
# lookup_city_bbox
# ---------------------------------------------------------------------------

class TestLookupCityBbox:
    @patch("satelite.geo.time.sleep")  # skip the rate-limit sleep
    @patch("satelite.geo.requests.get")
    def test_successful_lookup(self, mock_get, _mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "boundingbox": ["42.2550", "42.4502", "-83.2880", "-82.9103"],
                "display_name": "Detroit, Wayne County, Michigan, US",
            }
        ]
        mock_get.return_value = mock_resp

        display, state, bbox = lookup_city_bbox("Detroit, MI")

        assert display == "Detroit, Wayne County, Michigan, US"
        assert state == "Michigan"
        assert isinstance(bbox, BBox)
        assert abs(bbox.south - 42.2550) < 0.001
        assert abs(bbox.north - 42.4502) < 0.001
        assert abs(bbox.west - (-83.2880)) < 0.001
        assert abs(bbox.east - (-82.9103)) < 0.001

    @patch("satelite.geo.time.sleep")
    @patch("satelite.geo.requests.get")
    def test_city_not_found_raises(self, mock_get, _mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="City not found"):
            lookup_city_bbox("Nonexistentville, ZZ")

    @patch("satelite.geo.time.sleep")
    @patch("satelite.geo.requests.get")
    def test_short_display_name_state_is_none(self, mock_get, _mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "boundingbox": ["0.0", "1.0", "0.0", "1.0"],
                "display_name": "Somewhere, Country",
            }
        ]
        mock_get.return_value = mock_resp

        display, state, bbox = lookup_city_bbox("Somewhere")
        # Only 2 parts — state extraction needs >= 3
        assert state is None
