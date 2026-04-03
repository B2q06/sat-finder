"""Tests for the harvest stage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from satelite.db import get_connection
from satelite.models import BBox, SateliteConfig
from satelite.stages.harvest import _parse_elements, _query_cell, run_harvest


# ---------------------------------------------------------------------------
# _parse_elements
# ---------------------------------------------------------------------------

class TestParseElements:
    def test_node_element(self):
        elements = [{
            "type": "node",
            "id": 123,
            "lat": 42.33,
            "lon": -83.05,
            "tags": {
                "addr:housenumber": "100",
                "addr:street": "Main St",
                "addr:city": "Detroit",
                "addr:state": "MI",
                "addr:postcode": "48201",
            },
        }]
        result = _parse_elements(elements, "Fallback City", "Fallback State")
        assert len(result) == 1
        addr = result[0]
        assert addr["osm_id"] == 123
        assert addr["osm_type"] == "node"
        assert addr["housenumber"] == "100"
        assert addr["street"] == "Main St"
        assert addr["city"] == "Detroit"
        assert addr["state"] == "MI"
        assert addr["postcode"] == "48201"
        assert addr["lat"] == 42.33
        assert addr["lng"] == -83.05
        assert addr["full_address"] == "100 Main St, Detroit, MI"

    def test_way_element_uses_center(self):
        elements = [{
            "type": "way",
            "id": 456,
            "center": {"lat": 42.34, "lon": -83.06},
            "tags": {
                "addr:housenumber": "200",
                "addr:street": "Oak Ave",
            },
        }]
        result = _parse_elements(elements, "Detroit", "Michigan")
        assert len(result) == 1
        addr = result[0]
        assert addr["lat"] == 42.34
        assert addr["lng"] == -83.06
        assert addr["city"] == "Detroit"
        assert addr["state"] == "Michigan"

    def test_skips_elements_missing_housenumber(self):
        elements = [{
            "type": "node",
            "id": 789,
            "lat": 42.0,
            "lon": -83.0,
            "tags": {"addr:street": "Elm St"},
        }]
        assert _parse_elements(elements, "City", "State") == []

    def test_skips_elements_missing_street(self):
        elements = [{
            "type": "node",
            "id": 790,
            "lat": 42.0,
            "lon": -83.0,
            "tags": {"addr:housenumber": "10"},
        }]
        assert _parse_elements(elements, "City", "State") == []

    def test_skips_way_without_center(self):
        elements = [{
            "type": "way",
            "id": 800,
            "tags": {
                "addr:housenumber": "300",
                "addr:street": "Pine Rd",
            },
        }]
        assert _parse_elements(elements, "City", "State") == []

    def test_uses_fallback_city_and_state(self):
        elements = [{
            "type": "node",
            "id": 900,
            "lat": 42.0,
            "lon": -83.0,
            "tags": {
                "addr:housenumber": "50",
                "addr:street": "Birch Ln",
            },
        }]
        result = _parse_elements(elements, "Fallback", "FB State")
        assert result[0]["city"] == "Fallback"
        assert result[0]["state"] == "FB State"


# ---------------------------------------------------------------------------
# _query_cell
# ---------------------------------------------------------------------------

class TestQueryCell:
    CELL = BBox(south=42.33, north=42.35, west=-83.05, east=-83.03)
    ENDPOINT = "https://overpass-api.de/api/interpreter"

    def _mock_session(self, status_code=200, json_data=None):
        session = MagicMock(spec=requests.Session)
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {"elements": []}
        session.post.return_value = resp
        return session

    def test_success(self):
        elements = [{"type": "node", "id": 1}]
        session = self._mock_session(200, {"elements": elements})
        result = _query_cell(session, self.ENDPOINT, self.CELL, 30)
        assert result == elements
        session.post.assert_called_once()

    @patch("satelite.stages.harvest.time.sleep")
    def test_429_backoff_then_success(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"elements": [{"id": 1}]}
        session.post.side_effect = [resp_429, resp_200]

        result = _query_cell(session, self.ENDPOINT, self.CELL, 30)
        assert result == [{"id": 1}]
        assert session.post.call_count == 2
        mock_sleep.assert_called_once_with(30)

    @patch("satelite.stages.harvest.time.sleep")
    def test_429_max_retries_exhausted(self, mock_sleep):
        session = MagicMock(spec=requests.Session)
        resp_429 = MagicMock()
        resp_429.status_code = 429
        session.post.return_value = resp_429

        result = _query_cell(session, self.ENDPOINT, self.CELL, 30)
        assert result == []
        assert session.post.call_count == 5

    def test_504_skips_cell(self):
        session = self._mock_session(504)
        result = _query_cell(session, self.ENDPOINT, self.CELL, 30)
        assert result == []
        session.post.assert_called_once()

    def test_other_http_error_raises(self):
        session = MagicMock(spec=requests.Session)
        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        session.post.return_value = resp

        with pytest.raises(requests.HTTPError):
            _query_cell(session, self.ENDPOINT, self.CELL, 30)


# ---------------------------------------------------------------------------
# run_harvest (end-to-end with mocks)
# ---------------------------------------------------------------------------

class TestRunHarvest:
    BBOX = BBox(south=42.33, north=42.35, west=-83.05, east=-83.03)
    OVERPASS_RESPONSE = {
        "elements": [
            {
                "type": "node",
                "id": 1001,
                "lat": 42.331,
                "lon": -83.041,
                "tags": {
                    "addr:housenumber": "101",
                    "addr:street": "Elm St",
                    "addr:city": "Detroit",
                    "addr:state": "MI",
                    "addr:postcode": "48201",
                },
            },
            {
                "type": "way",
                "id": 2001,
                "center": {"lat": 42.332, "lon": -83.042},
                "tags": {
                    "addr:housenumber": "202",
                    "addr:street": "Oak Ave",
                },
            },
        ],
    }

    @patch("satelite.stages.harvest.time.sleep")
    @patch("satelite.stages.harvest.requests.Session")
    @patch("satelite.stages.harvest.lookup_city_bbox")
    def test_inserts_addresses(self, mock_lookup, mock_session_cls, mock_sleep, config):
        mock_lookup.return_value = ("Detroit, Wayne County, Michigan, US", "Michigan", self.BBOX)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self.OVERPASS_RESPONSE
        mock_session_cls.return_value.post.return_value = mock_resp

        run_harvest(config, "Detroit")

        conn = get_connection(config.pipeline.db_path)
        rows = conn.execute("SELECT * FROM addresses").fetchall()
        conn.close()
        assert len(rows) >= 2

    @patch("satelite.stages.harvest.time.sleep")
    @patch("satelite.stages.harvest.requests.Session")
    @patch("satelite.stages.harvest.lookup_city_bbox")
    def test_limit_flag(self, mock_lookup, mock_session_cls, mock_sleep, config):
        mock_lookup.return_value = ("Detroit, Wayne County, Michigan, US", "Michigan", self.BBOX)

        # Return many elements per cell
        many_elements = {
            "elements": [
                {
                    "type": "node",
                    "id": 5000 + i,
                    "lat": 42.33 + i * 0.0001,
                    "lon": -83.04,
                    "tags": {
                        "addr:housenumber": str(i),
                        "addr:street": "Test St",
                    },
                }
                for i in range(50)
            ],
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = many_elements
        mock_session_cls.return_value.post.return_value = mock_resp

        run_harvest(config, "Detroit", limit=5)

        conn = get_connection(config.pipeline.db_path)
        count = conn.execute("SELECT COUNT(*) as cnt FROM addresses").fetchone()["cnt"]
        conn.close()
        assert count == 5

    @patch("satelite.stages.harvest.time.sleep")
    @patch("satelite.stages.harvest.requests.Session")
    @patch("satelite.stages.harvest.lookup_city_bbox")
    def test_creates_city_record(self, mock_lookup, mock_session_cls, mock_sleep, config):
        mock_lookup.return_value = ("Detroit, Wayne County, Michigan, US", "Michigan", self.BBOX)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"elements": []}
        mock_session_cls.return_value.post.return_value = mock_resp

        run_harvest(config, "Detroit")

        conn = get_connection(config.pipeline.db_path)
        row = conn.execute("SELECT * FROM cities WHERE name='Detroit'").fetchone()
        conn.close()
        assert row is not None
        assert row["state"] == "Michigan"

    @patch("satelite.stages.harvest.time.sleep")
    @patch("satelite.stages.harvest.requests.Session")
    @patch("satelite.stages.harvest.lookup_city_bbox")
    def test_deduplicates_on_rerun(self, mock_lookup, mock_session_cls, mock_sleep, config):
        mock_lookup.return_value = ("Detroit, Wayne County, Michigan, US", "Michigan", self.BBOX)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self.OVERPASS_RESPONSE
        mock_session_cls.return_value.post.return_value = mock_resp

        run_harvest(config, "Detroit")
        run_harvest(config, "Detroit")

        conn = get_connection(config.pipeline.db_path)
        count = conn.execute("SELECT COUNT(*) as cnt FROM addresses").fetchone()["cnt"]
        conn.close()
        # Same OSM IDs should not be duplicated
        assert count == 2
