"""Tests for the describe stage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from satelite.db import (
    get_addresses_by_status,
    get_connection,
    insert_addresses,
    insert_city,
    insert_image,
    update_address_status,
)
from satelite.stages.describe import _parse_json_response, run_describe


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_valid_json(self):
        data = {"roof_condition": "poor", "distress_score": 7}
        assert _parse_json_response(json.dumps(data)) == data

    def test_json_in_markdown_code_block(self):
        text = '```json\n{"roof_condition": "good", "distress_score": 3}\n```'
        result = _parse_json_response(text)
        assert result == {"roof_condition": "good", "distress_score": 3}

    def test_json_in_backticks_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        assert _parse_json_response(text) == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'Here is my analysis:\n{"distress_score": 5}\nHope that helps!'
        assert _parse_json_response(text) == {"distress_score": 5}

    def test_malformed_json_returns_none(self):
        assert _parse_json_response("not json at all") is None

    def test_empty_string_returns_none(self):
        assert _parse_json_response("") is None

    def test_partial_json_returns_none(self):
        assert _parse_json_response('{"key": "value"') is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DESCRIPTION = {
    "roof_condition": "poor",
    "paint_condition": "fair",
    "yard_condition": "overgrown",
    "driveway_condition": "good",
    "windows_condition": "fair",
    "vacancy_signs": "yes",
    "maintenance_level": "neglected",
    "damage_list": ["peeling paint"],
    "distress_score": 7,
    "summary": "Neglected property with deferred maintenance.",
}


def _setup_captured_address(conn, tmp_path):
    """Insert a city, address, and image, returning (city_id, address_id, image_path)."""
    city_id = insert_city(conn, "Detroit", "Michigan", {
        "south": 42.25, "north": 42.45, "west": -83.29, "east": -82.91,
    })
    insert_addresses(conn, city_id, [{
        "osm_id": 5000,
        "osm_type": "node",
        "lat": 42.33,
        "lng": -83.05,
        "full_address": "100 Main St, Detroit, Michigan",
    }])
    addr = get_addresses_by_status(conn, city_id, "harvested")[0]
    update_address_status(conn, addr["id"], "captured")

    # Create a real image file
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    img_file = img_dir / "test_pano.jpg"
    img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header

    insert_image(conn, addr["id"], {
        "pano_id": "pano_5000",
        "lat": 42.33,
        "lng": -83.05,
        "heading": 90.0,
        "image_path": str(img_file),
    })
    return city_id, addr["id"], img_file


def _make_mock_message(response_text: str):
    """Build a mock anthropic Message return value."""
    content_block = SimpleNamespace(text=response_text)
    return SimpleNamespace(content=[content_block])


# ---------------------------------------------------------------------------
# run_describe integration tests
# ---------------------------------------------------------------------------

class TestRunDescribe:
    @patch("satelite.stages.describe.anthropic.Anthropic")
    @patch("satelite.stages.describe.get_anthropic_key")
    def test_happy_path(self, mock_key, mock_anthropic_cls, config, tmp_path):
        conn = get_connection(config.pipeline.db_path)
        city_id, addr_id, img_file = _setup_captured_address(conn, tmp_path)
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message(
            json.dumps(SAMPLE_DESCRIPTION)
        )

        run_describe(config, "Detroit")

        # Verify API was called
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        # Should include image content
        msg_content = call_kwargs.kwargs["messages"][0]["content"]
        assert msg_content[0]["type"] == "image"
        assert msg_content[1]["type"] == "text"

        # Verify address status updated
        conn = get_connection(config.pipeline.db_path)
        described = get_addresses_by_status(conn, city_id, "described")
        assert len(described) == 1
        assert described[0]["id"] == addr_id

        # Verify description inserted
        row = conn.execute(
            "SELECT * FROM descriptions WHERE address_id=?", (addr_id,)
        ).fetchone()
        assert row is not None
        assert row["roof_condition"] == "poor"
        assert row["distress_score"] == 7
        conn.close()

    @patch("satelite.stages.describe.anthropic.Anthropic")
    @patch("satelite.stages.describe.get_anthropic_key")
    def test_json_in_code_block(self, mock_key, mock_anthropic_cls, config, tmp_path):
        conn = get_connection(config.pipeline.db_path)
        _setup_captured_address(conn, tmp_path)
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message(
            f"```json\n{json.dumps(SAMPLE_DESCRIPTION)}\n```"
        )

        run_describe(config, "Detroit")

        conn = get_connection(config.pipeline.db_path)
        described = get_addresses_by_status(conn, 1, "described")
        assert len(described) == 1
        conn.close()

    @patch("satelite.stages.describe.anthropic.Anthropic")
    @patch("satelite.stages.describe.get_anthropic_key")
    def test_api_error_skips(self, mock_key, mock_anthropic_cls, config, tmp_path):
        conn = get_connection(config.pipeline.db_path)
        city_id, addr_id, _ = _setup_captured_address(conn, tmp_path)
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = anthropic.APIError(
            message="rate limit",
            request=MagicMock(),
            body=None,
        )

        run_describe(config, "Detroit")

        # Address should still be 'captured' (not updated)
        conn = get_connection(config.pipeline.db_path)
        captured = get_addresses_by_status(conn, city_id, "captured")
        assert len(captured) == 1
        assert get_addresses_by_status(conn, city_id, "described") == []
        conn.close()

    @patch("satelite.stages.describe.anthropic.Anthropic")
    @patch("satelite.stages.describe.get_anthropic_key")
    def test_missing_image_file_skips(self, mock_key, mock_anthropic_cls, config, tmp_path):
        conn = get_connection(config.pipeline.db_path)
        city_id = insert_city(conn, "Detroit", "Michigan", {
            "south": 42.25, "north": 42.45, "west": -83.29, "east": -82.91,
        })
        insert_addresses(conn, city_id, [{
            "osm_id": 6000, "osm_type": "node", "lat": 42.33, "lng": -83.05,
        }])
        addr = get_addresses_by_status(conn, city_id, "harvested")[0]
        update_address_status(conn, addr["id"], "captured")
        # Insert image pointing to nonexistent file
        insert_image(conn, addr["id"], {
            "pano_id": "pano_ghost", "lat": 42.33, "lng": -83.05,
            "heading": 90.0, "image_path": "/nonexistent/path.jpg",
        })
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        run_describe(config, "Detroit")

        # API should never be called
        mock_client.messages.create.assert_not_called()

        # Address stays captured
        conn = get_connection(config.pipeline.db_path)
        assert len(get_addresses_by_status(conn, city_id, "captured")) == 1
        conn.close()

    @patch("satelite.stages.describe.anthropic.Anthropic")
    @patch("satelite.stages.describe.get_anthropic_key")
    def test_no_image_record_skips(self, mock_key, mock_anthropic_cls, config, tmp_path):
        conn = get_connection(config.pipeline.db_path)
        city_id = insert_city(conn, "Detroit", "Michigan", {
            "south": 42.25, "north": 42.45, "west": -83.29, "east": -82.91,
        })
        insert_addresses(conn, city_id, [{
            "osm_id": 7000, "osm_type": "node", "lat": 42.33, "lng": -83.05,
        }])
        addr = get_addresses_by_status(conn, city_id, "harvested")[0]
        update_address_status(conn, addr["id"], "captured")
        # No image inserted
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        run_describe(config, "Detroit")

        mock_client.messages.create.assert_not_called()

    @patch("satelite.stages.describe.anthropic.Anthropic")
    @patch("satelite.stages.describe.get_anthropic_key")
    def test_unparseable_response_skips(self, mock_key, mock_anthropic_cls, config, tmp_path):
        conn = get_connection(config.pipeline.db_path)
        city_id, addr_id, _ = _setup_captured_address(conn, tmp_path)
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message(
            "I cannot analyze this image."
        )

        run_describe(config, "Detroit")

        conn = get_connection(config.pipeline.db_path)
        assert len(get_addresses_by_status(conn, city_id, "captured")) == 1
        assert get_addresses_by_status(conn, city_id, "described") == []
        conn.close()

    @patch("satelite.stages.describe.anthropic.Anthropic")
    @patch("satelite.stages.describe.get_anthropic_key")
    def test_city_not_found_returns_early(self, mock_key, mock_anthropic_cls, config):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        run_describe(config, "Nonexistent City")

        mock_client.messages.create.assert_not_called()
