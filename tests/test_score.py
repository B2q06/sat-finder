"""Tests for the score stage."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from satelite.db import (
    get_addresses_by_status,
    get_connection,
    insert_addresses,
    insert_city,
    insert_description,
    insert_image,
    update_address_status,
)
from satelite.stages.score import _parse_json_response, run_score


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_valid_json(self):
        data = {"qualified": True, "confidence": 0.85}
        assert _parse_json_response(json.dumps(data)) == data

    def test_json_in_markdown_code_block(self):
        text = '```json\n{"qualified": false, "confidence": 0.3}\n```'
        result = _parse_json_response(text)
        assert result == {"qualified": False, "confidence": 0.3}

    def test_json_with_surrounding_text(self):
        text = 'Analysis:\n{"qualified": true, "confidence": 0.9}\nEnd.'
        assert _parse_json_response(text) == {"qualified": True, "confidence": 0.9}

    def test_malformed_json_returns_none(self):
        assert _parse_json_response("this is not json") is None

    def test_empty_string_returns_none(self):
        assert _parse_json_response("") is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_SCORE = {
    "qualified": True,
    "confidence": 0.82,
    "reasoning": "High distress score with vacancy signs suggest motivated seller.",
    "wholesale_potential": "high",
    "estimated_discount_range": "25-35% below market",
}

SAMPLE_DESCRIPTION_DICT = {
    "roof_condition": "poor",
    "paint_condition": "fair",
    "yard_condition": "overgrown",
    "driveway_condition": "good",
    "windows_condition": "fair",
    "vacancy_signs": "yes",
    "maintenance_level": "neglected",
    "damage_list": ["peeling paint", "missing shingles"],
    "distress_score": 7,
    "summary": "Neglected property with deferred maintenance.",
}


def _setup_described_address(conn):
    """Insert city -> address -> image -> description, return (city_id, addr_id, desc_id)."""
    city_id = insert_city(conn, "Detroit", "Michigan", {
        "south": 42.25, "north": 42.45, "west": -83.29, "east": -82.91,
    })
    insert_addresses(conn, city_id, [{
        "osm_id": 8000,
        "osm_type": "node",
        "lat": 42.33,
        "lng": -83.05,
        "full_address": "100 Main St, Detroit, Michigan",
    }])
    addr = get_addresses_by_status(conn, city_id, "harvested")[0]
    update_address_status(conn, addr["id"], "described")

    image_id = insert_image(conn, addr["id"], {
        "pano_id": "pano_8000", "lat": 42.33, "lng": -83.05,
        "heading": 90.0, "image_path": "data/images/pano_8000.jpg",
    })
    desc_id = insert_description(
        conn, addr["id"], image_id,
        SAMPLE_DESCRIPTION_DICT,
        json.dumps(SAMPLE_DESCRIPTION_DICT),
        "test-model",
    )
    return city_id, addr["id"], desc_id


def _make_mock_message(response_text: str):
    content_block = SimpleNamespace(text=response_text)
    return SimpleNamespace(content=[content_block])


# ---------------------------------------------------------------------------
# run_score integration tests
# ---------------------------------------------------------------------------

class TestRunScore:
    @patch("satelite.stages.score.anthropic.Anthropic")
    @patch("satelite.stages.score.get_anthropic_key")
    def test_happy_path(self, mock_key, mock_anthropic_cls, config):
        conn = get_connection(config.pipeline.db_path)
        city_id, addr_id, desc_id = _setup_described_address(conn)
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message(
            json.dumps(SAMPLE_SCORE)
        )

        run_score(config, "Detroit")

        # Verify API called with text-only prompt
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        prompt_content = call_kwargs.kwargs["messages"][0]["content"]
        assert isinstance(prompt_content, str)

        # Verify address status updated to scored
        conn = get_connection(config.pipeline.db_path)
        scored = get_addresses_by_status(conn, city_id, "scored")
        assert len(scored) == 1
        assert scored[0]["id"] == addr_id

        # Verify score inserted
        row = conn.execute(
            "SELECT * FROM scores WHERE address_id=?", (addr_id,)
        ).fetchone()
        assert row is not None
        assert row["qualified"] == 1
        assert row["confidence"] == 0.82
        assert row["wholesale_potential"] == "high"
        conn.close()

    @patch("satelite.stages.score.anthropic.Anthropic")
    @patch("satelite.stages.score.get_anthropic_key")
    def test_prompt_includes_description_fields(self, mock_key, mock_anthropic_cls, config):
        conn = get_connection(config.pipeline.db_path)
        _setup_described_address(conn)
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message(
            json.dumps(SAMPLE_SCORE)
        )

        run_score(config, "Detroit")

        call_kwargs = mock_client.messages.create.call_args
        prompt = call_kwargs.kwargs["messages"][0]["content"]
        # Verify key fields from the description appear in the prompt
        assert "100 Main St, Detroit, Michigan" in prompt
        assert "poor" in prompt  # roof_condition
        assert "overgrown" in prompt  # yard_condition
        assert "neglected" in prompt  # maintenance_level
        assert "7" in prompt  # distress_score
        assert "peeling paint" in prompt  # from damage_list

    @patch("satelite.stages.score.anthropic.Anthropic")
    @patch("satelite.stages.score.get_anthropic_key")
    def test_api_error_skips(self, mock_key, mock_anthropic_cls, config):
        conn = get_connection(config.pipeline.db_path)
        city_id, addr_id, _ = _setup_described_address(conn)
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = anthropic.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )

        run_score(config, "Detroit")

        # Address stays described (not scored)
        conn = get_connection(config.pipeline.db_path)
        assert len(get_addresses_by_status(conn, city_id, "described")) == 1
        assert get_addresses_by_status(conn, city_id, "scored") == []
        conn.close()

    @patch("satelite.stages.score.anthropic.Anthropic")
    @patch("satelite.stages.score.get_anthropic_key")
    def test_missing_description_skips(self, mock_key, mock_anthropic_cls, config):
        conn = get_connection(config.pipeline.db_path)
        city_id = insert_city(conn, "Detroit", "Michigan", {
            "south": 42.25, "north": 42.45, "west": -83.29, "east": -82.91,
        })
        insert_addresses(conn, city_id, [{
            "osm_id": 9000, "osm_type": "node", "lat": 42.33, "lng": -83.05,
        }])
        addr = get_addresses_by_status(conn, city_id, "harvested")[0]
        update_address_status(conn, addr["id"], "described")
        # No description record inserted
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        run_score(config, "Detroit")

        mock_client.messages.create.assert_not_called()

    @patch("satelite.stages.score.anthropic.Anthropic")
    @patch("satelite.stages.score.get_anthropic_key")
    def test_unparseable_response_skips(self, mock_key, mock_anthropic_cls, config):
        conn = get_connection(config.pipeline.db_path)
        city_id, addr_id, _ = _setup_described_address(conn)
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message(
            "I cannot evaluate this property."
        )

        run_score(config, "Detroit")

        conn = get_connection(config.pipeline.db_path)
        assert len(get_addresses_by_status(conn, city_id, "described")) == 1
        assert get_addresses_by_status(conn, city_id, "scored") == []
        conn.close()

    @patch("satelite.stages.score.anthropic.Anthropic")
    @patch("satelite.stages.score.get_anthropic_key")
    def test_city_not_found_returns_early(self, mock_key, mock_anthropic_cls, config):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        run_score(config, "Nonexistent City")

        mock_client.messages.create.assert_not_called()

    @patch("satelite.stages.score.anthropic.Anthropic")
    @patch("satelite.stages.score.get_anthropic_key")
    def test_json_in_code_block(self, mock_key, mock_anthropic_cls, config):
        conn = get_connection(config.pipeline.db_path)
        city_id, _, _ = _setup_described_address(conn)
        conn.close()

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message(
            f"```json\n{json.dumps(SAMPLE_SCORE)}\n```"
        )

        run_score(config, "Detroit")

        conn = get_connection(config.pipeline.db_path)
        scored = get_addresses_by_status(conn, city_id, "scored")
        assert len(scored) == 1
        conn.close()
