"""Tests for satelite.stages.capture — panorama cropping and address processing."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from satelite.models import SateliteConfig
from satelite.stages.capture import _crop_panorama_facing, _process_address


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panorama_image(width: int = 3600, height: int = 1800) -> Image.Image:
    """Create a synthetic equirectangular panorama (solid color, easy to verify dims)."""
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def _make_config(**overrides) -> SateliteConfig:
    cfg = SateliteConfig()
    for k, v in overrides.items():
        parts = k.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


def _make_fake_pano(lat: float = 42.33, lon: float = -83.05, pano_id: str = "abc123"):
    return SimpleNamespace(id=pano_id, lat=lat, lon=lon, heading=90.0, date=None)


# ---------------------------------------------------------------------------
# _crop_panorama_facing — dimension and wrapping tests
# ---------------------------------------------------------------------------

class TestCropPanoramaFacing:
    def test_output_dimensions_default_fov(self):
        img = _make_panorama_image(3600, 1800)
        result = _crop_panorama_facing(img, target_bearing_deg=180.0)
        # FOV = 90 deg => 3600 * (90/360) = 900 wide
        # Height after trim: 1800 * 0.5 = 900
        assert result.size == (900, 900)

    def test_output_dimensions_custom_fov(self):
        img = _make_panorama_image(3600, 1800)
        result = _crop_panorama_facing(img, target_bearing_deg=180.0, fov_degrees=120)
        expected_w = int((120 / 360.0) * 3600)  # 1200
        expected_h = int(1800 * 0.5)  # 900
        assert result.size == (expected_w, expected_h)

    def test_bearing_zero_wraps_left_edge(self):
        """Bearing 0 means center_x=0, so left goes negative — triggers wrap."""
        img = _make_panorama_image(3600, 1800)
        result = _crop_panorama_facing(img, target_bearing_deg=0.0)
        assert result.size[0] == 900

    def test_bearing_near_360_wraps_right_edge(self):
        """Bearing close to 360 means right side overflows — triggers wrap."""
        img = _make_panorama_image(3600, 1800)
        result = _crop_panorama_facing(img, target_bearing_deg=350.0)
        assert result.size[0] == 900

    def test_bearing_180_no_wrapping(self):
        """Middle of the image — no wrapping needed."""
        img = _make_panorama_image(3600, 1800)
        result = _crop_panorama_facing(img, target_bearing_deg=180.0)
        assert result.size[0] == 900

    def test_sky_and_road_trimmed(self):
        """Top 25% and bottom 25% should be removed."""
        img = _make_panorama_image(3600, 1200)
        result = _crop_panorama_facing(img, target_bearing_deg=90.0)
        # Original height 1200 => after trim: 1200 * 0.5 = 600
        assert result.size[1] == 600

    def test_small_image(self):
        """Ensure it works on a small image without crashing."""
        img = _make_panorama_image(360, 180)
        result = _crop_panorama_facing(img, target_bearing_deg=45.0)
        expected_w = int((90 / 360.0) * 360)  # 90
        expected_h = int(180 * 0.5)  # 90
        assert result.size == (expected_w, expected_h)


# ---------------------------------------------------------------------------
# _process_address — success path
# ---------------------------------------------------------------------------

class TestProcessAddressSuccess:
    @patch("satelite.stages.capture.streetview")
    def test_captures_and_returns_record(self, mock_sv, tmp_path):
        fake_pano = _make_fake_pano()
        mock_sv.find_panorama.return_value = fake_pano
        mock_sv.get_panorama_async = AsyncMock(return_value=_make_panorama_image())

        config = _make_config()
        addr = {
            "id": 1,
            "lat": 42.34,
            "lng": -83.04,
            "full_address": "123 Main St, Detroit, Michigan",
        }
        sem = asyncio.Semaphore(5)

        result = asyncio.run(
            _process_address(addr, config, tmp_path, sem, session=None)
        )

        assert isinstance(result, tuple)
        assert result[0] == "captured"
        assert result[1] == 1  # addr_id
        pano_rec = result[2]
        assert pano_rec["pano_id"] == "abc123"
        assert pano_rec["lat"] == fake_pano.lat
        assert pano_rec["lng"] == fake_pano.lon

        # Verify file was saved
        addr_hash = hashlib.sha256(b"123 Main St, Detroit, Michigan").hexdigest()[:12]
        saved = tmp_path / f"{addr_hash}.jpg"
        assert saved.exists()

    @patch("satelite.stages.capture.streetview")
    def test_capture_date_included_when_present(self, mock_sv, tmp_path):
        fake_pano = _make_fake_pano()
        fake_pano.date = "2023-06-15"
        mock_sv.find_panorama.return_value = fake_pano
        mock_sv.get_panorama_async = AsyncMock(return_value=_make_panorama_image())

        config = _make_config()
        addr = {"id": 2, "lat": 42.0, "lng": -83.0, "full_address": "456 Oak Ave"}
        sem = asyncio.Semaphore(5)

        result = asyncio.run(
            _process_address(addr, config, tmp_path, sem, session=None)
        )
        assert result[2]["capture_date"] == "2023-06-15"


# ---------------------------------------------------------------------------
# _process_address — skipped paths
# ---------------------------------------------------------------------------

class TestProcessAddressSkipped:
    @patch("satelite.stages.capture.streetview")
    def test_no_panorama_found(self, mock_sv, tmp_path):
        mock_sv.find_panorama.return_value = None

        config = _make_config()
        addr = {"id": 3, "lat": 42.0, "lng": -83.0, "full_address": "No Pano Lane"}
        sem = asyncio.Semaphore(5)

        result = asyncio.run(
            _process_address(addr, config, tmp_path, sem, session=None)
        )
        assert result == "skipped"

    @patch("satelite.stages.capture.streetview")
    def test_find_panorama_raises(self, mock_sv, tmp_path):
        mock_sv.find_panorama.side_effect = RuntimeError("Google API flaky")

        config = _make_config()
        addr = {"id": 4, "lat": 42.0, "lng": -83.0, "full_address": "Error St"}
        sem = asyncio.Semaphore(5)

        result = asyncio.run(
            _process_address(addr, config, tmp_path, sem, session=None)
        )
        assert result == "skipped"

    @patch("satelite.stages.capture.streetview")
    def test_get_panorama_async_raises(self, mock_sv, tmp_path):
        fake_pano = _make_fake_pano()
        mock_sv.find_panorama.return_value = fake_pano
        mock_sv.get_panorama_async = AsyncMock(side_effect=ConnectionError("timeout"))

        config = _make_config()
        addr = {"id": 5, "lat": 42.0, "lng": -83.0, "full_address": "Timeout Blvd"}
        sem = asyncio.Semaphore(5)

        result = asyncio.run(
            _process_address(addr, config, tmp_path, sem, session=None)
        )
        assert result == "skipped"

    @patch("satelite.stages.capture.streetview")
    def test_get_panorama_async_returns_none(self, mock_sv, tmp_path):
        fake_pano = _make_fake_pano()
        mock_sv.find_panorama.return_value = fake_pano
        mock_sv.get_panorama_async = AsyncMock(return_value=None)

        config = _make_config()
        addr = {"id": 6, "lat": 42.0, "lng": -83.0, "full_address": "Null Image Dr"}
        sem = asyncio.Semaphore(5)

        result = asyncio.run(
            _process_address(addr, config, tmp_path, sem, session=None)
        )
        assert result == "skipped"
