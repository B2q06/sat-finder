"""Geographic utilities: Nominatim lookup, bearing calculation, city slug."""

from __future__ import annotations

import math
import re
import time

import requests

from satelite.models import BBox

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_HEADERS = {"User-Agent": "satelite-lead-finder/0.1"}


def lookup_city_bbox(city_query: str) -> tuple[str, str | None, BBox]:
    """Look up a city's bounding box via Nominatim.

    Returns (display_name, state, BBox).
    """
    resp = requests.get(
        _NOMINATIM_URL,
        params={"q": city_query, "format": "json", "limit": 1},
        headers=_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"City not found: {city_query}")

    result = results[0]
    bb = result["boundingbox"]  # [minlat, maxlat, minlon, maxlon]
    display = result.get("display_name", city_query)

    # Extract state from display_name (e.g. "Detroit, Wayne County, Michigan, US")
    parts = [p.strip() for p in display.split(",")]
    state = None
    if len(parts) >= 3:
        state = parts[-2]  # Usually the state

    bbox = BBox(
        south=float(bb[0]),
        north=float(bb[1]),
        west=float(bb[2]),
        east=float(bb[3]),
    )
    time.sleep(1)  # Respect Nominatim rate limit
    return display, state, bbox


def calculate_bearing(from_lat: float, from_lng: float, to_lat: float, to_lng: float) -> float:
    """Calculate bearing in degrees (0-360) from one point to another."""
    lat1 = math.radians(from_lat)
    lat2 = math.radians(to_lat)
    diff_lng = math.radians(to_lng - from_lng)

    x = math.sin(diff_lng) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(diff_lng)

    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360


def slugify_city(city_name: str) -> str:
    """Convert city name to filesystem-safe slug. 'Detroit, MI' -> 'detroit-mi'."""
    slug = city_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")
