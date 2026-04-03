"""Shared fixtures for satelite tests."""

from __future__ import annotations

import json

import pytest

from satelite.db import (
    get_connection,
    init_db,
    insert_addresses,
    insert_city,
    insert_description,
    insert_image,
    insert_score,
)
from satelite.models import SateliteConfig


@pytest.fixture()
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    yield db_path


@pytest.fixture()
def config(tmp_db, tmp_path):
    return SateliteConfig(
        pipeline={"db_path": tmp_db, "image_dir": str(tmp_path / "images")},
        overpass={"delay_between_queries": 0, "timeout": 30, "cell_size": 0.02},
    )


@pytest.fixture()
def populated_db(tmp_db):
    """Insert a city with addresses through the full pipeline for export/status tests."""
    conn = get_connection(tmp_db)

    city_id = insert_city(conn, "Detroit", "Michigan", {
        "south": 42.25, "north": 42.45, "west": -83.29, "east": -82.91,
    })

    addresses = [
        {
            "osm_id": 1000 + i,
            "osm_type": "node",
            "housenumber": str(100 + i),
            "street": "Main St",
            "city": "Detroit",
            "state": "Michigan",
            "postcode": "48201",
            "lat": 42.33 + i * 0.001,
            "lng": -83.05,
            "full_address": f"{100 + i} Main St, Detroit, Michigan",
        }
        for i in range(5)
    ]
    insert_addresses(conn, city_id, addresses)

    # Move 3 addresses through the full pipeline (captured -> described -> scored)
    rows = conn.execute(
        "SELECT id FROM addresses WHERE city_id=? LIMIT 3", (city_id,)
    ).fetchall()

    for row in rows:
        addr_id = row["id"]
        conn.execute("UPDATE addresses SET status='scored' WHERE id=?", (addr_id,))

        image_id = insert_image(conn, addr_id, {
            "pano_id": f"pano_{addr_id}",
            "lat": 42.33,
            "lng": -83.05,
            "heading": 90.0,
            "capture_date": "2024-06",
            "image_path": f"data/images/pano_{addr_id}.jpg",
        })

        desc_id = insert_description(conn, addr_id, image_id, {
            "roof_condition": "poor",
            "paint_condition": "fair",
            "yard_condition": "poor",
            "driveway_condition": "fair",
            "windows_condition": "fair",
            "vacancy_signs": "possible",
            "maintenance_level": "below_average",
            "damage_list": ["peeling paint", "missing shingles"],
            "distress_score": 7,
            "summary": "Property shows signs of deferred maintenance.",
        }, '{"raw": "test"}', "test-model")

        insert_score(conn, addr_id, desc_id, {
            "qualified": True,
            "confidence": 0.8,
            "reasoning": "High distress score with vacancy signs.",
            "wholesale_potential": "medium",
            "estimated_discount_range": "20-30%",
        }, "test-model")

    conn.commit()
    conn.close()
    return tmp_db, city_id
