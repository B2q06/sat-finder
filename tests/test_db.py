"""Tests for the database layer."""

from __future__ import annotations

import json

from satelite.db import (
    get_connection,
    get_addresses_by_status,
    get_city_id,
    get_pipeline_stats,
    get_qualified_leads,
    init_db,
    insert_addresses,
    insert_city,
    insert_description,
    insert_image,
    insert_score,
    update_address_status,
)


class TestInitDb:
    def test_creates_all_tables(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = get_connection(db_path)
        tables = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert {"cities", "addresses", "images", "descriptions", "scores"} <= tables

    def test_idempotent(self, tmp_db):
        # Running init twice should not raise
        init_db(tmp_db)
        conn = get_connection(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        assert len(tables) >= 5


class TestInsertCity:
    def test_insert_and_retrieve(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)
        assert city_id > 0

        retrieved = get_city_id(conn, "Detroit", "Michigan")
        assert retrieved == city_id
        conn.close()

    def test_upsert_on_conflict(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox1 = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        bbox2 = {"south": 42.1, "north": 42.6, "west": -83.4, "east": -82.9}
        id1 = insert_city(conn, "Detroit", "Michigan", bbox1)
        id2 = insert_city(conn, "Detroit", "Michigan", bbox2)
        assert id1 == id2

        row = conn.execute("SELECT bbox_south FROM cities WHERE id=?", (id1,)).fetchone()
        assert row["bbox_south"] == 42.1
        conn.close()

    def test_get_city_id_not_found(self, tmp_db):
        conn = get_connection(tmp_db)
        assert get_city_id(conn, "Nonexistent") is None
        conn.close()

    def test_get_city_id_without_state(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)
        assert get_city_id(conn, "Detroit") == city_id
        conn.close()


class TestInsertAddresses:
    def test_bulk_insert(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)

        addresses = [
            {"osm_id": 100, "osm_type": "node", "lat": 42.33, "lng": -83.05,
             "housenumber": "100", "street": "Main St", "city": "Detroit", "state": "MI"},
            {"osm_id": 101, "osm_type": "node", "lat": 42.34, "lng": -83.06,
             "housenumber": "101", "street": "Main St", "city": "Detroit", "state": "MI"},
        ]
        inserted = insert_addresses(conn, city_id, addresses)
        assert inserted == 2
        conn.close()

    def test_duplicate_handling(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)

        addresses = [
            {"osm_id": 200, "osm_type": "node", "lat": 42.33, "lng": -83.05},
        ]
        assert insert_addresses(conn, city_id, addresses) == 1
        # Second insert of same osm_id+osm_type should be skipped
        assert insert_addresses(conn, city_id, addresses) == 0
        conn.close()

    def test_default_status_is_harvested(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)

        insert_addresses(conn, city_id, [
            {"osm_id": 300, "osm_type": "node", "lat": 42.33, "lng": -83.05},
        ])
        rows = get_addresses_by_status(conn, city_id, "harvested")
        assert len(rows) == 1
        assert rows[0]["status"] == "harvested"
        conn.close()


class TestGetAddressesByStatus:
    def _seed(self, tmp_db, count=5):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)
        addresses = [
            {"osm_id": 400 + i, "osm_type": "node", "lat": 42.33 + i * 0.001,
             "lng": -83.05, "full_address": f"{i} Main St"}
            for i in range(count)
        ]
        insert_addresses(conn, city_id, addresses)
        return conn, city_id

    def test_filters_by_status(self, tmp_db):
        conn, city_id = self._seed(tmp_db, 3)
        # All start as 'harvested'
        rows = get_addresses_by_status(conn, city_id, "harvested")
        assert len(rows) == 3
        assert get_addresses_by_status(conn, city_id, "captured") == []
        conn.close()

    def test_limit(self, tmp_db):
        conn, city_id = self._seed(tmp_db, 5)
        rows = get_addresses_by_status(conn, city_id, "harvested", limit=2)
        assert len(rows) == 2
        conn.close()


class TestUpdateAddressStatus:
    def test_update(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)
        insert_addresses(conn, city_id, [
            {"osm_id": 500, "osm_type": "node", "lat": 42.33, "lng": -83.05},
        ])
        addr = get_addresses_by_status(conn, city_id, "harvested")[0]

        update_address_status(conn, addr["id"], "captured")

        assert get_addresses_by_status(conn, city_id, "harvested") == []
        captured = get_addresses_by_status(conn, city_id, "captured")
        assert len(captured) == 1
        assert captured[0]["id"] == addr["id"]
        conn.close()


class TestInsertImage:
    def test_insert_and_return_id(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)
        insert_addresses(conn, city_id, [
            {"osm_id": 600, "osm_type": "node", "lat": 42.33, "lng": -83.05},
        ])
        addr = get_addresses_by_status(conn, city_id, "harvested")[0]

        image_id = insert_image(conn, addr["id"], {
            "pano_id": "pano_test",
            "lat": 42.33,
            "lng": -83.05,
            "heading": 180.0,
            "capture_date": "2024-06",
            "image_path": "data/images/test.jpg",
        })
        assert image_id > 0

        row = conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        assert row["pano_id"] == "pano_test"
        assert row["heading"] == 180.0
        conn.close()


class TestInsertDescription:
    def test_insert_and_return_id(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)
        insert_addresses(conn, city_id, [
            {"osm_id": 700, "osm_type": "node", "lat": 42.33, "lng": -83.05},
        ])
        addr = get_addresses_by_status(conn, city_id, "harvested")[0]
        image_id = insert_image(conn, addr["id"], {
            "pano_id": "pano_d", "lat": 42.33, "lng": -83.05,
            "heading": 90.0, "image_path": "img.jpg",
        })

        desc_dict = {
            "roof_condition": "poor",
            "paint_condition": "fair",
            "yard_condition": "overgrown",
            "driveway_condition": "good",
            "windows_condition": "fair",
            "vacancy_signs": "yes",
            "maintenance_level": "neglected",
            "damage_list": ["cracked foundation", "missing gutters"],
            "distress_score": 7,
            "summary": "Neglected property.",
        }
        desc_id = insert_description(
            conn, addr["id"], image_id, desc_dict, '{"raw": true}', "test-model"
        )
        assert desc_id > 0

        row = conn.execute("SELECT * FROM descriptions WHERE id=?", (desc_id,)).fetchone()
        assert row["roof_condition"] == "poor"
        assert row["distress_score"] == 7
        assert json.loads(row["damage_list"]) == ["cracked foundation", "missing gutters"]
        assert row["model_used"] == "test-model"
        conn.close()


class TestInsertScore:
    def test_insert_and_return_id(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)
        insert_addresses(conn, city_id, [
            {"osm_id": 800, "osm_type": "node", "lat": 42.33, "lng": -83.05},
        ])
        addr = get_addresses_by_status(conn, city_id, "harvested")[0]
        image_id = insert_image(conn, addr["id"], {
            "pano_id": "pano_s", "lat": 42.33, "lng": -83.05,
            "heading": 90.0, "image_path": "img.jpg",
        })
        desc_id = insert_description(
            conn, addr["id"], image_id,
            {"distress_score": 8, "summary": "Bad shape."},
            "{}", "test-model",
        )

        score_dict = {
            "qualified": True,
            "confidence": 0.85,
            "reasoning": "High distress.",
            "wholesale_potential": "high",
            "estimated_discount_range": "25-35%",
        }
        score_id = insert_score(conn, addr["id"], desc_id, score_dict, "test-model")
        assert score_id > 0

        row = conn.execute("SELECT * FROM scores WHERE id=?", (score_id,)).fetchone()
        assert row["qualified"] == 1
        assert row["confidence"] == 0.85
        assert row["wholesale_potential"] == "high"
        conn.close()

    def test_qualified_false_stored_as_zero(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)
        insert_addresses(conn, city_id, [
            {"osm_id": 801, "osm_type": "node", "lat": 42.33, "lng": -83.05},
        ])
        addr = get_addresses_by_status(conn, city_id, "harvested")[0]
        image_id = insert_image(conn, addr["id"], {
            "pano_id": "pano_s2", "lat": 42.33, "lng": -83.05,
            "heading": 90.0, "image_path": "img.jpg",
        })
        desc_id = insert_description(
            conn, addr["id"], image_id, {"distress_score": 2}, "{}", "m",
        )

        score_id = insert_score(conn, addr["id"], desc_id, {"qualified": False}, "m")
        row = conn.execute("SELECT qualified FROM scores WHERE id=?", (score_id,)).fetchone()
        assert row["qualified"] == 0
        conn.close()


class TestGetQualifiedLeads:
    def test_returns_qualified_only(self, populated_db):
        db_path, city_id = populated_db
        conn = get_connection(db_path)
        leads = get_qualified_leads(conn, city_id)
        assert len(leads) == 3
        for lead in leads:
            assert lead["full_address"]
            assert lead["distress_score"] == 7
            assert lead["wholesale_potential"] == "medium"
            assert lead["image_path"]
        conn.close()

    def test_excludes_unqualified(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)
        insert_addresses(conn, city_id, [
            {"osm_id": 900, "osm_type": "node", "lat": 42.33, "lng": -83.05,
             "full_address": "900 Main St"},
        ])
        addr = get_addresses_by_status(conn, city_id, "harvested")[0]
        image_id = insert_image(conn, addr["id"], {
            "pano_id": "p", "lat": 42.33, "lng": -83.05,
            "heading": 0.0, "image_path": "i.jpg",
        })
        desc_id = insert_description(
            conn, addr["id"], image_id, {"distress_score": 3}, "{}", "m",
        )
        insert_score(conn, addr["id"], desc_id, {"qualified": False}, "m")

        leads = get_qualified_leads(conn, city_id)
        assert leads == []
        conn.close()

    def test_ordered_by_distress_then_confidence(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Detroit", "Michigan", bbox)

        for i, (distress, conf) in enumerate([(5, 0.9), (8, 0.7), (8, 0.9)]):
            insert_addresses(conn, city_id, [
                {"osm_id": 950 + i, "osm_type": "node", "lat": 42.33 + i * 0.001,
                 "lng": -83.05, "full_address": f"Addr {i}"},
            ])
            addr = conn.execute(
                "SELECT id FROM addresses WHERE osm_id=?", (950 + i,)
            ).fetchone()
            image_id = insert_image(conn, addr["id"], {
                "pano_id": f"p{i}", "lat": 42.33, "lng": -83.05,
                "heading": 0.0, "image_path": f"i{i}.jpg",
            })
            desc_id = insert_description(
                conn, addr["id"], image_id,
                {"distress_score": distress}, "{}", "m",
            )
            insert_score(conn, addr["id"], desc_id, {
                "qualified": True, "confidence": conf,
            }, "m")

        leads = get_qualified_leads(conn, city_id)
        assert len(leads) == 3
        # distress 8 first, then by confidence desc
        assert leads[0]["distress_score"] == 8
        assert leads[0]["confidence"] == 0.9
        assert leads[1]["distress_score"] == 8
        assert leads[1]["confidence"] == 0.7
        assert leads[2]["distress_score"] == 5
        conn.close()


class TestGetPipelineStats:
    def test_counts_by_status(self, populated_db):
        db_path, city_id = populated_db
        conn = get_connection(db_path)
        stats = get_pipeline_stats(conn, city_id)
        # 3 scored + 2 remaining harvested
        assert stats.get("scored", 0) == 3
        assert stats.get("harvested", 0) == 2
        assert stats["qualified_leads"] == 3
        conn.close()

    def test_empty_city(self, tmp_db):
        conn = get_connection(tmp_db)
        bbox = {"south": 42.0, "north": 42.5, "west": -83.5, "east": -83.0}
        city_id = insert_city(conn, "Empty", None, bbox)
        stats = get_pipeline_stats(conn, city_id)
        assert stats == {"qualified_leads": 0}
        conn.close()
