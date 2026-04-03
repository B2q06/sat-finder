"""SQLite database schema and operations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS cities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    state TEXT,
    bbox_south REAL NOT NULL,
    bbox_north REAL NOT NULL,
    bbox_west REAL NOT NULL,
    bbox_east REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(name, state)
);

CREATE TABLE IF NOT EXISTS addresses (
    id INTEGER PRIMARY KEY,
    city_id INTEGER NOT NULL REFERENCES cities(id),
    osm_id INTEGER,
    osm_type TEXT,
    housenumber TEXT,
    street TEXT,
    city TEXT,
    state TEXT,
    postcode TEXT,
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    full_address TEXT,
    status TEXT DEFAULT 'harvested',
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(osm_id, osm_type)
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    address_id INTEGER NOT NULL REFERENCES addresses(id),
    pano_id TEXT,
    pano_lat REAL,
    pano_lng REAL,
    heading REAL,
    capture_date TEXT,
    image_path TEXT NOT NULL,
    zoom_level INTEGER DEFAULT 3,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS descriptions (
    id INTEGER PRIMARY KEY,
    address_id INTEGER NOT NULL REFERENCES addresses(id),
    image_id INTEGER NOT NULL REFERENCES images(id),
    roof_condition TEXT,
    paint_condition TEXT,
    yard_condition TEXT,
    driveway_condition TEXT,
    windows_condition TEXT,
    vacancy_signs TEXT,
    maintenance_level TEXT,
    damage_list TEXT,
    distress_score INTEGER,
    summary TEXT,
    raw_response TEXT,
    model_used TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY,
    address_id INTEGER NOT NULL REFERENCES addresses(id),
    description_id INTEGER NOT NULL REFERENCES descriptions(id),
    qualified INTEGER NOT NULL,
    confidence REAL,
    reasoning TEXT,
    wholesale_potential TEXT,
    estimated_discount_range TEXT,
    model_used TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_addresses_status ON addresses(status);
CREATE INDEX IF NOT EXISTS idx_addresses_city_id ON addresses(city_id);
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str) -> None:
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    conn.close()


def insert_city(conn: sqlite3.Connection, name: str, state: str | None, bbox: dict) -> int:
    cursor = conn.execute(
        """INSERT INTO cities (name, state, bbox_south, bbox_north, bbox_west, bbox_east)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(name, state) DO UPDATE SET
             bbox_south=excluded.bbox_south, bbox_north=excluded.bbox_north,
             bbox_west=excluded.bbox_west, bbox_east=excluded.bbox_east
           RETURNING id""",
        (name, state, bbox["south"], bbox["north"], bbox["west"], bbox["east"]),
    )
    row = cursor.fetchone()
    conn.commit()
    return row["id"]


def get_city_id(conn: sqlite3.Connection, name: str, state: str | None = None) -> int | None:
    if state:
        row = conn.execute(
            "SELECT id FROM cities WHERE name=? AND state=?", (name, state)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id FROM cities WHERE name=?", (name,)
        ).fetchone()
    return row["id"] if row else None


def insert_addresses(conn: sqlite3.Connection, city_id: int, addresses: list[dict]) -> int:
    inserted = 0
    for addr in addresses:
        try:
            conn.execute(
                """INSERT INTO addresses
                   (city_id, osm_id, osm_type, housenumber, street, city, state, postcode, lat, lng, full_address, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'harvested')""",
                (
                    city_id,
                    addr.get("osm_id"),
                    addr.get("osm_type"),
                    addr.get("housenumber"),
                    addr.get("street"),
                    addr.get("city"),
                    addr.get("state"),
                    addr.get("postcode"),
                    addr["lat"],
                    addr["lng"],
                    addr.get("full_address", ""),
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    return inserted


def get_addresses_by_status(
    conn: sqlite3.Connection, city_id: int, status: str, limit: int | None = None
) -> list[dict[str, Any]]:
    query = "SELECT * FROM addresses WHERE city_id=? AND status=?"
    params: list[Any] = [city_id, status]
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def update_address_status(conn: sqlite3.Connection, address_id: int, status: str) -> None:
    conn.execute("UPDATE addresses SET status=? WHERE id=?", (status, address_id))
    conn.commit()


def insert_image(conn: sqlite3.Connection, address_id: int, pano: dict) -> int:
    cursor = conn.execute(
        """INSERT INTO images (address_id, pano_id, pano_lat, pano_lng, heading, capture_date, image_path, zoom_level)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (
            address_id,
            pano["pano_id"],
            pano["lat"],
            pano["lng"],
            pano["heading"],
            pano.get("capture_date"),
            pano["image_path"],
            pano.get("zoom_level", 3),
        ),
    )
    row = cursor.fetchone()
    conn.commit()
    return row["id"]


def insert_description(conn: sqlite3.Connection, address_id: int, image_id: int, desc: dict, raw: str, model: str) -> int:
    cursor = conn.execute(
        """INSERT INTO descriptions
           (address_id, image_id, roof_condition, paint_condition, yard_condition,
            driveway_condition, windows_condition, vacancy_signs, maintenance_level,
            damage_list, distress_score, summary, raw_response, model_used)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (
            address_id,
            image_id,
            desc.get("roof_condition"),
            desc.get("paint_condition"),
            desc.get("yard_condition"),
            desc.get("driveway_condition"),
            desc.get("windows_condition"),
            desc.get("vacancy_signs"),
            desc.get("maintenance_level"),
            json.dumps(desc.get("damage_list", [])),
            desc.get("distress_score"),
            desc.get("summary", ""),
            raw,
            model,
        ),
    )
    row = cursor.fetchone()
    conn.commit()
    return row["id"]


def insert_score(conn: sqlite3.Connection, address_id: int, description_id: int, score: dict, model: str) -> int:
    cursor = conn.execute(
        """INSERT INTO scores
           (address_id, description_id, qualified, confidence, reasoning,
            wholesale_potential, estimated_discount_range, model_used)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (
            address_id,
            description_id,
            1 if score.get("qualified") else 0,
            score.get("confidence", 0.5),
            score.get("reasoning", ""),
            score.get("wholesale_potential", "low"),
            score.get("estimated_discount_range", ""),
            model,
        ),
    )
    row = cursor.fetchone()
    conn.commit()
    return row["id"]


def get_qualified_leads(conn: sqlite3.Connection, city_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT a.full_address, a.lat, a.lng,
                  d.distress_score, d.maintenance_level, d.roof_condition,
                  d.paint_condition, d.yard_condition, d.windows_condition,
                  d.vacancy_signs, d.damage_list, d.summary,
                  s.wholesale_potential, s.confidence, s.reasoning,
                  s.estimated_discount_range,
                  i.image_path, i.capture_date
           FROM scores s
           JOIN descriptions d ON s.description_id = d.id
           JOIN addresses a ON s.address_id = a.id
           JOIN images i ON d.image_id = i.id
           WHERE a.city_id=? AND s.qualified=1
           ORDER BY d.distress_score DESC, s.confidence DESC""",
        (city_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_pipeline_stats(conn: sqlite3.Connection, city_id: int) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM addresses WHERE city_id=? GROUP BY status",
        (city_id,),
    ).fetchall()
    stats = {r["status"]: r["cnt"] for r in rows}
    total_qualified = conn.execute(
        """SELECT COUNT(*) as cnt FROM scores s
           JOIN addresses a ON s.address_id = a.id
           WHERE a.city_id=? AND s.qualified=1""",
        (city_id,),
    ).fetchone()
    stats["qualified_leads"] = total_qualified["cnt"] if total_qualified else 0
    return stats
