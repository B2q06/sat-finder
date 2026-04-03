"""Tests for the export stage."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from satelite.db import get_connection, init_db
from satelite.models import SateliteConfig
from satelite.stages.export import _find_city, run_export, show_status


# ---------------------------------------------------------------------------
# _find_city
# ---------------------------------------------------------------------------

class TestFindCity:
    def test_exact_match(self, populated_db):
        db_path, _ = populated_db
        conn = get_connection(db_path)
        result = _find_city(conn, "Detroit")
        conn.close()
        assert result is not None
        city_id, name = result
        assert name == "Detroit"

    def test_fuzzy_match(self, populated_db):
        db_path, _ = populated_db
        conn = get_connection(db_path)
        result = _find_city(conn, "etro")
        conn.close()
        assert result is not None
        _, name = result
        assert name == "Detroit"

    def test_no_match(self, populated_db):
        db_path, _ = populated_db
        conn = get_connection(db_path)
        result = _find_city(conn, "Nonexistent")
        conn.close()
        assert result is None

    def test_empty_db(self, tmp_db):
        conn = get_connection(tmp_db)
        result = _find_city(conn, "Detroit")
        conn.close()
        assert result is None


# ---------------------------------------------------------------------------
# run_export
# ---------------------------------------------------------------------------

class TestRunExport:
    def test_creates_csv_with_correct_columns(self, populated_db, tmp_path):
        db_path, _ = populated_db
        config = SateliteConfig(pipeline={"db_path": db_path, "image_dir": str(tmp_path / "img")})
        out = str(tmp_path / "leads.csv")

        run_export(config, "Detroit", output_path=out)

        assert Path(out).exists()
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        expected_cols = {
            "full_address", "lat", "lng", "distress_score",
            "wholesale_potential", "confidence",
        }
        assert expected_cols.issubset(set(reader.fieldnames))

    def test_default_output_path(self, populated_db, tmp_path, monkeypatch):
        db_path, _ = populated_db
        config = SateliteConfig(pipeline={"db_path": db_path, "image_dir": str(tmp_path / "img")})
        # Change working directory so default export goes to tmp_path
        monkeypatch.chdir(tmp_path)

        run_export(config, "Detroit")

        exports = list(Path(tmp_path / "data" / "exports").glob("detroit_*.csv"))
        assert len(exports) == 1

    def test_city_not_found(self, tmp_db, tmp_path):
        config = SateliteConfig(pipeline={"db_path": tmp_db, "image_dir": str(tmp_path / "img")})
        with pytest.raises(SystemExit):
            run_export(config, "Nonexistent")

    def test_no_qualified_leads(self, tmp_db, tmp_path):
        """City exists but has no qualified leads -- should print warning, not crash."""
        conn = get_connection(tmp_db)
        from satelite.db import insert_city
        insert_city(conn, "EmptyTown", "State", {
            "south": 40.0, "north": 40.1, "west": -80.0, "east": -79.9,
        })
        conn.close()

        config = SateliteConfig(pipeline={"db_path": tmp_db, "image_dir": str(tmp_path / "img")})
        # Should not raise
        run_export(config, "EmptyTown")

        # No CSV should be created
        exports = list(tmp_path.glob("**/*.csv"))
        assert len(exports) == 0


# ---------------------------------------------------------------------------
# show_status
# ---------------------------------------------------------------------------

class TestShowStatus:
    def test_single_city(self, populated_db, tmp_path):
        db_path, _ = populated_db
        config = SateliteConfig(pipeline={"db_path": db_path, "image_dir": str(tmp_path / "img")})
        # Should not raise
        show_status(config, "Detroit")

    def test_all_cities(self, populated_db, tmp_path):
        db_path, _ = populated_db
        config = SateliteConfig(pipeline={"db_path": db_path, "image_dir": str(tmp_path / "img")})
        show_status(config, None)

    def test_city_not_found(self, tmp_db, tmp_path):
        config = SateliteConfig(pipeline={"db_path": tmp_db, "image_dir": str(tmp_path / "img")})
        with pytest.raises(SystemExit):
            show_status(config, "Ghost")

    def test_empty_database(self, tmp_db, tmp_path):
        config = SateliteConfig(pipeline={"db_path": tmp_db, "image_dir": str(tmp_path / "img")})
        # No cities at all -- should print message, not crash
        show_status(config, None)
