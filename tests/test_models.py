"""Tests for satelite.models — Pydantic models and BBox cell splitting."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from satelite.models import (
    BBox,
    LeadScore,
    PipelineConfig,
    PropertyDescription,
    SateliteConfig,
    StreetviewConfig,
)


# ---------------------------------------------------------------------------
# BBox.split_into_cells
# ---------------------------------------------------------------------------

class TestBBoxSplitIntoCells:
    def test_single_cell_when_bbox_smaller_than_cell_size(self):
        bbox = BBox(south=42.0, north=42.01, west=-83.0, east=-82.99)
        cells = bbox.split_into_cells(cell_size=0.02)
        assert len(cells) == 1
        assert cells[0].south == 42.0
        assert cells[0].north == 42.01
        assert cells[0].west == -83.0
        assert cells[0].east == -82.99

    def test_four_cells_for_2x2_grid(self):
        bbox = BBox(south=0.0, north=0.04, west=0.0, east=0.04)
        cells = bbox.split_into_cells(cell_size=0.02)
        assert len(cells) == 4

    def test_cells_cover_entire_bbox(self):
        bbox = BBox(south=10.0, north=10.05, west=20.0, east=20.06)
        cells = bbox.split_into_cells(cell_size=0.02)
        assert cells[0].south == bbox.south
        assert cells[0].west == bbox.west
        assert cells[-1].north == bbox.north
        assert cells[-1].east == bbox.east

    def test_cell_boundaries_do_not_exceed_bbox(self):
        bbox = BBox(south=0.0, north=0.05, west=0.0, east=0.05)
        cells = bbox.split_into_cells(cell_size=0.02)
        for cell in cells:
            assert cell.south >= bbox.south
            assert cell.north <= bbox.north
            assert cell.west >= bbox.west
            assert cell.east <= bbox.east

    def test_default_cell_size(self):
        bbox = BBox(south=0.0, north=0.04, west=0.0, east=0.04)
        cells = bbox.split_into_cells()  # default 0.02
        assert len(cells) == 4

    def test_non_uniform_bbox(self):
        bbox = BBox(south=0.0, north=0.02, west=0.0, east=0.06)
        cells = bbox.split_into_cells(cell_size=0.02)
        # 1 row x 3 cols = 3
        assert len(cells) == 3


# ---------------------------------------------------------------------------
# PropertyDescription
# ---------------------------------------------------------------------------

class TestPropertyDescription:
    def test_defaults(self):
        desc = PropertyDescription()
        assert desc.distress_score == 5
        assert desc.roof_condition == "not_visible"
        assert desc.damage_list == []
        assert desc.summary == ""

    def test_valid_distress_score_bounds(self):
        low = PropertyDescription(distress_score=1)
        assert low.distress_score == 1
        high = PropertyDescription(distress_score=10)
        assert high.distress_score == 10

    def test_distress_score_below_minimum_rejected(self):
        with pytest.raises(ValidationError):
            PropertyDescription(distress_score=0)

    def test_distress_score_above_maximum_rejected(self):
        with pytest.raises(ValidationError):
            PropertyDescription(distress_score=11)

    def test_custom_fields(self):
        desc = PropertyDescription(
            roof_condition="poor",
            paint_condition="peeling",
            damage_list=["broken window", "sagging roof"],
            distress_score=8,
            summary="Significant deterioration",
        )
        assert desc.roof_condition == "poor"
        assert len(desc.damage_list) == 2


# ---------------------------------------------------------------------------
# LeadScore
# ---------------------------------------------------------------------------

class TestLeadScore:
    def test_defaults(self):
        score = LeadScore()
        assert score.qualified is False
        assert score.confidence == 0.5
        assert score.wholesale_potential == "low"

    def test_valid_confidence_bounds(self):
        LeadScore(confidence=0.0)
        LeadScore(confidence=1.0)

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            LeadScore(confidence=-0.1)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            LeadScore(confidence=1.1)

    def test_qualified_lead(self):
        score = LeadScore(
            qualified=True,
            confidence=0.9,
            reasoning="Clear signs of distress",
            wholesale_potential="high",
            estimated_discount_range="20-30%",
        )
        assert score.qualified is True
        assert score.estimated_discount_range == "20-30%"


# ---------------------------------------------------------------------------
# SateliteConfig
# ---------------------------------------------------------------------------

class TestSateliteConfig:
    def test_all_defaults(self):
        cfg = SateliteConfig()
        assert cfg.pipeline.batch_size == 50
        assert cfg.pipeline.image_dir == "data/images"
        assert cfg.pipeline.db_path == "data/satelite.db"
        assert cfg.streetview.zoom == 3
        assert cfg.streetview.concurrency == 5
        assert cfg.streetview.search_radius == 50
        assert cfg.overpass.cell_size == 0.02
        assert cfg.anthropic.model == "claude-sonnet-4-20250514"
        assert cfg.scoring.min_distress_score == 5
        assert cfg.scoring.min_confidence == 0.6

    def test_override_nested(self):
        cfg = SateliteConfig(
            pipeline=PipelineConfig(batch_size=10, db_path="/tmp/test.db"),
            streetview=StreetviewConfig(zoom=4, concurrency=2),
        )
        assert cfg.pipeline.batch_size == 10
        assert cfg.streetview.zoom == 4
        # Other defaults still hold
        assert cfg.overpass.timeout == 180
