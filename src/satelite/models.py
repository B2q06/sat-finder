"""Pydantic models for data flowing through the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    south: float
    north: float
    west: float
    east: float

    def split_into_cells(self, cell_size: float = 0.02) -> list[BBox]:
        cells = []
        lat = self.south
        while lat < self.north:
            lng = self.west
            while lng < self.east:
                cells.append(BBox(
                    south=lat,
                    north=min(lat + cell_size, self.north),
                    west=lng,
                    east=min(lng + cell_size, self.east),
                ))
                lng += cell_size
            lat += cell_size
        return cells


class CityConfig(BaseModel):
    name: str
    state: str | None = None
    bbox: BBox


class Address(BaseModel):
    osm_id: int | None = None
    osm_type: str | None = None
    housenumber: str | None = None
    street: str | None = None
    city: str | None = None
    state: str | None = None
    postcode: str | None = None
    lat: float
    lng: float
    full_address: str = ""
    status: str = "harvested"


class PanoramaResult(BaseModel):
    pano_id: str
    lat: float
    lng: float
    heading: float
    capture_date: str | None = None
    image_path: str


class PropertyDescription(BaseModel):
    roof_condition: str = "not_visible"
    paint_condition: str = "not_visible"
    yard_condition: str = "not_visible"
    driveway_condition: str = "not_visible"
    windows_condition: str = "not_visible"
    vacancy_signs: str = "unclear"
    maintenance_level: str = "average"
    damage_list: list[str] = Field(default_factory=list)
    distress_score: int = Field(ge=1, le=10, default=5)
    summary: str = ""


class LeadScore(BaseModel):
    qualified: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = ""
    wholesale_potential: str = "low"
    estimated_discount_range: str = ""


class OverpassConfig(BaseModel):
    endpoint: str = "https://overpass-api.de/api/interpreter"
    timeout: int = 180
    cell_size: float = 0.02
    delay_between_queries: float = 5.0


class StreetviewConfig(BaseModel):
    zoom: int = 3
    search_radius: int = 50
    concurrency: int = 5
    delay_between_batches: float = 1.0


class AnthropicConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024


class PipelineConfig(BaseModel):
    batch_size: int = 50
    image_dir: str = "data/images"
    db_path: str = "data/satelite.db"


class ScoringConfig(BaseModel):
    min_distress_score: int = 5
    min_confidence: float = 0.6


class SateliteConfig(BaseModel):
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    overpass: OverpassConfig = Field(default_factory=OverpassConfig)
    streetview: StreetviewConfig = Field(default_factory=StreetviewConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
