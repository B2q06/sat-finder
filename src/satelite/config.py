"""Config loading from YAML file with environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from satelite.models import SateliteConfig


def load_config(config_path: str = "config.yaml", db_override: str | None = None) -> SateliteConfig:
    data = {}
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    config = SateliteConfig(**data)

    if db_override:
        config.pipeline.db_path = db_override

    # Ensure data directories exist
    Path(config.pipeline.db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.pipeline.image_dir).mkdir(parents=True, exist_ok=True)

    return config


def get_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is required for AI stages. "
            "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return key
