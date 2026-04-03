"""CLI entry point for the Satelite pipeline."""

from __future__ import annotations

import click
from rich.console import Console

from satelite.config import load_config
from satelite.db import init_db

console = Console()


@click.group()
@click.option("--config", "config_path", type=click.Path(), default="config.yaml", help="Config file path")
@click.option("--db", "db_path", type=click.Path(), default=None, help="Override database path")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, config_path: str, db_path: str | None, verbose: bool) -> None:
    """Satelite: Distressed property lead finder for real estate wholesaling."""
    ctx.ensure_object(dict)
    cfg = load_config(config_path, db_override=db_path)
    ctx.obj["config"] = cfg
    ctx.obj["verbose"] = verbose
    ctx.obj["console"] = console
    init_db(cfg.pipeline.db_path)


@cli.command()
@click.argument("city")
@click.option("--limit", default=None, type=int, help="Max addresses to harvest")
@click.pass_context
def harvest(ctx: click.Context, city: str, limit: int | None) -> None:
    """Stage 1: Harvest residential addresses from OpenStreetMap."""
    from satelite.stages.harvest import run_harvest

    cfg = ctx.obj["config"]
    run_harvest(cfg, city, limit=limit)


@cli.command()
@click.option("--city", required=True, help="City to capture images for")
@click.option("--batch-size", default=None, type=int, help="Override batch size")
@click.option("--concurrency", default=None, type=int, help="Override download concurrency")
@click.pass_context
def capture(ctx: click.Context, city: str, batch_size: int | None, concurrency: int | None) -> None:
    """Stage 2: Capture street view images for harvested addresses."""
    from satelite.stages.capture import run_capture

    cfg = ctx.obj["config"]
    if batch_size:
        cfg.pipeline.batch_size = batch_size
    if concurrency:
        cfg.streetview.concurrency = concurrency
    run_capture(cfg, city)


@cli.command()
@click.option("--city", required=True, help="City to describe properties for")
@click.option("--model", default=None, help="Override Claude model")
@click.option("--limit", default=None, type=int, help="Max properties to describe")
@click.pass_context
def describe(ctx: click.Context, city: str, model: str | None, limit: int | None) -> None:
    """Stage 3: Generate AI descriptions of property conditions."""
    from satelite.stages.describe import run_describe

    cfg = ctx.obj["config"]
    if model:
        cfg.anthropic.model = model
    run_describe(cfg, city, limit=limit)


@cli.command()
@click.option("--city", required=True, help="City to score properties for")
@click.option("--limit", default=None, type=int, help="Max properties to score")
@click.pass_context
def score(ctx: click.Context, city: str, limit: int | None) -> None:
    """Stage 4: Score properties for wholesale potential."""
    from satelite.stages.score import run_score

    cfg = ctx.obj["config"]
    run_score(cfg, city, limit=limit)


@cli.command()
@click.option("--city", required=True, help="City to export leads for")
@click.option("-o", "--output", type=click.Path(), default=None, help="Output CSV path")
@click.pass_context
def export(ctx: click.Context, city: str, output: str | None) -> None:
    """Stage 5: Export qualified leads to CSV."""
    from satelite.stages.export import run_export

    cfg = ctx.obj["config"]
    run_export(cfg, city, output_path=output)


@cli.command()
@click.argument("city")
@click.option("--limit", default=None, type=int, help="Max addresses per stage")
@click.pass_context
def run(ctx: click.Context, city: str, limit: int | None) -> None:
    """Run the full pipeline for a city."""
    ctx.invoke(harvest, city=city, limit=limit)
    ctx.invoke(capture, city=city)
    ctx.invoke(describe, city=city, limit=limit)
    ctx.invoke(score, city=city, limit=limit)
    ctx.invoke(export, city=city)


@cli.command()
@click.option("--city", default=None, help="City to show stats for (all cities if omitted)")
@click.pass_context
def status(ctx: click.Context, city: str | None) -> None:
    """Show pipeline status and statistics."""
    from satelite.stages.export import show_status

    cfg = ctx.obj["config"]
    show_status(cfg, city)
