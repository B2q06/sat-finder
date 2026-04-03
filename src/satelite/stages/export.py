"""Stage 5 — Export qualified leads to CSV and show pipeline status."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from satelite.db import get_connection, get_pipeline_stats, get_qualified_leads
from satelite.geo import slugify_city
from satelite.models import SateliteConfig


def _find_city(conn, city_query: str) -> tuple[int, str] | None:
    """Find a city by fuzzy name match. Returns (city_id, city_name) or None."""
    row = conn.execute(
        "SELECT id, name FROM cities WHERE name LIKE ? LIMIT 1",
        (f"%{city_query}%",),
    ).fetchone()
    if row:
        return row["id"], row["name"]
    return None


def run_export(config: SateliteConfig, city_query: str, output_path: str | None = None) -> None:
    """Export qualified leads for a city to CSV."""
    console = Console(stderr=True)

    conn = get_connection(config.pipeline.db_path)
    result = _find_city(conn, city_query)
    if not result:
        conn.close()
        console.print(f"[red]City not found in database:[/red] {city_query}")
        raise SystemExit(1)

    city_id, city_name = result
    leads = get_qualified_leads(conn, city_id)
    conn.close()

    if not leads:
        console.print(f"[yellow]No qualified leads found for {city_name}.[/yellow]")
        return

    df = pd.DataFrame(leads)

    if output_path is None:
        export_dir = Path("data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        slug = slugify_city(city_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(export_dir / f"{slug}_{timestamp}.csv")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"[green]Exported {len(df)} leads to:[/green] {output_path}")

    # Summary table — top 10 by distress score
    table = Table(title=f"Top Leads — {city_name} ({len(leads)} total)")
    table.add_column("Address", style="cyan", max_width=45)
    table.add_column("Distress", justify="right")
    table.add_column("Wholesale", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Discount Range", justify="center")

    for lead in leads[:10]:
        table.add_row(
            str(lead["full_address"]),
            str(lead["distress_score"]),
            str(lead["wholesale_potential"]),
            f"{lead['confidence']:.0%}" if lead["confidence"] is not None else "—",
            str(lead.get("estimated_discount_range") or "—"),
        )

    console.print(table)


def show_status(config: SateliteConfig, city_query: str | None = None) -> None:
    """Display pipeline status for one or all cities."""
    console = Console(stderr=True)
    conn = get_connection(config.pipeline.db_path)

    if city_query:
        result = _find_city(conn, city_query)
        if not result:
            conn.close()
            console.print(f"[red]City not found in database:[/red] {city_query}")
            raise SystemExit(1)
        cities = [result]
    else:
        rows = conn.execute("SELECT id, name FROM cities ORDER BY name").fetchall()
        cities = [(r["id"], r["name"]) for r in rows]

    if not cities:
        conn.close()
        console.print("[yellow]No cities in database. Run harvest first.[/yellow]")
        return

    for city_id, city_name in cities:
        stats = get_pipeline_stats(conn, city_id)

        total = sum(v for k, v in stats.items() if k != "qualified_leads")
        harvested = stats.get("harvested", 0)
        captured = stats.get("captured", 0)
        described = stats.get("described", 0)
        scored = stats.get("scored", 0)
        skipped = stats.get("skipped", 0)
        qualified = stats.get("qualified_leads", 0)

        table = Table(title=f"Pipeline Status — {city_name}")
        table.add_column("Stage", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("", justify="left")

        table.add_row("Total addresses", str(total), "")
        table.add_row("Harvested", str(harvested), _bar(harvested, total))
        table.add_row("Captured", str(captured), _bar(captured, total))
        table.add_row("Described", str(described), _bar(described, total))
        table.add_row("Scored", str(scored), _bar(scored, total))
        table.add_row("Skipped", str(skipped), f"[dim]{_pct(skipped, total)} skip rate[/dim]")
        table.add_row(
            "[green]Qualified leads[/green]",
            f"[green]{qualified}[/green]",
            f"[green]{_pct(qualified, scored)} of scored[/green]" if scored else "",
        )

        console.print(table)
        console.print()

    conn.close()


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0%"
    return f"{part / whole:.0%}"


def _bar(part: int, whole: int, width: int = 20) -> str:
    if whole == 0:
        return ""
    filled = round(part / whole * width)
    return f"[green]{'█' * filled}[/green][dim]{'░' * (width - filled)}[/dim] {_pct(part, whole)}"
