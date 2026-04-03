"""Stage 2 — Harvest residential addresses from OpenStreetMap via Overpass API."""

from __future__ import annotations

import time

import requests
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from satelite.db import get_connection, init_db, insert_addresses, insert_city
from satelite.geo import lookup_city_bbox
from satelite.models import BBox, SateliteConfig

_OVERPASS_QUERY = """\
[out:json][timeout:{timeout}];
(
  way["building"="residential"]["addr:housenumber"]["addr:street"]({south},{west},{north},{east});
  way["building"="house"]["addr:housenumber"]["addr:street"]({south},{west},{north},{east});
  way["building"="detached"]["addr:housenumber"]["addr:street"]({south},{west},{north},{east});
  way["building"="semidetached_house"]["addr:housenumber"]["addr:street"]({south},{west},{north},{east});
  way["building"="terrace"]["addr:housenumber"]["addr:street"]({south},{west},{north},{east});
  way["building"="apartments"]["addr:housenumber"]["addr:street"]({south},{west},{north},{east});
  node["addr:housenumber"]["addr:street"]["building"="residential"]({south},{west},{north},{east});
  node["addr:housenumber"]["addr:street"]["building"="house"]({south},{west},{north},{east});
);
out center body;
"""

_MAX_RETRIES = 5
_BACKOFF_BASE = 30


def _parse_elements(elements: list[dict], city_fallback: str, state_fallback: str | None) -> list[dict]:
    """Extract address dicts from Overpass JSON elements."""
    addresses: list[dict] = []
    for el in elements:
        tags = el.get("tags", {})
        housenumber = tags.get("addr:housenumber")
        street = tags.get("addr:street")
        if not housenumber or not street:
            continue

        if el["type"] == "way":
            center = el.get("center", {})
            lat = center.get("lat")
            lng = center.get("lon")
        else:
            lat = el.get("lat")
            lng = el.get("lon")

        if lat is None or lng is None:
            continue

        city = tags.get("addr:city", city_fallback)
        state = tags.get("addr:state", state_fallback)

        addresses.append({
            "osm_id": el.get("id"),
            "osm_type": el["type"],
            "housenumber": housenumber,
            "street": street,
            "city": city,
            "state": state,
            "postcode": tags.get("addr:postcode"),
            "lat": lat,
            "lng": lng,
            "full_address": f"{housenumber} {street}, {city}, {state}",
        })
    return addresses


def _query_cell(
    session: requests.Session,
    endpoint: str,
    cell: BBox,
    timeout: int,
) -> list[dict]:
    """POST a single cell query to Overpass, returning raw elements.

    Handles 429 with exponential backoff and 504 by returning an empty list.
    """
    query = _OVERPASS_QUERY.format(
        timeout=timeout,
        south=cell.south,
        west=cell.west,
        north=cell.north,
        east=cell.east,
    )

    for attempt in range(_MAX_RETRIES):
        resp = session.post(endpoint, data={"data": query}, timeout=timeout + 30)

        if resp.status_code == 200:
            return resp.json().get("elements", [])

        if resp.status_code == 429:
            wait = _BACKOFF_BASE * (2 ** attempt)
            from rich.console import Console
            Console(stderr=True).print(
                f"[yellow]Rate limited (429). Backing off {wait}s (attempt {attempt + 1}/{_MAX_RETRIES})[/yellow]"
            )
            time.sleep(wait)
            continue

        if resp.status_code == 504:
            from rich.console import Console
            Console(stderr=True).print(
                f"[yellow]Timeout (504) for cell ({cell.south:.4f},{cell.west:.4f})-"
                f"({cell.north:.4f},{cell.east:.4f}). Skipping.[/yellow]"
            )
            return []

        resp.raise_for_status()

    from rich.console import Console
    Console(stderr=True).print(
        f"[red]Max retries ({_MAX_RETRIES}) exceeded for cell "
        f"({cell.south:.4f},{cell.west:.4f})-({cell.north:.4f},{cell.east:.4f}). Skipping.[/red]"
    )
    return []


def run_harvest(config: SateliteConfig, city_query: str, limit: int | None = None) -> None:
    """Harvest residential addresses for a city from OpenStreetMap."""
    from rich.console import Console

    console = Console(stderr=True)

    console.print(f"[bold]Looking up city:[/bold] {city_query}")
    display_name, state, bbox = lookup_city_bbox(city_query)
    console.print(f"[green]Found:[/green] {display_name}")
    console.print(f"  BBox: S={bbox.south:.4f} N={bbox.north:.4f} W={bbox.west:.4f} E={bbox.east:.4f}")

    cells = bbox.split_into_cells(config.overpass.cell_size)
    console.print(f"  Grid: {len(cells)} cells (cell_size={config.overpass.cell_size})")

    # Initialise DB and register city
    init_db(config.pipeline.db_path)
    conn = get_connection(config.pipeline.db_path)
    city_name = display_name.split(",")[0].strip()
    city_id = insert_city(conn, city_name, state, bbox.model_dump())

    session = requests.Session()
    session.headers.update({"User-Agent": "satelite-lead-finder/0.1"})

    total_inserted = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("| {task.fields[addresses]} addresses"),
        console=console,
    ) as progress:
        task = progress.add_task("Querying cells", total=len(cells), addresses=0)

        for i, cell in enumerate(cells):
            elements = _query_cell(
                session,
                config.overpass.endpoint,
                cell,
                config.overpass.timeout,
            )

            addresses = _parse_elements(elements, city_name, state)

            if limit is not None:
                remaining = limit - total_inserted
                if remaining <= 0:
                    break
                addresses = addresses[:remaining]

            if addresses:
                inserted = insert_addresses(conn, city_id, addresses)
                total_inserted += inserted

            progress.update(task, advance=1, addresses=total_inserted)

            if limit is not None and total_inserted >= limit:
                break

            # Delay between queries (skip after last cell)
            if i < len(cells) - 1:
                time.sleep(config.overpass.delay_between_queries)

    conn.close()
    console.print(f"\n[bold green]Harvest complete.[/bold green] {total_inserted} addresses inserted for {city_name}.")
