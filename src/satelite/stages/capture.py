"""Stage 3 — Capture Google Street View panoramas and crop to face each house."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from streetlevel import streetview

from satelite.db import get_connection, get_addresses_by_status, insert_image, update_address_status
from satelite.geo import calculate_bearing, slugify_city
from satelite.models import SateliteConfig

logger = logging.getLogger(__name__)


def _crop_panorama_facing(image: Image.Image, target_bearing_deg: float, fov_degrees: float = 90) -> Image.Image:
    """Crop an equirectangular panorama to a FOV facing the target bearing, then trim sky/road."""
    width, height = image.size

    center_x = int((target_bearing_deg / 360.0) * width) % width
    fov_pixels = int((fov_degrees / 360.0) * width)
    half_fov = fov_pixels // 2

    left = center_x - half_fov
    right = center_x + half_fov

    if left < 0:
        part1 = image.crop((width + left, 0, width, height))
        part2 = image.crop((0, 0, right, height))
        result = Image.new(image.mode, (fov_pixels, height))
        result.paste(part1, (0, 0))
        result.paste(part2, (part1.width, 0))
    elif right > width:
        part1 = image.crop((left, 0, width, height))
        part2 = image.crop((0, 0, right - width, height))
        result = Image.new(image.mode, (fov_pixels, height))
        result.paste(part1, (0, 0))
        result.paste(part2, (part1.width, 0))
    else:
        result = image.crop((left, 0, right, height))

    # Trim top 25% (sky) and bottom 25% (road)
    rw, rh = result.size
    top = int(rh * 0.25)
    bottom = int(rh * 0.75)
    result = result.crop((0, top, rw, bottom))
    return result


async def _process_address(
    addr: dict,
    config: SateliteConfig,
    image_dir: Path,
    sem: asyncio.Semaphore,
    session,
) -> str:
    """Process a single address: find pano, download, crop, save. Returns status string."""
    async with sem:
        addr_id = addr["id"]
        lat, lng = addr["lat"], addr["lng"]
        full_address = addr.get("full_address", "")

        try:
            pano = streetview.find_panorama(lat, lng, radius=config.streetview.search_radius)
        except Exception:
            logger.debug("streetview.find_panorama failed for %s", full_address, exc_info=True)
            return "skipped"

        if pano is None:
            return "skipped"

        try:
            pano_image = await streetview.get_panorama_async(pano, zoom=config.streetview.zoom, session=session)
        except Exception:
            logger.debug("streetview.get_panorama_async failed for pano %s", pano.id, exc_info=True)
            return "skipped"

        if pano_image is None:
            return "skipped"

        # Calculate bearing from panorama location to the house
        bearing = calculate_bearing(pano.lat, pano.lon, lat, lng)

        # Crop panorama to face the house
        cropped = _crop_panorama_facing(pano_image, bearing)

        # Save image
        addr_hash = hashlib.sha256(full_address.encode()).hexdigest()[:12]
        out_path = image_dir / f"{addr_hash}.jpg"
        cropped.save(str(out_path), "JPEG", quality=85)

        # Build pano record for DB
        capture_date = None
        if hasattr(pano, "date") and pano.date is not None:
            capture_date = str(pano.date)

        pano_record = {
            "pano_id": str(pano.id),
            "lat": pano.lat,
            "lng": pano.lon,
            "heading": bearing,
            "capture_date": capture_date,
            "image_path": str(out_path),
            "zoom_level": config.streetview.zoom,
        }

        return ("captured", addr_id, pano_record)


async def _capture_batch(
    addresses: list[dict],
    config: SateliteConfig,
    image_dir: Path,
    conn,
    progress,
    task_id,
    counters: dict,
) -> None:
    """Download and process a batch of addresses concurrently."""
    import aiohttp

    sem = asyncio.Semaphore(config.streetview.concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [
            _process_address(addr, config, image_dir, sem, session)
            for addr in addresses
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for addr, result in zip(addresses, results):
        addr_id = addr["id"]

        if isinstance(result, Exception):
            logger.debug("Exception processing address %d: %s", addr_id, result)
            update_address_status(conn, addr_id, "skipped")
            counters["skipped"] += 1
        elif isinstance(result, tuple) and result[0] == "captured":
            _, _, pano_record = result
            insert_image(conn, addr_id, pano_record)
            update_address_status(conn, addr_id, "captured")
            counters["captured"] += 1
        else:
            update_address_status(conn, addr_id, "skipped")
            counters["skipped"] += 1

        progress.update(
            task_id,
            advance=1,
            captured=counters["captured"],
            skipped=counters["skipped"],
        )


def run_capture(config: SateliteConfig, city_query: str) -> None:
    """Capture street view images for all harvested addresses in a city."""
    console = Console(stderr=True)

    conn = get_connection(config.pipeline.db_path)

    # Fuzzy city lookup from DB — no external API call needed
    row = conn.execute(
        "SELECT id, name FROM cities WHERE name LIKE ? LIMIT 1",
        (f"%{city_query.split(',')[0].strip()}%",),
    ).fetchone()
    if not row:
        console.print(f"[red]City not found in database. Run harvest first.[/red]")
        conn.close()
        return
    city_id, city_name = row["id"], row["name"]
    console.print(f"[green]Found city:[/green] {city_name}")

    addresses = get_addresses_by_status(conn, city_id, "harvested")
    if not addresses:
        console.print("[yellow]No harvested addresses to capture. Run harvest first or check status.[/yellow]")
        conn.close()
        return

    console.print(f"  Addresses to capture: {len(addresses)}")

    # Prepare image output directory
    city_slug = slugify_city(city_name)
    image_dir = Path(config.pipeline.image_dir) / city_slug
    image_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config.pipeline.batch_size
    batches = [addresses[i : i + batch_size] for i in range(0, len(addresses), batch_size)]

    counters = {"captured": 0, "skipped": 0}

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("| captured={task.fields[captured]} skipped={task.fields[skipped]}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Capturing", total=len(addresses), captured=0, skipped=0)

        for i, batch in enumerate(batches):
            asyncio.run(
                _capture_batch(batch, config, image_dir, conn, progress, task_id, counters)
            )

            # Delay between batches (skip after last)
            if i < len(batches) - 1:
                import time
                time.sleep(config.streetview.delay_between_batches)

    conn.close()
    console.print(
        f"\n[bold green]Capture complete.[/bold green] "
        f"{counters['captured']} captured, {counters['skipped']} skipped."
    )
