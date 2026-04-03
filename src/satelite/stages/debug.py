"""Debug pipeline — Rich TUI showing real-time data flow through all stages."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import time
from pathlib import Path

from PIL import Image
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from satelite.config import get_anthropic_key, load_config
from satelite.db import (
    get_addresses_by_status,
    get_connection,
    init_db,
    insert_addresses,
    insert_city,
    insert_description,
    insert_image,
    insert_score,
    update_address_status,
)
from satelite.geo import calculate_bearing, lookup_city_bbox, slugify_city
from satelite.models import LeadScore, PropertyDescription, SateliteConfig
from satelite.stages.capture import _crop_panorama_facing
from satelite.stages.harvest import _parse_elements, _query_cell

logger = logging.getLogger(__name__)

# ── Agent state ──────────────────────────────────────────────────────────────

AGENT_INACTIVE = ""
AGENT_ACTIVE_STYLE = "blink bold yellow"


def _agent_indicator(active: bool, label: str) -> Text:
    if not active:
        return Text(f"  {label}", style="dim")
    return Text(f"  ● {label}", style=AGENT_ACTIVE_STYLE)


# ── Panel builders ───────────────────────────────────────────────────────────


class PipelineState:
    """Mutable state container for the debug TUI."""

    def __init__(self) -> None:
        # Harvest
        self.harvest_active = False
        self.harvest_addresses: list[str] = []
        self.harvest_total = 0
        self.harvest_current = ""

        # Capture
        self.capture_active = False
        self.capture_current = ""
        self.capture_current_coords = ""
        self.capture_status = ""
        self.capture_image_path = ""
        self.capture_total = 0
        self.capture_done = 0
        self.capture_skipped = 0
        self.capture_log: list[tuple[str, str, str]] = []  # (address, image_path, status)

        # Describe
        self.describe_active = False
        self.describe_current = ""
        self.describe_image_path = ""
        self.describe_obj: dict | None = None
        self.describe_total = 0
        self.describe_done = 0
        self.describe_skipped = 0

        # Score
        self.score_active = False
        self.score_current = ""
        self.score_obj: dict | None = None
        self.score_total = 0
        self.score_done = 0
        self.score_qualified = 0

        # Results
        self.results: list[dict] = []

    def build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=3),
            Layout(name="harvest_capture", size=14),
            Layout(name="describe", size=18),
            Layout(name="score", size=16),
            Layout(name="results", size=8),
        )
        layout["harvest_capture"].split_row(
            Layout(name="harvest"),
            Layout(name="capture"),
        )
        layout["top"].update(self._build_header())
        layout["harvest"].update(self._build_harvest_panel())
        layout["capture"].update(self._build_capture_panel())
        layout["describe"].update(self._build_describe_panel())
        layout["score"].update(self._build_score_panel())
        layout["results"].update(self._build_results_panel())
        return layout

    def _build_header(self) -> Panel:
        stages = []
        for name, active in [
            ("HARVEST", self.harvest_active),
            ("CAPTURE", self.capture_active),
            ("DESCRIBE", self.describe_active),
            ("SCORE", self.score_active),
        ]:
            if active:
                stages.append(f"[blink bold green]● {name}[/]")
            else:
                stages.append(f"[dim]○ {name}[/]")
        flow = "  →  ".join(stages)
        return Panel(Text.from_markup(flow, justify="center"), title="[bold]SATELITE DEBUG PIPELINE[/]", border_style="blue")

    def _build_harvest_panel(self) -> Panel:
        parts: list[Text | str] = []

        if self.harvest_active:
            parts.append(Text.from_markup(f"[blink bold yellow]● Harvesting...[/]"))
        else:
            parts.append(Text("", style="dim"))

        if self.harvest_current:
            parts.append(Text.from_markup(f"  [bold]Current:[/] {self.harvest_current}"))

        # Show recent addresses
        parts.append("")
        recent = self.harvest_addresses[-8:] if self.harvest_addresses else []
        for addr in recent:
            parts.append(Text.from_markup(f"  [green]✓[/] {addr}"))

        parts.append("")
        parts.append(Text.from_markup(f"  [bold]{self.harvest_total}[/] addresses harvested"))

        return Panel(
            Group(*parts),
            title="[bold blue]HARVEST[/] — Addresses",
            border_style="blue",
        )

    def _build_capture_panel(self) -> Panel:
        parts: list[Text | str] = []

        if self.capture_active:
            parts.append(Text.from_markup(f"[blink bold yellow]● Searching Street View...[/]"))
        else:
            parts.append(Text("", style="dim"))

        if self.capture_current:
            parts.append(Text.from_markup(f"  [bold]Address:[/] {self.capture_current}"))
        if self.capture_current_coords:
            parts.append(Text.from_markup(f"  [dim]{self.capture_current_coords}[/]"))
        if self.capture_status:
            parts.append(Text.from_markup(f"  {self.capture_status}"))

        # Recent captures
        parts.append("")
        recent = self.capture_log[-5:] if self.capture_log else []
        for addr, img, status in recent:
            short_addr = addr[:35] + "..." if len(addr) > 35 else addr
            if status == "captured":
                parts.append(Text.from_markup(f"  [green]✓[/] {short_addr} → [cyan]{img}[/]"))
            else:
                parts.append(Text.from_markup(f"  [red]✗[/] {short_addr} [dim](skipped)[/]"))

        parts.append("")
        parts.append(Text.from_markup(
            f"  [bold]{self.capture_done}[/] captured  [dim]|[/]  [yellow]{self.capture_skipped}[/] skipped  [dim]|[/]  {self.capture_total} total"
        ))

        return Panel(
            Group(*parts),
            title="[bold green]CAPTURE[/] — Street View",
            border_style="green",
        )

    def _build_describe_panel(self) -> Panel:
        parts: list[Text | str] = []

        if self.describe_active:
            parts.append(Text.from_markup("[blink bold yellow]● Claude Vision analyzing...[/]"))
        else:
            parts.append(Text("", style="dim"))

        if self.describe_current:
            parts.append(Text.from_markup(f"  [bold]Address:[/] {self.describe_current}"))
        if self.describe_image_path:
            parts.append(Text.from_markup(f"  [bold]Image:[/]   [cyan]{self.describe_image_path}[/]"))

        if self.describe_obj:
            parts.append("")
            obj_text = _format_json_colored(self.describe_obj)
            parts.append(obj_text)
        elif self.describe_active and self.describe_current:
            parts.append("")
            parts.append(Text.from_markup("  [dim italic]Waiting for Claude response...[/]"))

        parts.append("")
        parts.append(Text.from_markup(
            f"  [bold]{self.describe_done}[/] described  [dim]|[/]  [yellow]{self.describe_skipped}[/] skipped  [dim]|[/]  {self.describe_total} total"
        ))

        return Panel(
            Group(*parts),
            title="[bold magenta]DESCRIBE[/] — Property Assessment (Claude Vision)",
            border_style="magenta",
        )

    def _build_score_panel(self) -> Panel:
        parts: list[Text | str] = []

        if self.score_active:
            parts.append(Text.from_markup("[blink bold yellow]● Scoring agent evaluating...[/]"))
        else:
            parts.append(Text("", style="dim"))

        if self.score_current:
            parts.append(Text.from_markup(f"  [bold]Address:[/] {self.score_current}"))

        if self.score_obj:
            parts.append("")
            obj_text = _format_score_colored(self.score_obj)
            parts.append(obj_text)
        elif self.score_active and self.score_current:
            parts.append("")
            parts.append(Text.from_markup("  [dim italic]Waiting for scoring response...[/]"))

        parts.append("")
        parts.append(Text.from_markup(
            f"  [bold]{self.score_done}[/] scored  [dim]|[/]  "
            f"[green]{self.score_qualified}[/] qualified  [dim]|[/]  {self.score_total} total"
        ))

        return Panel(
            Group(*parts),
            title="[bold yellow]SCORE[/] — Wholesale Lead Assessment",
            border_style="yellow",
        )

    def _build_results_panel(self) -> Panel:
        if not self.results:
            return Panel(
                Text("  Waiting for scored leads...", style="dim"),
                title="[bold green]QUALIFIED LEADS[/]",
                border_style="green",
            )

        table = Table(show_header=True, expand=True, padding=(0, 1))
        table.add_column("Address", style="cyan", max_width=40)
        table.add_column("Distress", justify="center")
        table.add_column("Potential", justify="center")
        table.add_column("Confidence", justify="center")
        table.add_column("Discount", justify="center")

        for r in self.results[-5:]:
            distress = r.get("distress_score", "?")
            d_style = "bold red" if (isinstance(distress, int) and distress >= 7) else "yellow" if (isinstance(distress, int) and distress >= 5) else ""
            table.add_row(
                str(r.get("address", ""))[:40],
                Text(str(distress), style=d_style),
                str(r.get("wholesale_potential", "")),
                f"{r.get('confidence', 0):.0%}" if r.get("confidence") else "—",
                str(r.get("estimated_discount_range", "—")),
            )

        return Panel(table, title=f"[bold green]QUALIFIED LEADS[/] ({len(self.results)} found)", border_style="green")


def _format_json_colored(obj: dict) -> Text:
    """Format a property description dict as colored key-value lines."""
    lines = []
    condition_colors = {
        "good": "green", "fair": "yellow", "poor": "red", "damaged": "bold red",
        "maintained": "green", "overgrown": "red", "dead": "bold red",
        "boarded": "bold red", "not_visible": "dim",
        "well_maintained": "green", "average": "yellow",
        "neglected": "red", "severely_neglected": "bold red",
        "yes": "red", "no": "green", "unclear": "yellow",
    }

    for key, val in obj.items():
        if key in ("damage_list", "summary"):
            continue
        color = condition_colors.get(str(val), "white")
        lines.append(f"  [dim]{key}:[/] [{color}]{val}[/]")

    if obj.get("damage_list"):
        dmg = obj["damage_list"]
        if isinstance(dmg, list):
            dmg = ", ".join(dmg)
        lines.append(f"  [dim]damage_list:[/] [red]{dmg}[/]")

    if obj.get("summary"):
        lines.append(f"  [dim]summary:[/] [italic]{obj['summary']}[/]")

    score = obj.get("distress_score", "?")
    score_bar = ""
    if isinstance(score, int):
        filled = score
        empty = 10 - score
        score_bar = f"  [dim]distress:[/] [red]{'█' * filled}[/][dim]{'░' * empty}[/] {score}/10"
        lines.append(score_bar)

    return Text.from_markup("\n".join(lines))


def _format_score_colored(obj: dict) -> Text:
    lines = []
    qualified = obj.get("qualified", False)
    q_text = "[bold green]✓ QUALIFIED[/]" if qualified else "[red]✗ NOT QUALIFIED[/]"
    lines.append(f"  {q_text}")

    confidence = obj.get("confidence", 0)
    conf_color = "green" if confidence >= 0.7 else "yellow" if confidence >= 0.4 else "red"
    lines.append(f"  [dim]confidence:[/] [{conf_color}]{confidence:.0%}[/]")

    pot = obj.get("wholesale_potential", "low")
    pot_color = {"high": "bold green", "medium": "yellow", "low": "red"}.get(pot, "white")
    lines.append(f"  [dim]wholesale_potential:[/] [{pot_color}]{pot}[/]")

    discount = obj.get("estimated_discount_range", "")
    if discount:
        lines.append(f"  [dim]estimated_discount:[/] [cyan]{discount}[/]")

    reasoning = obj.get("reasoning", "")
    if reasoning:
        lines.append(f"  [dim]reasoning:[/] [italic]{reasoning[:120]}[/]")

    return Text.from_markup("\n".join(lines))


# ── Pipeline execution ───────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def run_debug(config: SateliteConfig, city_query: str, limit: int | None = None) -> None:
    """Run the full pipeline with a rich debug TUI."""
    import requests as req_lib
    from streetlevel import streetview

    console = Console()
    state = PipelineState()

    # Validate anthropic key upfront
    api_key = get_anthropic_key()

    init_db(config.pipeline.db_path)
    conn = get_connection(config.pipeline.db_path)

    with Live(state.build_layout(), console=console, refresh_per_second=8, screen=True) as live:

        def refresh():
            live.update(state.build_layout())

        # ── STAGE 1: HARVEST ─────────────────────────────────────────────
        state.harvest_active = True
        refresh()

        display_name, geo_state, bbox = lookup_city_bbox(city_query)
        city_name = display_name.split(",")[0].strip()
        city_id = insert_city(conn, city_name, geo_state, bbox.model_dump())

        cells = bbox.split_into_cells(config.overpass.cell_size)
        session = req_lib.Session()
        session.headers.update({"User-Agent": "satelite-lead-finder/0.1"})

        total_inserted = 0
        effective_limit = limit or 20  # Debug default to 20

        for cell in cells:
            if total_inserted >= effective_limit:
                break

            state.harvest_current = f"Querying cell ({cell.south:.4f}, {cell.west:.4f})..."
            refresh()

            elements = _query_cell(session, config.overpass.endpoint, cell, config.overpass.timeout)
            addresses = _parse_elements(elements, city_name, geo_state)

            remaining = effective_limit - total_inserted
            addresses = addresses[:remaining]

            if addresses:
                inserted = insert_addresses(conn, city_id, addresses)
                total_inserted += inserted
                for a in addresses[:inserted]:
                    fa = a.get("full_address", "")
                    state.harvest_addresses.append(fa)
                    state.harvest_total = total_inserted
                    refresh()

            if total_inserted < effective_limit:
                time.sleep(config.overpass.delay_between_queries)

        state.harvest_active = False
        state.harvest_current = ""
        refresh()

        # ── STAGE 2: CAPTURE ─────────────────────────────────────────────
        state.capture_active = True
        refresh()

        harvested = get_addresses_by_status(conn, city_id, "harvested")
        state.capture_total = len(harvested)

        city_slug = slugify_city(city_name)
        image_dir = Path(config.pipeline.image_dir) / city_slug
        image_dir.mkdir(parents=True, exist_ok=True)

        for addr in harvested:
            addr_id = addr["id"]
            lat, lng = addr["lat"], addr["lng"]
            full_address = addr.get("full_address", "")

            state.capture_current = full_address
            state.capture_current_coords = f"lat: {lat:.5f}, lng: {lng:.5f}"
            state.capture_status = "[yellow]Finding panorama...[/]"
            refresh()

            try:
                pano = streetview.find_panorama(lat, lng, radius=config.streetview.search_radius)
            except Exception:
                pano = None

            if pano is None:
                update_address_status(conn, addr_id, "skipped")
                state.capture_skipped += 1
                state.capture_status = "[red]No panorama found[/]"
                state.capture_log.append((full_address, "", "skipped"))
                refresh()
                time.sleep(0.3)
                continue

            state.capture_status = f"[cyan]Downloading pano {pano.id}...[/]"
            refresh()

            try:
                pano_image = asyncio.run(
                    _download_pano(pano, config.streetview.zoom)
                )
            except Exception:
                update_address_status(conn, addr_id, "skipped")
                state.capture_skipped += 1
                state.capture_log.append((full_address, "", "skipped"))
                refresh()
                continue

            if pano_image is None:
                update_address_status(conn, addr_id, "skipped")
                state.capture_skipped += 1
                state.capture_log.append((full_address, "", "skipped"))
                refresh()
                continue

            bearing = calculate_bearing(pano.lat, pano.lon, lat, lng)
            cropped = _crop_panorama_facing(pano_image, bearing)

            addr_hash = hashlib.sha256(full_address.encode()).hexdigest()[:12]
            out_path = image_dir / f"{addr_hash}.jpg"
            cropped.save(str(out_path), "JPEG", quality=85)

            capture_date = str(pano.date) if hasattr(pano, "date") and pano.date else None
            pano_record = {
                "pano_id": str(pano.id),
                "lat": pano.lat,
                "lng": pano.lon,
                "heading": bearing,
                "capture_date": capture_date,
                "image_path": str(out_path),
                "zoom_level": config.streetview.zoom,
            }
            insert_image(conn, addr_id, pano_record)
            update_address_status(conn, addr_id, "captured")

            state.capture_done += 1
            rel_path = str(out_path)
            state.capture_status = f"[green]✓ Saved → {rel_path}[/]"
            state.capture_log.append((full_address, str(out_path.name), "captured"))
            refresh()
            time.sleep(0.2)

        state.capture_active = False
        state.capture_current = ""
        state.capture_status = ""
        refresh()

        # ── STAGE 3: DESCRIBE ────────────────────────────────────────────
        import anthropic

        state.describe_active = True
        refresh()

        prompts_dir = Path(__file__).parent.parent.parent.parent / "prompts"
        describe_prompt = (prompts_dir / "describe.txt").read_text()
        score_prompt_template = (prompts_dir / "score.txt").read_text()

        client = anthropic.Anthropic()
        model = config.anthropic.model

        captured = get_addresses_by_status(conn, city_id, "captured")
        state.describe_total = len(captured)

        for addr in captured:
            addr_id = addr["id"]
            full_address = addr.get("full_address", "")

            state.describe_current = full_address
            state.describe_obj = None
            refresh()

            # Get image for this address
            image_row = conn.execute(
                "SELECT * FROM images WHERE address_id=? ORDER BY id DESC LIMIT 1",
                (addr_id,),
            ).fetchone()

            if not image_row:
                state.describe_skipped += 1
                refresh()
                continue

            image_row = dict(image_row)
            image_path = Path(image_row["image_path"])
            state.describe_image_path = str(image_path)
            refresh()

            if not image_path.exists():
                state.describe_skipped += 1
                refresh()
                continue

            try:
                image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

                message = client.messages.create(
                    model=model,
                    max_tokens=config.anthropic.max_tokens,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                            {"type": "text", "text": describe_prompt},
                        ],
                    }],
                )

                raw_response = message.content[0].text
                parsed = _parse_json_response(raw_response)

                if parsed is None:
                    state.describe_skipped += 1
                    refresh()
                    continue

                desc = PropertyDescription(**parsed)
                desc_dict = desc.model_dump()

                # Show the filled object
                state.describe_obj = desc_dict
                refresh()
                time.sleep(1.5)  # Pause so user can read

                insert_description(conn, addr_id, image_row["id"], desc_dict, raw_response, model)
                update_address_status(conn, addr_id, "described")
                state.describe_done += 1

            except Exception as e:
                logger.debug("Describe error: %s", e)
                state.describe_skipped += 1

            refresh()

        state.describe_active = False
        state.describe_current = ""
        refresh()

        # ── STAGE 4: SCORE ───────────────────────────────────────────────
        state.score_active = True
        refresh()

        described = get_addresses_by_status(conn, city_id, "described")
        state.score_total = len(described)

        for addr in described:
            addr_id = addr["id"]
            full_address = addr.get("full_address", "")

            state.score_current = full_address
            state.score_obj = None
            refresh()

            desc_row = conn.execute(
                "SELECT * FROM descriptions WHERE address_id=? ORDER BY id DESC LIMIT 1",
                (addr_id,),
            ).fetchone()

            if not desc_row:
                state.score_done += 1
                refresh()
                continue

            desc_row = dict(desc_row)

            damage_list = desc_row.get("damage_list", "[]")
            if isinstance(damage_list, str):
                try:
                    damage_list = json.loads(damage_list)
                except json.JSONDecodeError:
                    damage_list = []

            try:
                formatted_prompt = score_prompt_template.format(
                    full_address=full_address,
                    roof_condition=desc_row.get("roof_condition", "not_visible"),
                    paint_condition=desc_row.get("paint_condition", "not_visible"),
                    yard_condition=desc_row.get("yard_condition", "not_visible"),
                    driveway_condition=desc_row.get("driveway_condition", "not_visible"),
                    windows_condition=desc_row.get("windows_condition", "not_visible"),
                    vacancy_signs=desc_row.get("vacancy_signs", "unclear"),
                    maintenance_level=desc_row.get("maintenance_level", "average"),
                    damage_list=", ".join(damage_list) if damage_list else "None noted",
                    distress_score=desc_row.get("distress_score", 5),
                    summary=desc_row.get("summary", ""),
                )

                message = client.messages.create(
                    model=model,
                    max_tokens=config.anthropic.max_tokens,
                    messages=[{"role": "user", "content": formatted_prompt}],
                )

                raw_response = message.content[0].text
                parsed = _parse_json_response(raw_response)

                if parsed is None:
                    state.score_done += 1
                    refresh()
                    continue

                score = LeadScore(**parsed)
                score_dict = score.model_dump()

                # Show the score object
                state.score_obj = score_dict
                refresh()
                time.sleep(1.5)  # Pause so user can read

                insert_score(conn, addr_id, desc_row["id"], score_dict, model)
                update_address_status(conn, addr_id, "scored")
                state.score_done += 1

                if score.qualified:
                    state.score_qualified += 1
                    state.results.append({
                        "address": full_address,
                        "distress_score": desc_row.get("distress_score"),
                        "wholesale_potential": score.wholesale_potential,
                        "confidence": score.confidence,
                        "estimated_discount_range": score.estimated_discount_range,
                    })

            except Exception as e:
                logger.debug("Score error: %s", e)
                state.score_done += 1

            refresh()

        state.score_active = False
        state.score_current = ""
        refresh()

        # Final pause to show results
        time.sleep(3)

    conn.close()

    # Print final summary outside Live
    console.print()
    console.print("[bold green]Pipeline complete![/]")
    console.print(f"  Harvested: {state.harvest_total}")
    console.print(f"  Captured:  {state.capture_done} ({state.capture_skipped} skipped)")
    console.print(f"  Described: {state.describe_done} ({state.describe_skipped} skipped)")
    console.print(f"  Scored:    {state.score_done}")
    console.print(f"  [bold green]Qualified:  {state.score_qualified}[/]")

    if state.results:
        console.print()
        table = Table(title="Qualified Leads for Outreach")
        table.add_column("Address", style="cyan")
        table.add_column("Distress", justify="center")
        table.add_column("Potential", justify="center")
        table.add_column("Confidence", justify="center")
        table.add_column("Discount", justify="center")
        for r in state.results:
            table.add_row(
                str(r["address"]),
                str(r["distress_score"]),
                str(r["wholesale_potential"]),
                f"{r['confidence']:.0%}" if r.get("confidence") else "—",
                str(r.get("estimated_discount_range", "—")),
            )
        console.print(table)


async def _download_pano(pano, zoom: int) -> Image.Image | None:
    """Download a panorama using aiohttp."""
    import aiohttp
    from streetlevel import streetview

    async with aiohttp.ClientSession() as session:
        return await streetview.get_panorama_async(pano, zoom=zoom, session=session)
