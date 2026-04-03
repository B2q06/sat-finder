"""Stage 4 — Score described properties as wholesale leads using Claude."""

from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from satelite.config import get_anthropic_key
from satelite.db import get_connection, init_db, get_addresses_by_status, get_city_id, update_address_status, insert_score
from satelite.models import LeadScore, SateliteConfig

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"


def _load_prompt() -> str:
    return (_PROMPTS_DIR / "score.txt").read_text()


def _parse_json_response(text: str) -> dict | None:
    """Try to parse JSON from Claude's response, handling markdown fences."""
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


def run_score(config: SateliteConfig, city_query: str, limit: int | None = None) -> None:
    """Score described properties as wholesale leads."""
    console = Console(stderr=True)

    get_anthropic_key()

    init_db(config.pipeline.db_path)
    conn = get_connection(config.pipeline.db_path)

    # Fuzzy city lookup
    row = conn.execute(
        "SELECT id, name FROM cities WHERE name LIKE ? LIMIT 1",
        (f"%{city_query.split(',')[0].strip()}%",),
    ).fetchone()
    if not row:
        console.print(f"[red]City not found in database.[/red] Run harvest first.")
        conn.close()
        return
    city_id, city_name = row["id"], row["name"]

    addresses = get_addresses_by_status(conn, city_id, "described", limit)
    if not addresses:
        console.print("[yellow]No described addresses to score.[/yellow]")
        conn.close()
        return

    console.print(f"[bold]Scoring {len(addresses)} properties for {city_name}[/bold]")

    prompt_template = _load_prompt()
    client = anthropic.Anthropic()
    model = config.anthropic.model

    processed = 0
    skipped = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("| {task.fields[skipped]} skipped"),
        console=console,
    ) as progress:
        task = progress.add_task("Scoring", total=len(addresses), skipped=0)

        for addr in addresses:
            address_id = addr["id"]

            # Get the most recent description for this address
            desc_row = conn.execute(
                "SELECT * FROM descriptions WHERE address_id=? ORDER BY id DESC LIMIT 1",
                (address_id,),
            ).fetchone()

            if not desc_row:
                console.print(f"[yellow]No description for address {address_id}, skipping[/yellow]")
                skipped += 1
                progress.update(task, advance=1, skipped=skipped)
                continue

            desc_row = dict(desc_row)

            # Parse damage_list from JSON string
            damage_list = desc_row.get("damage_list", "[]")
            if isinstance(damage_list, str):
                try:
                    damage_list = json.loads(damage_list)
                except json.JSONDecodeError:
                    damage_list = []

            # Format the prompt with description fields
            formatted_prompt = prompt_template.format(
                full_address=addr.get("full_address", "Unknown"),
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

            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=config.anthropic.max_tokens,
                    messages=[{"role": "user", "content": formatted_prompt}],
                )

                raw_response = message.content[0].text
                parsed = _parse_json_response(raw_response)

                if parsed is None:
                    console.print(f"[yellow]Failed to parse JSON for address {address_id}, skipping[/yellow]")
                    skipped += 1
                    progress.update(task, advance=1, skipped=skipped)
                    continue

                # Validate through Pydantic
                score = LeadScore(**parsed)
                score_dict = score.model_dump()

                insert_score(conn, address_id, desc_row["id"], score_dict, model)
                update_address_status(conn, address_id, "scored")
                processed += 1

            except anthropic.APIError as e:
                console.print(f"[red]API error for address {address_id}: {e}[/red]")
                skipped += 1
            except Exception as e:
                console.print(f"[red]Error for address {address_id}: {e}[/red]")
                skipped += 1

            progress.update(task, advance=1, skipped=skipped)

    conn.close()
    console.print(
        f"\n[bold green]Scoring complete.[/bold green] "
        f"{processed} scored, {skipped} skipped."
    )
