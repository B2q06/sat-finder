"""Stage 3 — Describe property condition using Claude Vision."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import anthropic
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from satelite.config import get_anthropic_key
from satelite.db import get_connection, init_db, get_addresses_by_status, get_city_id, update_address_status, insert_description
from satelite.models import PropertyDescription, SateliteConfig

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"


def _load_prompt() -> str:
    return (_PROMPTS_DIR / "describe.txt").read_text()


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


def run_describe(config: SateliteConfig, city_query: str, limit: int | None = None) -> None:
    """Describe captured property images using Claude Vision."""
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

    addresses = get_addresses_by_status(conn, city_id, "captured", limit)
    if not addresses:
        console.print("[yellow]No captured addresses to describe.[/yellow]")
        conn.close()
        return

    console.print(f"[bold]Describing {len(addresses)} properties for {city_name}[/bold]")

    prompt_text = _load_prompt()
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
        task = progress.add_task("Describing", total=len(addresses), skipped=0)

        for addr in addresses:
            address_id = addr["id"]

            # Get the most recent image for this address
            image_row = conn.execute(
                "SELECT * FROM images WHERE address_id=? ORDER BY id DESC LIMIT 1",
                (address_id,),
            ).fetchone()

            if not image_row:
                console.print(f"[yellow]No image for address {address_id}, skipping[/yellow]")
                skipped += 1
                progress.update(task, advance=1, skipped=skipped)
                continue

            image_row = dict(image_row)
            image_path = Path(image_row["image_path"])
            if not image_path.exists():
                console.print(f"[yellow]Image not found: {image_path}, skipping[/yellow]")
                skipped += 1
                progress.update(task, advance=1, skipped=skipped)
                continue

            try:
                image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

                message = client.messages.create(
                    model=model,
                    max_tokens=config.anthropic.max_tokens,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt_text,
                            },
                        ],
                    }],
                )

                raw_response = message.content[0].text
                parsed = _parse_json_response(raw_response)

                if parsed is None:
                    console.print(f"[yellow]Failed to parse JSON for address {address_id}, skipping[/yellow]")
                    skipped += 1
                    progress.update(task, advance=1, skipped=skipped)
                    continue

                # Validate through Pydantic
                desc = PropertyDescription(**parsed)
                desc_dict = desc.model_dump()

                insert_description(conn, address_id, image_row["id"], desc_dict, raw_response, model)
                update_address_status(conn, address_id, "described")
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
        f"\n[bold green]Describe complete.[/bold green] "
        f"{processed} described, {skipped} skipped."
    )
