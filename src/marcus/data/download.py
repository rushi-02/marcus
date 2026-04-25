"""Download Stoic source texts from Standard Ebooks, Project Gutenberg, and HuggingFace.

Supported sources:
- Marcus Aurelius' Meditations (Standard Ebooks, George Long translation)
- Epictetus' Discourses (Project Gutenberg)
- Epictetus' Enchiridion (Project Gutenberg)
- Seneca's Letters to Lucilius (Project Gutenberg)
- HuggingFace: Philosophy Data Project (filtered for Stoic authors)
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import track

console = Console()

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------

SOURCES = {
    "meditations": {
        "url": "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",
        "filename": "meditations.txt",
        "description": "Marcus Aurelius — Meditations (George Long translation)",
    },
    "discourses": {
        "url": "https://www.gutenberg.org/cache/epub/10661/pg10661.txt",
        "filename": "discourses.txt",
        "description": "Epictetus — Discourses (George Long translation)",
    },
    "enchiridion": {
        "url": "https://www.gutenberg.org/cache/epub/45109/pg45109.txt",
        "filename": "enchiridion.txt",
        "description": "Epictetus — Enchiridion (Elizabeth Carter translation)",
    },
    "seneca_morals": {
        "url": "https://www.gutenberg.org/cache/epub/56075/pg56075.txt",
        "filename": "seneca_morals.txt",
        "description": "Seneca — Morals of a Happy Life, Benefits, Anger and Clemency",
    },
}


def download_text(url: str, output_path: Path) -> Path:
    """Download a plain text file from a URL."""
    if output_path.exists():
        console.print(f"  [dim]Already exists:[/dim] {output_path.name}")
        return output_path

    console.print(f"  [cyan]Downloading:[/cyan] {url}")
    response = httpx.get(url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding="utf-8")
    console.print(f"  [green]Saved:[/green] {output_path} ({len(response.text):,} chars)")
    return output_path


def download_all_sources(data_dir: Path) -> dict[str, Path]:
    """Download all Stoic source texts to data/raw/.

    Returns:
        Dict mapping source name to local file path.
    """
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for name, source in SOURCES.items():
        console.print(f"\n[bold]{source['description']}[/bold]")
        output_path = raw_dir / source["filename"]
        try:
            downloaded[name] = download_text(source["url"], output_path)
        except Exception as e:
            console.print(f"  [red]Failed:[/red] {e}")

    console.print(f"\n[bold green]Downloaded {len(downloaded)}/{len(SOURCES)} sources.[/bold green]")
    return downloaded


def strip_gutenberg_header_footer(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    # Find start marker
    start_markers = [
        r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG",
        r"\*\*\*START OF (THE|THIS) PROJECT GUTENBERG",
    ]
    for marker in start_markers:
        match = re.search(marker, text)
        if match:
            text = text[match.end() :]
            # Skip past the line containing the marker
            newline_pos = text.find("\n")
            if newline_pos != -1:
                text = text[newline_pos + 1 :]
            break

    # Find end marker
    end_markers = [
        r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG",
        r"\*\*\*END OF (THE|THIS) PROJECT GUTENBERG",
        r"End of (the )?Project Gutenberg",
    ]
    for marker in end_markers:
        match = re.search(marker, text)
        if match:
            text = text[: match.start()]
            break

    return text.strip()
