#!/usr/bin/env python3
"""Standalone script: download Stoic source texts.

Usage:
    uv run python scripts/download_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marcus.config import load_config
from marcus.data.download import download_all_sources
from rich.console import Console

console = Console()

if __name__ == "__main__":
    config = load_config()
    console.print("[bold cyan]Downloading Stoic source texts...[/bold cyan]")
    paths = download_all_sources(config.data.data_dir)
    console.print(f"\n[bold green]Done. {len(paths)} sources downloaded.[/bold green]")
