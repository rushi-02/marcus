"""Clean and chunk Stoic source texts into passage-level segments.

Pipeline: raw text → strip boilerplate → normalize → chunk into passages → JSONL
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from rich.console import Console

from marcus.data.download import strip_gutenberg_header_footer

console = Console()


def normalize_text(text: str) -> str:
    """Normalize unicode, whitespace, and encoding artifacts."""
    # Normalize unicode (NFC)
    text = unicodedata.normalize("NFC", text)

    # Replace fancy quotes and dashes
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": " -- ",
        "\u2026": "...",
        "\r\n": "\n",
        "\r": "\n",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Collapse multiple blank lines into two newlines (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace from lines
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


def chunk_into_passages(
    text: str,
    min_words: int = 30,
    max_words: int = 300,
) -> list[str]:
    """Split text into paragraph-level passages suitable for training.

    Strategy:
    1. Split on double newlines (paragraph boundaries)
    2. Merge short paragraphs with their neighbors
    3. Split long paragraphs at sentence boundaries
    """
    # Split into raw paragraphs
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    passages = []
    buffer = ""

    for para in raw_paragraphs:
        # Skip very short lines that are likely headers/numbers
        if len(para.split()) < 3 and not buffer:
            continue

        if buffer:
            buffer += " " + para
        else:
            buffer = para

        word_count = len(buffer.split())

        if word_count >= min_words:
            if word_count <= max_words:
                passages.append(buffer)
                buffer = ""
            else:
                # Split at sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", buffer)
                current = ""
                for sent in sentences:
                    if len((current + " " + sent).split()) <= max_words:
                        current = (current + " " + sent).strip()
                    else:
                        if current and len(current.split()) >= min_words:
                            passages.append(current)
                        current = sent

                if current and len(current.split()) >= min_words:
                    passages.append(current)
                buffer = ""

    # Don't lose the trailing buffer
    if buffer and len(buffer.split()) >= min_words:
        passages.append(buffer)

    return passages


def clean_source(raw_path: Path, source_name: str) -> list[dict]:
    """Clean a single source text and return passage dicts.

    Returns:
        List of {"source": str, "passage": str} dicts.
    """
    console.print(f"  [cyan]Cleaning:[/cyan] {raw_path.name}")
    text = raw_path.read_text(encoding="utf-8")

    # Strip Gutenberg boilerplate
    text = strip_gutenberg_header_footer(text)

    # Normalize
    text = normalize_text(text)

    # Chunk
    passages = chunk_into_passages(text)

    console.print(f"  [green]Extracted:[/green] {len(passages)} passages from {source_name}")

    return [{"source": source_name, "passage": p} for p in passages]


def clean_all_sources(data_dir: Path) -> list[dict]:
    """Clean all raw source texts and save to data/processed/stoic_passages.jsonl.

    Returns:
        Full list of passage dicts.
    """
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_passages = []

    source_files = {
        "meditations": "meditations.txt",
        "discourses": "discourses.txt",
        "enchiridion": "enchiridion.txt",
        "seneca_morals": "seneca_morals.txt",
        "seneca_letters": "seneca_letters.txt",  # Inwood translation, from PDF
    }

    for source_name, filename in source_files.items():
        raw_path = raw_dir / filename
        if raw_path.exists():
            passages = clean_source(raw_path, source_name)
            all_passages.extend(passages)
        else:
            console.print(f"  [yellow]Skipping:[/yellow] {filename} (not found)")

    # Save
    output_path = processed_dir / "stoic_passages.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in all_passages:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    console.print(
        f"\n[bold green]Total: {len(all_passages)} passages "
        f"saved to {output_path}[/bold green]"
    )
    return all_passages
