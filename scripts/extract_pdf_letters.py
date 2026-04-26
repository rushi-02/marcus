#!/usr/bin/env python3
"""Extract Seneca's Letters to Lucilius from the Inwood PDF edition.

Inwood, Brad (ed/trans). 'Seneca: Selected Philosophical Letters' (Oxford
Clarendon Later Ancient Philosophers series). 436 pages.

Strategy:
1. Read all pages.
2. Skip front matter (intro, TOC, preface) — letters start around page 50.
3. Skip back matter (bibliography, indices) — usually last ~80 pages.
4. Strip footnote refs and page numbers.
5. Save as a single text file in data/raw/seneca_letters.txt.

Run:
    uv run python scripts/extract_pdf_letters.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pypdf import PdfReader
from rich.console import Console

console = Console()


def extract_letters(pdf_path: Path, output_path: Path) -> int:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    console.print(f"[cyan]Reading:[/cyan] {pdf_path.name} ({total_pages} pages)")

    # Heuristic: letters span roughly pages 49 to 350 in this 436-page edition.
    # The intro ends around page 48; commentary/bibliography starts around page 350.
    # Inspecting the file: letter content is the bulk of the volume.
    start_page = 47
    end_page = total_pages - 80  # conservative cutoff for back matter

    pages_text = []
    in_letter = False
    for i in range(start_page, end_page):
        try:
            text = reader.pages[i].extract_text()
        except Exception:
            continue

        if not text:
            continue

        # Detect 'LETTER' markers — these indicate we've reached letter content
        if re.search(r"^\s*LETTER\b", text, re.MULTILINE):
            in_letter = True
        # Detect commentary/back-matter section headers
        if re.search(r"^(BIBLIOGRAPHY|INDEX|COMMENTARY)\b", text, re.MULTILINE):
            in_letter = False
            console.print(f"  [yellow]Stopping at page {i}: hit back matter[/yellow]")
            break

        if not in_letter:
            continue

        pages_text.append(text)

    raw = "\n\n".join(pages_text)
    console.print(f"  [dim]Raw chars before cleaning: {len(raw):,}[/dim]")

    # Clean common PDF artifacts
    cleaned = raw

    # Remove page numbers (standalone digits on their own line)
    cleaned = re.sub(r"^\s*\d{1,4}\s*$", "", cleaned, flags=re.MULTILINE)

    # Remove footnote references like " 12 " sandwiched between letters/words
    # (very rough — drops some legitimate numbers but the Letters don't depend on them)
    cleaned = re.sub(r"(?<=\w)\s\d{1,3}\s(?=\w)", " ", cleaned)

    # Collapse hyphenated line breaks: "philoso-\nphy" → "philosophy"
    cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)

    # Replace single newlines (mid-paragraph) with spaces;
    # keep double newlines as paragraph breaks
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)

    # Collapse multiple spaces
    cleaned = re.sub(r" {2,}", " ", cleaned)

    # Collapse multi-blank-lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Strip footnote markers like "n. 1" or superscript digits that survive
    cleaned = re.sub(r"\bn\. ?\d+\b", "", cleaned)

    # Smart quotes → ASCII
    cleaned = cleaned.translate(str.maketrans({
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": " -- ", "…": "..."
    }))

    cleaned = cleaned.strip()
    console.print(f"  [green]Cleaned chars: {len(cleaned):,}[/green]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")
    console.print(f"  [bold green]Saved to {output_path}[/bold green]")

    # Show a sample
    console.print("\n[dim]Sample (first 600 chars after cleaning):[/dim]")
    console.print(cleaned[:600])

    return len(cleaned)


if __name__ == "__main__":
    pdf_path = Path("data/books/Seneca-Letters.pdf")
    out_path = Path("data/raw/seneca_letters.txt")

    if not pdf_path.exists():
        console.print(f"[red]PDF not found: {pdf_path}[/red]")
        sys.exit(1)

    chars = extract_letters(pdf_path, out_path)
    console.print(f"\n[bold]Extracted {chars:,} chars of Seneca's Letters.[/bold]")
    console.print("Run 'uv run marcus data clean' to chunk into passages.")
