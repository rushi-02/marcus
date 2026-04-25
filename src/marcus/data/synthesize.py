"""Generate synthetic instruction-response training pairs from Stoic passages.

Uses an LLM (Claude API by default) to generate realistic user questions
and Marcus Aurelius responses from raw philosophical passages.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are an expert at creating high-quality training data for
fine-tuning a Stoic AI philosopher. Your task is to generate realistic conversational
pairs where a user shares a life situation and Marcus Aurelius responds with wisdom."""

SYNTHESIS_PROMPT = """Given this passage from Stoic philosophy:

\"{passage}\"

Generate a realistic conversational pair where:
1. A modern person describes a real-life struggle or asks for guidance
2. Marcus Aurelius responds with Stoic wisdom inspired by the passage

Rules for the user message:
- Sound natural and emotional (stressed, anxious, confused, angry, sad)
- Be about a specific modern situation (work, relationships, health, loss, decisions)
- Category: {category}

Rules for Marcus's response:
- First person, as Marcus Aurelius speaking directly to a friend
- Reference specific Stoic principles (dichotomy of control, amor fati, memento mori, etc.)
- Warm but direct tone — empathize first, then guide
- 50-150 words (suitable for spoken dialogue)
- Do NOT quote the passage verbatim

Output ONLY valid JSON (no markdown):
{{"user": "...", "marcus": "..."}}"""

CATEGORIES = [
    "work stress and career anxiety",
    "relationship difficulties",
    "fear and anxiety about the future",
    "anger and frustration",
    "grief and loss",
    "decision-making under uncertainty",
    "self-doubt and imposter syndrome",
    "health concerns and mortality",
    "daily Stoic practice",
    "finding purpose and meaning",
    "dealing with criticism",
    "managing expectations",
]


def _build_prompt(passage: str) -> str:
    """Build a synthesis prompt for a given passage."""
    category = random.choice(CATEGORIES)
    return SYNTHESIS_PROMPT.format(passage=passage, category=category)


async def generate_pairs_anthropic(
    passages: list[str],
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    max_concurrent: int = 5,
    pairs_per_passage: int = 1,
) -> list[dict]:
    """Generate instruction-response pairs using the Anthropic API.

    Args:
        passages: List of Stoic passage texts.
        api_key: Anthropic API key.
        model: Model to use for generation.
        max_concurrent: Max concurrent API calls.
        pairs_per_passage: Number of pairs to generate per passage.

    Returns:
        List of {"user": str, "marcus": str} dicts.
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        console.print("[red]Install anthropic: uv pip install 'marcus[train]'[/red]")
        raise

    client = AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[dict] = []
    errors = 0

    async def _generate_one(passage: str) -> dict | None:
        nonlocal errors
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=512,
                    system=SYNTHESIS_SYSTEM,
                    messages=[{"role": "user", "content": _build_prompt(passage)}],
                )
                text = response.content[0].text.strip()
                # Parse JSON — handle potential markdown wrapping
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                pair = json.loads(text)
                if "user" in pair and "marcus" in pair:
                    return pair
                else:
                    errors += 1
                    return None
            except Exception as e:
                errors += 1
                if errors <= 5:
                    console.print(f"  [yellow]Warning:[/yellow] {e}")
                return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Generating pairs from {len(passages)} passages...",
            total=len(passages) * pairs_per_passage,
        )

        for _ in range(pairs_per_passage):
            tasks = [_generate_one(p) for p in passages]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    results.append(result)
                progress.advance(task)

    console.print(
        f"\n[bold green]Generated {len(results)} pairs "
        f"({errors} errors)[/bold green]"
    )
    return results


def generate_pairs_local(
    passages: list[str],
    pairs_per_passage: int = 1,
) -> list[dict]:
    """Generate instruction-response pairs using a local MLX model.

    This is a fallback for when no API key is available. Uses the loaded
    LLM to generate pairs (slower but free).
    """
    from marcus.models.llm import MarcusLLM
    from marcus.config import load_config

    config = load_config()
    llm = MarcusLLM(config.llm)

    results = []
    for i, passage in enumerate(passages):
        console.print(f"  [{i + 1}/{len(passages)}] Generating pair...")
        prompt = _build_prompt(passage)
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            response_text = llm.generate(messages)
            # Try to parse JSON from response
            if "{" in response_text and "}" in response_text:
                json_str = response_text[response_text.index("{") : response_text.rindex("}") + 1]
                pair = json.loads(json_str)
                if "user" in pair and "marcus" in pair:
                    results.append(pair)
        except Exception as e:
            console.print(f"  [yellow]Warning:[/yellow] {e}")

    console.print(f"\n[bold green]Generated {len(results)} pairs locally.[/bold green]")
    return results


def save_synthetic_pairs(pairs: list[dict], output_dir: Path) -> Path:
    """Save synthetic pairs to JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "instruction_pairs.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    console.print(f"[green]Saved {len(pairs)} pairs to {output_path}[/green]")
    return output_path
