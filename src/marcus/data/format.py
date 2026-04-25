"""Format training data into mlx-lm chat JSONL for fine-tuning.

Converts instruction-response pairs into the chat format expected by
mlx-lm LoRA training:

    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rich.console import Console

console = Console()


def load_system_prompt(path: str = "configs/system_prompt.txt") -> str:
    """Load the Marcus Aurelius system prompt."""
    return Path(path).read_text(encoding="utf-8").strip()


def pairs_to_chat_format(
    pairs: list[dict],
    system_prompt: str,
) -> list[dict]:
    """Convert instruction-response pairs to mlx-lm chat JSONL format.

    Args:
        pairs: List of {"user": str, "marcus": str} dicts.
        system_prompt: The system prompt for Marcus's persona.

    Returns:
        List of {"messages": [...]} dicts ready for training.
    """
    formatted = []
    for pair in pairs:
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pair["user"]},
                {"role": "assistant", "content": pair["marcus"]},
            ]
        }
        formatted.append(entry)
    return formatted


def train_val_split(
    data: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split data into training and validation sets.

    Returns:
        (train_data, val_data) tuple.
    """
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
    return shuffled[:split_idx], shuffled[split_idx:]


def save_training_data(
    train_data: list[dict],
    val_data: list[dict],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save training and validation JSONL files.

    mlx-lm LoRA expects files named 'train.jsonl' and 'valid.jsonl'
    in the data directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "valid.jsonl"

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    console.print(
        f"[green]Saved training data:[/green]\n"
        f"  Train: {train_path} ({len(train_data)} examples)\n"
        f"  Valid: {val_path} ({len(val_data)} examples)"
    )
    return train_path, val_path


def format_pipeline(
    pairs_path: Path,
    output_dir: Path,
    system_prompt_path: str = "configs/system_prompt.txt",
    val_ratio: float = 0.1,
) -> tuple[Path, Path]:
    """End-to-end formatting pipeline.

    Reads synthetic pairs → formats to chat → splits → saves.
    """
    # Load pairs
    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    console.print(f"[cyan]Loaded {len(pairs)} instruction-response pairs[/cyan]")

    # Load system prompt
    system_prompt = load_system_prompt(system_prompt_path)

    # Format
    formatted = pairs_to_chat_format(pairs, system_prompt)

    # Split
    train_data, val_data = train_val_split(formatted, val_ratio=val_ratio)

    # Save
    return save_training_data(train_data, val_data, output_dir)
