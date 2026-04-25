#!/usr/bin/env python3
"""Standalone script: evaluate Marcus adapter quality vs base model.

Usage:
    uv run python scripts/eval_model.py --adapter-path adapters/marcus-sft-v1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marcus.config import load_config
from marcus.training.sft import evaluate_adapter
from rich.console import Console
from rich.table import Table

console = Console()

TEST_PROMPTS = [
    "I feel completely overwhelmed by everything at work. I don't know where to start.",
    "Someone I trusted deeply just betrayed me. I'm furious and heartbroken.",
    "I'm terrified of dying before I achieve anything meaningful with my life.",
    "My boss humiliated me in front of my whole team. I want to quit.",
    "I keep procrastinating on the things that matter most to me. I feel like a failure.",
    "I'm anxious about a big exam tomorrow. I don't think I'm ready.",
    "I'm grieving the loss of someone I loved. I don't know how to go on.",
    "I feel angry all the time and I don't know how to control it.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Marcus adapter quality")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--verbose", action="store_true", help="Print all responses")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    console.print(f"[bold cyan]Evaluating adapter:[/bold cyan] {args.adapter_path}\n")

    results = evaluate_adapter(config, args.adapter_path, TEST_PROMPTS)

    if args.verbose:
        console.print("\n[bold]Sample Responses[/bold]\n")
        for i, (base, adapter) in enumerate(
            zip(results["base"][:3], results["adapter"][:3])
        ):
            console.print(f"[dim]Prompt:[/dim] {TEST_PROMPTS[i]}\n")
            console.print(f"[yellow]Base:[/yellow]    {base['response']}\n")
            console.print(f"[cyan]Adapter:[/cyan] {adapter['response']}\n")
            console.print("─" * 60)

    # Print scores table
    table = Table(title="Reward Scores by Prompt")
    table.add_column("Prompt", style="dim", max_width=40)
    table.add_column("Base", justify="center")
    table.add_column("Adapter", justify="center")
    table.add_column("Δ", justify="center")

    for base, adapter in zip(results["base"], results["adapter"]):
        delta = adapter["score"] - base["score"]
        color = "green" if delta > 0 else "red"
        table.add_row(
            base["prompt"][:40],
            f"{base['score']:.3f}",
            f"{adapter['score']:.3f}",
            f"[{color}]{delta:+.3f}[/{color}]",
        )

    console.print(table)


if __name__ == "__main__":
    main()
