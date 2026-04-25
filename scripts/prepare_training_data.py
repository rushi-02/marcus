#!/usr/bin/env python3
"""Standalone script: run the full data pipeline end-to-end.

Steps:
1. Download Stoic source texts (Meditations, Discourses, Letters, Enchiridion)
2. Clean and chunk into passages
3. Generate synthetic instruction-response pairs via LLM
4. Format into mlx-lm training JSONL

Usage:
    uv run python scripts/prepare_training_data.py --api-key sk-ant-...
    uv run python scripts/prepare_training_data.py --local  # use local MLX model
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marcus.config import load_config
from marcus.data.clean import clean_all_sources
from marcus.data.download import download_all_sources
from marcus.data.format import format_pipeline
from marcus.data.synthesize import (
    generate_pairs_anthropic,
    generate_pairs_local,
    save_synthetic_pairs,
)
from rich.console import Console

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Marcus training data")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--api-key", default=None, help="Anthropic API key")
    parser.add_argument("--local", action="store_true", help="Use local MLX model")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-synthesize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_dir = config.data.data_dir

    # 1. Download
    if not args.skip_download:
        console.print("\n[bold]Step 1/4: Downloading source texts[/bold]")
        download_all_sources(data_dir)

    # 2. Clean
    if not args.skip_clean:
        console.print("\n[bold]Step 2/4: Cleaning and chunking[/bold]")
        clean_all_sources(data_dir)

    # 3. Synthesize
    passages_path = data_dir / "processed" / "stoic_passages.jsonl"
    synthetic_path = data_dir / "synthetic" / "instruction_pairs.jsonl"

    if not args.skip_synthesize:
        console.print("\n[bold]Step 3/4: Generating instruction-response pairs[/bold]")

        passages = []
        with open(passages_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                passages.append(entry["passage"])

        console.print(f"Loaded {len(passages)} passages.")

        if args.local:
            pairs = generate_pairs_local(passages)
        else:
            api_key = args.api_key
            if not api_key:
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                console.print("[red]Set ANTHROPIC_API_KEY or use --local[/red]")
                sys.exit(1)

            pairs = asyncio.run(
                generate_pairs_anthropic(
                    passages=passages[: config.data.target_pairs],
                    api_key=api_key,
                    model=config.data.synthesis_model,
                    pairs_per_passage=config.data.pairs_per_passage,
                )
            )

        save_synthetic_pairs(pairs, data_dir / "synthetic")

    # 4. Format
    console.print("\n[bold]Step 4/4: Formatting for training[/bold]")
    format_pipeline(
        pairs_path=synthetic_path,
        output_dir=data_dir / "training",
        system_prompt_path=config.llm.system_prompt_path,
        val_ratio=config.data.val_ratio,
    )

    console.print("\n[bold green]✓ Data pipeline complete![/bold green]")
    console.print(f"  Training data: {data_dir}/training/train.jsonl")
    console.print("  Next: uv run marcus train sft")


if __name__ == "__main__":
    main()
