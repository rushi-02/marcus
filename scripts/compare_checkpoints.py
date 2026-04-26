#!/usr/bin/env python3
"""Compare LoRA checkpoints to find the best one before overfitting kicks in.

For each checkpoint in adapters/<name>/0000{N}_adapters.safetensors, this script:
1. Renames it temporarily to adapters.safetensors (the file mlx-lm loads)
2. Loads the model with that adapter
3. Generates responses to a held-out test set
4. Scores them with composite_reward()
5. Reports the score table

Usage:
    uv run python scripts/compare_checkpoints.py --adapter-dir adapters/marcus-sft-v1
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

console = Console()

TEST_PROMPTS = [
    "I just bombed an interview I worked weeks for. I feel useless.",
    "My partner of five years just left me without warning.",
    "I keep procrastinating on the things I care about most.",
    "I'm anxious about a big presentation tomorrow.",
    "My elderly mother is dying and I'm terrified.",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default="adapters/marcus-sft-v1")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def list_checkpoints(adapter_dir: Path) -> list[tuple[int, Path]]:
    """Return [(iter_num, path), ...] sorted by iter."""
    files = sorted(adapter_dir.glob("0*_adapters.safetensors"))
    out = []
    for f in files:
        try:
            iter_num = int(f.stem.split("_")[0])
            out.append((iter_num, f))
        except (ValueError, IndexError):
            continue
    return out


def evaluate_checkpoint(
    config,
    adapter_dir: Path,
    checkpoint: Path,
    test_prompts: list[str],
    system_prompt: str,
) -> dict:
    """Swap in a specific checkpoint and evaluate it."""
    from marcus.config import LLMConfig
    from marcus.models.llm import MarcusLLM
    from marcus.rewards.composite import composite_reward
    from marcus.rewards.stoic_alignment import stoic_alignment_score
    from marcus.rewards.coherence import length_reward, persona_consistency_score

    # Swap the named adapter file
    main_adapter = adapter_dir / "adapters.safetensors"
    backup = adapter_dir / "_eval_backup.safetensors"

    # Backup current
    if main_adapter.exists():
        shutil.copy(main_adapter, backup)
    shutil.copy(checkpoint, main_adapter)

    try:
        llm_config = LLMConfig(
            model_id=config.llm.model_id,
            adapter_path=str(adapter_dir),
            max_tokens=200,
            temperature=0.7,
        )
        llm = MarcusLLM(llm_config)

        scores = {"composite": [], "stoic": [], "persona": [], "length": []}
        responses = []
        t0 = time.time()
        for prompt in test_prompts:
            response = llm.generate([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ])
            scores["composite"].append(composite_reward(response))
            scores["stoic"].append(stoic_alignment_score(response))
            scores["persona"].append(persona_consistency_score(response))
            scores["length"].append(length_reward(response))
            responses.append(response)
        elapsed = time.time() - t0

        return {
            "scores": {k: sum(v) / len(v) for k, v in scores.items()},
            "responses": responses,
            "elapsed": elapsed,
        }
    finally:
        # Restore backup
        if backup.exists():
            shutil.copy(backup, main_adapter)
            backup.unlink()


def main():
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)

    if not adapter_dir.exists():
        console.print(f"[red]Adapter dir not found: {adapter_dir}[/red]")
        sys.exit(1)

    checkpoints = list_checkpoints(adapter_dir)
    if not checkpoints:
        console.print(f"[red]No checkpoints found in {adapter_dir}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Found {len(checkpoints)} checkpoints[/cyan]")

    from marcus.config import load_config
    from marcus.pipeline.conversation import load_system_prompt

    config = load_config(args.config)
    system_prompt = load_system_prompt(config.llm.system_prompt_path)

    table = Table(title="Checkpoint Quality Comparison")
    table.add_column("Iter", justify="right")
    table.add_column("Composite", justify="right")
    table.add_column("Stoic", justify="right")
    table.add_column("Persona", justify="right")
    table.add_column("Length", justify="right")
    table.add_column("Time", justify="right")

    best_iter = None
    best_score = -1.0
    all_results = {}

    for iter_num, checkpoint in checkpoints:
        console.print(f"\n[dim]Evaluating iter {iter_num}...[/dim]")
        result = evaluate_checkpoint(
            config, adapter_dir, checkpoint, TEST_PROMPTS, system_prompt,
        )
        scores = result["scores"]
        all_results[iter_num] = result

        if scores["composite"] > best_score:
            best_score = scores["composite"]
            best_iter = iter_num

        table.add_row(
            str(iter_num),
            f"{scores['composite']:.3f}",
            f"{scores['stoic']:.3f}",
            f"{scores['persona']:.3f}",
            f"{scores['length']:.3f}",
            f"{result['elapsed']:.1f}s",
        )

    console.print()
    console.print(table)
    console.print(
        f"\n[bold green]Best checkpoint: iter {best_iter} "
        f"(composite score {best_score:.3f})[/bold green]"
    )

    # Show sample response from best checkpoint
    if best_iter is not None:
        console.print(f"\n[bold]Sample response from iter {best_iter}:[/bold]")
        console.print(f"[dim]Prompt:[/dim] {TEST_PROMPTS[0]}")
        console.print(f"[cyan]Marcus:[/cyan] {all_results[best_iter]['responses'][0]}")

    console.print(
        f"\nTo use the best checkpoint, run:\n"
        f"  cp {adapter_dir}/{best_iter:07d}_adapters.safetensors "
        f"{adapter_dir}/adapters.safetensors"
    )


if __name__ == "__main__":
    main()
