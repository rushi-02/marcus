"""Supervised Fine-Tuning (SFT) with LoRA for the Marcus LLM.

Two execution paths:
1. Local (mlx-lm LoRA) — runs on M2 Pro, no GPU required.
   Slower than cloud but zero-cost and no data upload.

2. Cloud (Unsloth + TRL) — runs on Colab free T4 or PARAM Shakti A100.
   Use notebooks/02_sft_training.ipynb for the cloud path.
   ~30-60 min for 1,500 examples on Colab T4.

This module implements the LOCAL path via mlx-lm's built-in LoRA trainer.
The cloud path is in notebooks/02_sft_training.ipynb.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.console import Console

from marcus.config import MarcusConfig

console = Console()


def train_sft_local(
    config: MarcusConfig,
    data_dir: str | Path | None = None,
    adapter_path: str | Path | None = None,
) -> Path:
    """Run LoRA SFT locally using mlx-lm's built-in trainer.

    Calls `mlx_lm.lora` as a subprocess (its standard invocation method).
    Requires mlx-lm >= 0.22 with LoRA support.

    Args:
        config: MarcusConfig with training hyperparameters.
        data_dir: Directory containing train.jsonl and valid.jsonl.
                  Defaults to config.data.data_dir / "training".
        adapter_path: Where to save LoRA adapter weights.
                      Defaults to adapters/marcus-sft-v1.

    Returns:
        Path to the saved adapter directory.
    """
    data_dir = Path(data_dir or config.data.data_dir / "training")
    adapter_path = Path(adapter_path or "adapters/marcus-sft-v1")
    adapter_path.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_file}. "
            "Run: marcus data prepare"
        )

    t = config.training
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", config.llm.model_id,
        "--data", str(data_dir),
        "--train",
        "--batch-size", str(t.batch_size),
        "--lora-rank", str(t.lora_rank),
        "--lora-alpha", str(t.lora_alpha),
        "--lora-layers", str(t.lora_layers),
        "--epochs", str(t.sft_epochs),
        "--learning-rate", str(t.sft_lr),
        "--max-seq-length", str(t.max_seq_length),
        "--adapter-path", str(adapter_path),
        "--val-batches", "25",
    ]

    console.print(f"[cyan]Starting local SFT training...[/cyan]")
    console.print(f"  Model: {config.llm.model_id}")
    console.print(f"  Data:  {data_dir}")
    console.print(f"  Adapter will be saved to: {adapter_path}")
    console.print(f"  Epochs: {t.sft_epochs}, LR: {t.sft_lr}, Rank: {t.lora_rank}")
    console.print("\n[dim]This may take 20-60 min on M2 Pro...[/dim]\n")

    result = subprocess.run(cmd, check=True)

    if result.returncode == 0:
        console.print(f"\n[bold green]SFT training complete![/bold green]")
        console.print(f"Adapter saved to: {adapter_path}")
        console.print(
            "\nTo use the adapter, update configs/default.yaml:\n"
            f"  llm:\n    adapter_path: \"{adapter_path}\""
        )
    else:
        console.print(f"[red]Training failed with code {result.returncode}[/red]")

    return adapter_path


def evaluate_adapter(
    config: MarcusConfig,
    adapter_path: str | Path,
    test_prompts: list[str] | None = None,
) -> dict:
    """Evaluate a LoRA adapter against the base model.

    Generates responses to test prompts and computes reward scores,
    comparing adapter vs base model quality.

    Returns:
        Dict with scores for adapter and base model.
    """
    from marcus.models.llm import MarcusLLM
    from marcus.pipeline.conversation import load_system_prompt
    from marcus.rewards.composite import composite_reward

    if test_prompts is None:
        test_prompts = [
            "I feel overwhelmed at work and don't know how to cope.",
            "Someone I trusted deeply betrayed me. I'm angry and hurt.",
            "I'm afraid of dying before I achieve anything meaningful.",
            "My boss criticized me in front of everyone. I feel humiliated.",
            "I keep procrastinating on things that matter to me. How do I stop?",
        ]

    system_prompt = load_system_prompt(config.llm.system_prompt_path)
    results: dict = {"base": [], "adapter": []}

    # Evaluate base model
    console.print("[cyan]Evaluating base model...[/cyan]")
    base_config = config.llm.model_copy(update={"adapter_path": None})
    base_llm = MarcusLLM(base_config)
    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = base_llm.generate(messages)
        score = composite_reward(response)
        results["base"].append({"prompt": prompt, "response": response, "score": score})

    # Evaluate with adapter
    console.print(f"[cyan]Evaluating with adapter: {adapter_path}[/cyan]")
    adapter_config = config.llm.model_copy(update={"adapter_path": str(adapter_path)})
    adapter_llm = MarcusLLM(adapter_config)
    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = adapter_llm.generate(messages)
        score = composite_reward(response)
        results["adapter"].append({"prompt": prompt, "response": response, "score": score})

    base_avg = sum(r["score"] for r in results["base"]) / len(results["base"])
    adapter_avg = sum(r["score"] for r in results["adapter"]) / len(results["adapter"])

    console.print(f"\n[bold]Evaluation Results[/bold]")
    console.print(f"  Base model avg score:    {base_avg:.3f}")
    console.print(f"  Adapter avg score:       {adapter_avg:.3f}")
    console.print(f"  Improvement:             {(adapter_avg - base_avg):+.3f}")

    return results
