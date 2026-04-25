"""GRPO reinforcement learning training for the Marcus LLM.

GRPO (Group Relative Policy Optimization) from DeepSeek:
- No critic model needed (unlike PPO) → lower memory
- Samples G=4 completions per prompt, scores with composite_reward()
- Group-relative advantage: each completion's advantage = reward - group_mean

Two execution paths:
1. Cloud (TRL GRPOTrainer + Unsloth) — recommended, use notebooks/03_grpo_training.ipynb
2. Local (MLX-GRPO) — experimental, use train_grpo_local() if available

Trigger retraining when >= config.training.min_feedback_samples have been collected.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from marcus.config import MarcusConfig
from marcus.rewards.composite import batch_rewards

console = Console()


def load_feedback_data(feedback_log_path: str | Path = "data/feedback_log.jsonl") -> list[dict]:
    """Load collected user feedback from JSONL log.

    Returns:
        List of feedback entries with keys: user, assistant, reward.
    """
    path = Path(feedback_log_path)
    if not path.exists():
        return []

    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    return entries


def prepare_grpo_prompts(
    feedback_data: list[dict],
    output_path: str | Path = "data/training/grpo_prompts.jsonl",
) -> Path:
    """Convert feedback data into GRPO training prompts.

    GRPO needs prompts (not prompt+response pairs) — it generates its own
    completions and scores them. We use the user messages as prompts.

    Output format per line:
        {"prompt": "user message", "reward": float (from feedback)}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = []
    for entry in feedback_data:
        if "user" in entry:
            prompts.append({
                "prompt": entry["user"],
                "feedback": entry.get("reward", 0.0),
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    console.print(f"[green]Prepared {len(prompts)} GRPO prompts → {output_path}[/green]")
    return output_path


def should_retrain(
    config: MarcusConfig,
    feedback_log_path: str | Path = "data/feedback_log.jsonl",
) -> bool:
    """Check whether enough feedback has been collected to trigger GRPO retraining."""
    data = load_feedback_data(feedback_log_path)
    count = len(data)
    threshold = config.training.min_feedback_samples
    console.print(
        f"Feedback collected: {count}/{threshold} samples "
        f"({'ready' if count >= threshold else 'not yet ready'} for retraining)"
    )
    return count >= threshold


def train_grpo_cloud_instructions(config: MarcusConfig) -> None:
    """Print instructions for running GRPO training on cloud (Colab/PARAM Shakti).

    The cloud path uses TRL GRPOTrainer + Unsloth (CUDA-based).
    See notebooks/03_grpo_training.ipynb for the full notebook.
    """
    console.print(
        "\n[bold cyan]GRPO Training — Cloud Instructions[/bold cyan]\n"
        "\n1. Upload your feedback data:"
        "\n   scp data/feedback_log.jsonl <colab-or-server>:~/\n"
        "\n2. Open notebooks/03_grpo_training.ipynb in Google Colab"
        "\n   (or upload to PARAM Shakti Jupyter environment)\n"
        "\n3. Set the adapter path to your current SFT adapter:"
        f"\n   adapter_path = '{config.llm.adapter_path}'\n"
        "\n4. Run all cells. Training takes ~30-60 min on Colab T4.\n"
        "\n5. Download the new adapter:"
        "\n   adapters/marcus-grpo-v1/\n"
        "\n6. Update configs/default.yaml:"
        "\n   llm:"
        "\n     adapter_path: \"adapters/marcus-grpo-v1\"\n"
    )


def score_responses_locally(responses: list[str], prompts: list[str] | None = None) -> list[float]:
    """Score a batch of responses using the composite reward function.

    Useful for debugging reward quality without running full GRPO.
    """
    return batch_rewards(responses)
