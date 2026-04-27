"""Marcus CLI — terminal interface for the Stoic voice agent.

Commands:
    marcus chat         Voice-to-voice conversation (streaming, with barge-in)
    marcus text         Text-only conversation (no microphone/TTS required)
    marcus data         Data pipeline subcommands
    marcus train        Training subcommands
    marcus record-ref   Record a reference voice sample for TTS cloning

Usage:
    uv run marcus chat
    uv run marcus text
    uv run marcus data download
    uv run marcus data prepare
    uv run marcus train sft
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(
    name="marcus",
    help="Marcus — Real-Time Stoic Voice Agent",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Data pipeline commands")
train_app = typer.Typer(help="Model training commands")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")

console = Console()

MARCUS_BANNER = """
[bold cyan]Marcus Aurelius — Stoic Voice Agent[/bold cyan]
[dim]"You have power over your mind, not outside events."[/dim]
"""


# ---------------------------------------------------------------------------
# Chat commands
# ---------------------------------------------------------------------------

@app.command()
def chat(
    config_path: str = typer.Option("configs/default.yaml", "--config", "-c"),
    streaming: bool = typer.Option(True, "--streaming/--no-streaming", help="Enable sentence-level TTS streaming"),
) -> None:
    """Start a real-time voice conversation with Marcus Aurelius.

    Speak into your microphone. Marcus will respond in a deep baritone voice.
    Interrupt him at any time by speaking — he'll stop and listen.
    """
    from marcus.config import load_config
    from marcus.pipeline.agent import MarcusAgent

    console.print(Panel(MARCUS_BANNER, border_style="cyan"))

    config = load_config(config_path)
    agent = MarcusAgent(config)

    try:
        if streaming:
            asyncio.run(agent.run_streaming())
        else:
            asyncio.run(agent.run())
    except KeyboardInterrupt:
        console.print("\n[dim]Session ended. Farewell.[/dim]")


@app.command()
def text(
    config_path: str = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Text-only conversation with Marcus (no microphone or TTS needed).

    Useful for testing the LLM persona and fine-tuned adapter quality
    without requiring audio hardware.
    """
    from marcus.config import load_config
    from marcus.pipeline.agent import MarcusAgent

    console.print(Panel(MARCUS_BANNER, border_style="cyan"))

    config = load_config(config_path)
    agent = MarcusAgent(config)

    try:
        asyncio.run(agent.text_chat())
    except KeyboardInterrupt:
        console.print("\n[dim]Session ended. Farewell.[/dim]")


@app.command(name="calibrate")
def calibrate_audio() -> None:
    """Calibrate audio thresholds for your environment.

    Measures ambient noise, your voice, and TTS bleed-back via speakers.
    Recommends silence_threshold and barge-in parameters tuned for your
    setup. Update configs/default.yaml with the suggested values.
    """
    from marcus.ui.calibrate import run_calibration
    run_calibration()


@app.command(name="record-ref")
def record_ref(
    output_path: str = typer.Option("data/reference_voice/marcus_voice.wav", "--output", "-o"),
    duration: int = typer.Option(20, "--duration", "-d", help="Recording duration in seconds"),
) -> None:
    """Record a reference voice sample for TTS cloning.

    Speak in a low, deliberate baritone voice for the specified duration.
    The recording will be used as the reference for voice cloning.

    Example: Read a passage from Meditations aloud in your deepest voice.
    """
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    console.print(f"[cyan]Recording {duration}s of reference audio...[/cyan]")
    console.print("[dim]Speak clearly in a deep, deliberate voice. Press Ctrl+C to stop early.[/dim]\n")

    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording stopped early.[/yellow]")

    sf.write(str(output_path_obj), recording, sample_rate)
    console.print(f"\n[green]Saved reference audio to:[/green] {output_path}")
    console.print("Update configs/default.yaml → tts.voice_ref_path if needed.")


# ---------------------------------------------------------------------------
# Data commands
# ---------------------------------------------------------------------------

@data_app.command(name="download")
def data_download(
    config_path: str = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Download Stoic source texts (Meditations, Discourses, Letters, Enchiridion)."""
    from marcus.config import load_config
    from marcus.data.download import download_all_sources

    config = load_config(config_path)
    console.print("[cyan]Downloading Stoic source texts...[/cyan]")
    download_all_sources(config.data.data_dir)


@data_app.command(name="clean")
def data_clean(
    config_path: str = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Clean and chunk raw source texts into passage-level segments."""
    from marcus.config import load_config
    from marcus.data.clean import clean_all_sources

    config = load_config(config_path)
    console.print("[cyan]Cleaning and chunking source texts...[/cyan]")
    clean_all_sources(config.data.data_dir)


@data_app.command(name="synthesize")
def data_synthesize(
    config_path: str = typer.Option("configs/default.yaml", "--config", "-c"),
    api_key: str = typer.Option(None, "--api-key", envvar="ANTHROPIC_API_KEY"),
    local: bool = typer.Option(False, "--local", help="Use local MLX model instead of API"),
) -> None:
    """Generate synthetic instruction-response pairs using Claude API or local LLM."""
    import json
    from marcus.config import load_config
    from marcus.data.synthesize import (
        generate_pairs_anthropic,
        generate_pairs_local,
        save_synthetic_pairs,
    )

    config = load_config(config_path)
    passages_path = config.data.data_dir / "processed" / "stoic_passages.jsonl"

    if not passages_path.exists():
        console.print("[red]Run 'marcus data clean' first.[/red]")
        raise typer.Exit(1)

    passages = []
    with open(passages_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            passages.append(entry["passage"])

    console.print(f"[cyan]Loaded {len(passages)} passages.[/cyan]")

    if local:
        pairs = generate_pairs_local(passages, config.training.batch_size)
    else:
        if not api_key:
            console.print("[red]Set ANTHROPIC_API_KEY or use --local flag.[/red]")
            raise typer.Exit(1)
        pairs = asyncio.run(
            generate_pairs_anthropic(
                passages=passages[:config.data.target_pairs],
                api_key=api_key,
                model=config.data.synthesis_model,
                pairs_per_passage=config.data.pairs_per_passage,
            )
        )

    save_synthetic_pairs(pairs, config.data.data_dir / "synthetic")


@data_app.command(name="format")
def data_format(
    config_path: str = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Format synthetic pairs into mlx-lm training JSONL (train.jsonl, valid.jsonl)."""
    from marcus.config import load_config
    from marcus.data.format import format_pipeline

    config = load_config(config_path)
    pairs_path = config.data.data_dir / "synthetic" / "instruction_pairs.jsonl"

    if not pairs_path.exists():
        console.print("[red]Run 'marcus data synthesize' first.[/red]")
        raise typer.Exit(1)

    format_pipeline(
        pairs_path=pairs_path,
        output_dir=config.data.data_dir / "training",
        system_prompt_path=config.llm.system_prompt_path,
        val_ratio=config.data.val_ratio,
    )


@data_app.command(name="prepare")
def data_prepare(
    config_path: str = typer.Option("configs/default.yaml", "--config", "-c"),
    api_key: str = typer.Option(None, "--api-key", envvar="ANTHROPIC_API_KEY"),
    local: bool = typer.Option(False, "--local"),
) -> None:
    """Run the full data pipeline: download → clean → synthesize → format."""
    console.print("[bold cyan]Running full data pipeline...[/bold cyan]\n")
    ctx = typer.Context(data_prepare)
    data_download(config_path)
    data_clean(config_path)
    data_synthesize(config_path, api_key=api_key, local=local)
    data_format(config_path)
    console.print("\n[bold green]Data pipeline complete![/bold green]")


# ---------------------------------------------------------------------------
# Training commands
# ---------------------------------------------------------------------------

@train_app.command(name="sft")
def train_sft(
    config_path: str = typer.Option("configs/default.yaml", "--config", "-c"),
    adapter_path: str = typer.Option("adapters/marcus-sft-v1", "--adapter-path"),
) -> None:
    """Run LoRA SFT fine-tuning locally using mlx-lm.

    Requires training data at data/training/train.jsonl.
    Run 'marcus data prepare' first.

    For faster training, use notebooks/02_sft_training.ipynb on Google Colab.
    """
    from marcus.config import load_config
    from marcus.training.sft import train_sft_local

    config = load_config(config_path)
    train_sft_local(config, adapter_path=adapter_path)


@train_app.command(name="eval")
def train_eval(
    config_path: str = typer.Option("configs/default.yaml", "--config", "-c"),
    adapter_path: str = typer.Option(None, "--adapter-path"),
) -> None:
    """Evaluate adapter quality vs base model on test prompts."""
    from marcus.config import load_config
    from marcus.training.sft import evaluate_adapter

    config = load_config(config_path)
    path = adapter_path or config.llm.adapter_path
    if not path:
        console.print("[red]Specify --adapter-path or set llm.adapter_path in config.[/red]")
        raise typer.Exit(1)
    evaluate_adapter(config, adapter_path=path)


@train_app.command(name="grpo")
def train_grpo(
    config_path: str = typer.Option("configs/default.yaml", "--config", "-c"),
) -> None:
    """Show instructions for GRPO training on cloud (Colab / PARAM Shakti)."""
    from marcus.config import load_config
    from marcus.training.grpo import should_retrain, train_grpo_cloud_instructions

    config = load_config(config_path)
    ready = should_retrain(config)
    if ready:
        train_grpo_cloud_instructions(config)
    else:
        console.print(
            "[yellow]Not enough feedback collected yet.[/yellow] "
            f"Need {config.training.min_feedback_samples} samples. "
            "Keep using Marcus and providing thumbs up/down feedback."
        )
