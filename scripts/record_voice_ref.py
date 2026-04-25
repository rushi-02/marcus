#!/usr/bin/env python3
"""Standalone script: record a reference voice sample for TTS cloning.

Usage:
    uv run python scripts/record_voice_ref.py
    uv run python scripts/record_voice_ref.py --duration 30 --output data/reference_voice/my_voice.wav
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

console = Console()

REFERENCE_TEXT = """
"You have power over your mind, not outside events. Realize this, and you will find strength.
The impediment to action advances action. What stands in the way becomes the way.
Waste no more time arguing about what a good man should be. Be one."
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Record TTS voice reference sample")
    parser.add_argument("--output", default="data/reference_voice/marcus_voice.wav")
    parser.add_argument("--duration", type=int, default=20, help="Recording duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=24000)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        console.print("[red]Install sounddevice and soundfile: uv pip install sounddevice soundfile[/red]")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Marcus Voice Reference Recorder[/bold cyan]")
    console.print("\nRead the following passage aloud in your deepest, most deliberate voice:")
    console.print(f"\n[italic dim]{REFERENCE_TEXT}[/italic dim]")
    console.print(f"\n[yellow]Recording will start in 3 seconds. Duration: {args.duration}s[/yellow]")
    console.print("Press Ctrl+C to stop early.\n")

    import time
    for i in range(3, 0, -1):
        console.print(f"  {i}...")
        time.sleep(1)

    console.print("[bold green]Recording...[/bold green]")

    try:
        recording = sd.rec(
            int(args.duration * args.sample_rate),
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording stopped early.[/yellow]")

    sf.write(str(output_path), recording, args.sample_rate, subtype="PCM_16")

    # Check quality
    rms = float(np.sqrt(np.mean(recording**2)))
    console.print(f"\n[green]✓ Saved:[/green] {output_path}")
    console.print(f"  Duration: {len(recording) / args.sample_rate:.1f}s")
    console.print(f"  Sample rate: {args.sample_rate} Hz")
    console.print(f"  RMS level: {rms:.4f}", end="")

    if rms < 0.01:
        console.print(" [red]← Very quiet! Check your microphone.[/red]")
    elif rms > 0.5:
        console.print(" [yellow]← Loud! May be clipping. Try speaking softer.[/yellow]")
    else:
        console.print(" [green]← Good level.[/green]")

    console.print(f"\nUpdate configs/default.yaml → tts.voice_ref_path: \"{output_path}\"")


if __name__ == "__main__":
    main()
