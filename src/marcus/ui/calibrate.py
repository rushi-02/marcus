"""Calibrate audio thresholds for your environment.

Measures three RMS levels:
  1. Ambient room noise (3 seconds of you not speaking)
  2. Your speech (5 seconds of you talking normally)
  3. TTS bleed-back via speakers (Marcus speaks, mic captures the echo)

Then recommends silence_threshold, barge_in_threshold_multiplier,
and barge_in_min_duration tuned for your specific environment.

Usage:
    marcus calibrate
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.table import Table

from marcus.config import load_config

console = Console()


REFERENCE_SCRIPT = (
    "Please read aloud at your normal speaking volume:\n\n"
    "  Stoicism teaches that we cannot control external events,\n"
    "  but we can control our judgments about them.\n"
)


def _record_rms(duration_s: float, sample_rate: int = 16000) -> tuple[float, float, list[float]]:
    """Record `duration_s` seconds; return (mean_rms, peak_rms, per_chunk_rms)."""
    chunk_samples = int(0.1 * sample_rate)  # 100ms chunks for granular reading
    chunks_to_record = int(duration_s / 0.1)

    rms_values: deque[float] = deque(maxlen=chunks_to_record)

    def callback(indata, frames, t, status):
        chunk = indata[:, 0]
        rms = float(np.sqrt(np.mean(chunk**2)))
        rms_values.append(rms)
        # Live readout
        bar = "█" * min(40, int(rms * 800))
        console.print(f"  rms={rms:.4f}  {bar}", end="\r")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        blocksize=chunk_samples,
        dtype="float32",
        callback=callback,
    ):
        time.sleep(duration_s + 0.1)

    if not rms_values:
        return 0.0, 0.0, []
    vals = list(rms_values)
    return float(np.mean(vals)), float(np.max(vals)), vals


def _measure_tts_bleed(config) -> float:
    """Play a TTS sample and measure how loudly it leaks back into the mic."""
    from marcus.models.tts import MarcusTTS

    console.print("\n[bold cyan]== Measuring TTS bleed-back ==[/bold cyan]")
    console.print(
        "Marcus will speak. [yellow]Stay silent[/yellow] — we're measuring "
        "how much of his voice the mic picks up via your speakers."
    )
    if input("Press ENTER when ready (or 'q' to skip): ").strip().lower() == "q":
        return 0.0

    tts = MarcusTTS(config.tts)
    tts._load()
    sample_text = (
        "Remember, you have power over your mind, not outside events. "
        "Realize this and you will find strength."
    )

    # Synthesize audio first, then play and record concurrently
    audio_out = tts.synthesize(sample_text)
    play_duration = len(audio_out) / config.tts.sample_rate

    rms_during: deque[float] = deque()

    def rec_callback(indata, frames, t, status):
        chunk = indata[:, 0]
        rms = float(np.sqrt(np.mean(chunk**2)))
        rms_during.append(rms)

    chunk_samples = int(0.1 * config.audio.sample_rate)

    # Start recording, then start playback (overlapping)
    with sd.InputStream(
        samplerate=config.audio.sample_rate,
        channels=1,
        blocksize=chunk_samples,
        dtype="float32",
        callback=rec_callback,
    ):
        sd.play(audio_out, samplerate=config.tts.sample_rate)
        sd.wait()
        time.sleep(0.3)  # tail capture

    if not rms_during:
        return 0.0
    # Take the peak — bleed varies; we want the worst case for our threshold
    return float(np.percentile(list(rms_during), 95))


def run_calibration() -> None:
    """Interactive calibration."""
    config = load_config()

    console.print(
        "\n[bold cyan]Marcus Audio Calibration[/bold cyan]\n"
        "We'll measure your environment in three steps.\n"
    )

    # 1. Ambient noise
    console.print("[bold]== Step 1/3: Ambient noise ==[/bold]")
    console.print("Sit silently for 3 seconds. Don't speak.")
    if input("Press ENTER when ready: ").strip().lower() == "q":
        return

    ambient_mean, ambient_peak, _ = _record_rms(duration_s=3.0)
    console.print()
    console.print(
        f"  Ambient: mean={ambient_mean:.4f}, peak={ambient_peak:.4f}"
    )

    # 2. Speech
    console.print("\n[bold]== Step 2/3: Your speech ==[/bold]")
    console.print(REFERENCE_SCRIPT)
    if input("Press ENTER when ready, then start reading: ").strip().lower() == "q":
        return

    speech_mean, speech_peak, _ = _record_rms(duration_s=6.0)
    console.print()
    console.print(
        f"  Speech: mean={speech_mean:.4f}, peak={speech_peak:.4f}"
    )

    # 3. TTS bleed
    bleed_peak = _measure_tts_bleed(config)
    console.print()
    if bleed_peak > 0:
        console.print(f"  TTS bleed: {bleed_peak:.4f} (95th percentile)")

    # ---------------------------------------------------------------
    # Recommendations
    # ---------------------------------------------------------------
    console.print("\n[bold cyan]== Recommendations ==[/bold cyan]")

    # silence_threshold: above ambient noise, well below speech
    # Pick midpoint, with a floor at 0.005 and ceiling at speech_mean / 2
    rec_silence_thr = max(0.005, min(speech_mean / 2, ambient_peak * 1.5))

    # barge_in threshold needs to clear bleed but stay below speech
    # We want: silence_threshold * multiplier > bleed_peak,
    # and:     silence_threshold * multiplier < speech_mean
    if bleed_peak > 0:
        # multiplier must put threshold above bleed
        rec_multiplier = max(1.2, bleed_peak / rec_silence_thr * 1.3)
        # but cap at speech_mean / silence_thr to ensure speech still triggers
        speech_cap = speech_mean / rec_silence_thr * 0.7
        if speech_cap > 1.2:
            rec_multiplier = min(rec_multiplier, speech_cap)
    else:
        rec_multiplier = 1.5  # no bleed measured (headphones?)

    # min_duration: shorter for headphones, longer for speakers
    # Heuristic: if bleed is significant, require longer sustained voice
    bleed_ratio = bleed_peak / speech_mean if speech_mean > 0 else 0
    if bleed_ratio < 0.1:
        rec_min_duration = 0.2  # negligible bleed → snappy
    elif bleed_ratio < 0.3:
        rec_min_duration = 0.3
    else:
        rec_min_duration = 0.5  # significant bleed → conservative

    table = Table(title="Recommended audio config")
    table.add_column("Setting", style="cyan")
    table.add_column("Current", style="dim")
    table.add_column("Recommended", style="green")
    table.add_row("silence_threshold", f"{config.audio.silence_threshold:.3f}", f"{rec_silence_thr:.3f}")
    table.add_row("barge_in_threshold_multiplier", f"{config.audio.barge_in_threshold_multiplier:.1f}", f"{rec_multiplier:.1f}")
    table.add_row("barge_in_min_duration", f"{config.audio.barge_in_min_duration:.2f}", f"{rec_min_duration:.2f}")
    console.print(table)

    can_barge_in = (rec_silence_thr * rec_multiplier) < speech_mean
    if can_barge_in:
        console.print(
            f"\n[green]✓ Barge-in should work[/green] — speech (mean {speech_mean:.4f}) "
            f"clears the playback threshold ({rec_silence_thr * rec_multiplier:.4f})."
        )
    else:
        console.print(
            f"\n[yellow]⚠ Barge-in may be hard[/yellow] — your speech ({speech_mean:.4f}) "
            f"is close to bleed level ({bleed_peak:.4f}). Consider headphones."
        )

    console.print(
        "\nUpdate [cyan]configs/default.yaml[/cyan] under the [cyan]audio:[/cyan] section "
        "with the recommended values above."
    )
