"""Real-time audio I/O: microphone capture with VAD and interruptible playback.

Two modes, controlled by `AudioConfig.barge_in`:

1. Half-duplex (`barge_in=False`)
   Mic is paused during TTS playback. Marcus finishes his thought before
   listening again. Rock-solid but no interruption support. Used as the
   fallback when echo conditions are too noisy for safe barge-in.

2. Echo-resistant barge-in (`barge_in=True`, default)
   Mic stays open during playback. To avoid Marcus interrupting himself
   via speaker bleed-back, two gates kick in during playback:
     a) Energy threshold is multiplied by `barge_in_threshold_multiplier`
        (default 3x) — direct user speech is typically ≥6dB above echo.
     b) Sustained-voice gate: need `barge_in_min_duration` of continuous
        loud speech (default 600ms) before declaring barge-in.
   Brief echo blips don't pass either gate; real user speech does.
   With headphones the echo path is gone and you can drop both gates.

State machine (barge-in mode):
    LISTENING → user speaks → silence → utterance enqueued
    PROCESSING → ASR → LLM streams tokens → TTS → SPEAKING
        ↓ (sustained loud user voice during playback)
    BARGE-IN → player.interrupt() → drop captured echo → wait for user
        → silence → fresh utterance enqueued → PROCESSING ...
"""

from __future__ import annotations

import asyncio
from collections import deque
from enum import Enum, auto

import numpy as np
import sounddevice as sd
from rich.console import Console

from marcus.config import AudioConfig

console = Console()


class AgentState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


class AudioCapture:
    """Capture microphone audio with energy-based VAD and echo-resistant barge-in."""

    def __init__(self, config: AudioConfig, player: "AudioPlayer | None" = None) -> None:
        self.config = config
        self.player = player
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._buffer: deque[np.ndarray] = deque()
        self._is_speaking = False
        self._silence_chunk_count = 0
        self._voiced_chunk_count = 0
        self._voiced_during_playback = 0  # for sustained-voice barge-in gate
        self._chunk_samples = int(config.chunk_duration * config.sample_rate)
        self._max_silence_chunks = int(
            config.silence_duration / config.chunk_duration
        )
        self._min_voiced_chunks_to_start = max(
            1, int(0.3 / config.chunk_duration)  # 300ms of speech
        )
        self._barge_in_min_chunks = max(
            1, int(config.barge_in_min_duration / config.chunk_duration)
        )

        self._paused = False
        # Set true for one chunk-callback cycle after barge-in fires, so
        # the callback ignores the few echo-tail samples left in the air.
        self._barge_in_cooldown_chunks = 0

    # ------------------------------------------------------------------
    # Public control (called by the agent loop)
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Stop accepting input — used by half-duplex mode only."""
        self._paused = True
        self._reset_buffers()

    def resume(self) -> None:
        """Re-enable input."""
        self._reset_buffers()
        self._paused = False

    def _reset_buffers(self) -> None:
        self._buffer.clear()
        self._is_speaking = False
        self._silence_chunk_count = 0
        self._voiced_chunk_count = 0
        self._voiced_during_playback = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # Audio thread callback
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Runs in audio thread. Branches on playback state for echo resistance."""
        if self._paused:
            return

        chunk = indata[:, 0].copy().astype(np.float32)
        rms = float(np.sqrt(np.mean(chunk**2)))

        playback_active = bool(self.player and self.player.is_playing)

        # Use a stricter threshold during playback so faint echo doesn't
        # register as voiced. Direct user speech is typically much louder
        # than the echo bleed-through.
        if playback_active and self.config.barge_in:
            threshold = self.config.silence_threshold * self.config.barge_in_threshold_multiplier
        else:
            threshold = self.config.silence_threshold

        is_voiced = rms > threshold

        if self.config.debug_audio:
            tag = "PLAY" if playback_active else "    "
            voiced = "VOICED" if is_voiced else "      "
            console.print(
                f"[dim]{tag} rms={rms:.4f} thr={threshold:.4f} {voiced}[/dim]"
            )

        # ---- Barge-in detection ----
        if playback_active and self.config.barge_in:
            if is_voiced:
                self._voiced_during_playback += 1
                if self._voiced_during_playback >= self._barge_in_min_chunks:
                    # Sustained loud speech → real interrupt
                    self.player.interrupt()
                    self._voiced_during_playback = 0
                    self._barge_in_cooldown_chunks = 2  # ignore brief echo tail
            else:
                self._voiced_during_playback = 0
            # Don't accumulate buffer during playback — wait for the
            # interrupt to fire and drain the playback queue first.
            return

        # ---- Cooldown after barge-in ----
        if self._barge_in_cooldown_chunks > 0:
            self._barge_in_cooldown_chunks -= 1
            return

        # ---- Normal VAD path (no playback active) ----
        if is_voiced:
            self._voiced_chunk_count += 1
            if not self._is_speaking:
                if self._voiced_chunk_count >= self._min_voiced_chunks_to_start:
                    self._is_speaking = True
                    self._silence_chunk_count = 0
                self._buffer.append(chunk)
            else:
                self._silence_chunk_count = 0
                self._buffer.append(chunk)

        elif self._is_speaking:
            self._buffer.append(chunk)
            self._silence_chunk_count += 1

            if self._silence_chunk_count >= self._max_silence_chunks:
                utterance = np.concatenate(list(self._buffer))
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._queue.put(utterance), self._loop
                    )
                self._reset_buffers()
        else:
            # Idle silence
            self._voiced_chunk_count = 0
            self._buffer.clear()

    # ------------------------------------------------------------------
    # Async generator interface
    # ------------------------------------------------------------------

    async def listen(self) -> "AsyncIterator[np.ndarray]":
        """Async generator that yields complete utterances (numpy float32 arrays)."""
        self._loop = asyncio.get_running_loop()

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self._chunk_samples,
            dtype="float32",
            device=self.config.device_input,
            callback=self._audio_callback,
        ):
            console.print("[dim]Microphone active. Speak to Marcus...[/dim]")
            while True:
                utterance = await self._queue.get()
                yield utterance


class AudioPlayer:
    """Play audio arrays to speaker with chunk-level interrupt support."""

    def __init__(self, config: AudioConfig, sample_rate: int | None = None) -> None:
        self.config = config
        self._sample_rate = sample_rate or config.sample_rate
        self._interrupted = False
        self._is_playing = False
        self._chunk_samples = int(
            (config.playback_chunk_ms / 1000) * self._sample_rate
        )

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def was_interrupted(self) -> bool:
        return self._interrupted

    def interrupt(self) -> None:
        """Signal interrupt — playback stops on the next chunk boundary."""
        self._interrupted = True

    def reset_interrupt(self) -> None:
        """Clear the interrupt flag before starting a new playback session."""
        self._interrupted = False

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        blocking: bool = True,
    ) -> bool:
        """Play audio array. Returns True if completed, False if interrupted."""
        if len(audio) == 0:
            return True
        if self._interrupted:
            # Already in interrupted state — drop further audio until reset
            return False

        sr = sample_rate or self._sample_rate
        chunk_samples = int((self.config.playback_chunk_ms / 1000) * sr)

        self._is_playing = True
        try:
            with sd.OutputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                device=self.config.device_output,
            ) as stream:
                offset = 0
                while offset < len(audio):
                    if self._interrupted:
                        return False
                    end = min(offset + chunk_samples, len(audio))
                    stream.write(audio[offset:end])
                    offset = end
        finally:
            self._is_playing = False

        return not self._interrupted

    def play_interruptible(self, audio: np.ndarray, sample_rate: int | None = None) -> bool:
        """Alias kept for backwards compatibility."""
        return self.play(audio, sample_rate=sample_rate)
