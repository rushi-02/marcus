"""Real-time audio I/O: microphone capture with VAD and interruptible playback.

DESIGN: half-duplex (walkie-talkie) by default.

Without proper acoustic echo cancellation (AEC), Marcus's TTS audio bleeds
through the speakers back into the mic. This causes two failures:
- Whisper transcribes the bleed-back as user speech and hallucinates
  ("freaking freaking freaking..." on near-silence is a known Whisper mode).
- Self-interrupt: the mic detects Marcus's own voice as "user speaking"
  and triggers barge-in, cutting Marcus off mid-sentence.

The robust fix is half-duplex: the capture stream is closed during playback.
The user cannot interrupt Marcus mid-sentence — they wait for him to finish.
This is the same pattern as walkie-talkies, voicemail prompts, and most
phone IVRs. We trade barge-in for stability.

If you want barge-in back, set `half_duplex=False` in AudioConfig and
ensure your environment has AEC (e.g., headphones, an external AEC
DSP, or a quiet room with directional mic).

Architecture (half-duplex):
    LISTENING → user speaks → silence detected → utterance enqueued →
    PROCESSING (mic still on but ignored) → SPEAKING (mic CLOSED) →
    LISTENING (mic reopened, fresh).
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
    """Capture microphone audio with energy-based VAD.

    Each utterance: speech onset detected → buffer audio → silence_duration
    of silence → push complete utterance to async queue.

    Supports `pause()`/`resume()` for half-duplex mode: the agent calls
    `pause()` before TTS playback and `resume()` after, ensuring the
    speaker bleed-back is never picked up.
    """

    def __init__(self, config: AudioConfig, player: "AudioPlayer | None" = None) -> None:
        self.config = config
        self.player = player
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._buffer: deque[np.ndarray] = deque()
        self._is_speaking = False
        self._silence_chunk_count = 0
        self._voiced_chunk_count = 0
        self._chunk_samples = int(config.chunk_duration * config.sample_rate)
        self._max_silence_chunks = int(
            config.silence_duration / config.chunk_duration
        )
        # Require this many continuously-voiced chunks before we accept
        # a real utterance. Filters out single-chunk noise spikes.
        self._min_voiced_chunks_to_start = max(
            1, int(0.3 / config.chunk_duration)  # 300ms of speech
        )

        self._paused = False  # set True during TTS playback

    # ------------------------------------------------------------------
    # Public control (called by the agent loop)
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Stop accepting input (call before TTS playback in half-duplex mode)."""
        self._paused = True
        # Drop any in-progress utterance and queued utterances —
        # they could be Marcus's own voice bleeding into the mic.
        self._buffer.clear()
        self._is_speaking = False
        self._silence_chunk_count = 0
        self._voiced_chunk_count = 0
        # Clear the queue too — anything enqueued during processing
        # is suspect (likely echo of TTS that was just generated).
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def resume(self) -> None:
        """Re-enable input after TTS playback finishes."""
        self._buffer.clear()
        self._is_speaking = False
        self._silence_chunk_count = 0
        self._voiced_chunk_count = 0
        self._paused = False

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
        """sounddevice InputStream callback (runs in audio thread, not async)."""
        if self._paused:
            return  # half-duplex: drop everything during playback

        chunk = indata[:, 0].copy().astype(np.float32)
        rms = float(np.sqrt(np.mean(chunk**2)))
        is_voiced = rms > self.config.silence_threshold

        if is_voiced:
            self._voiced_chunk_count += 1
            if not self._is_speaking:
                # Wait for sustained voiced activity before declaring speech
                if self._voiced_chunk_count >= self._min_voiced_chunks_to_start:
                    self._is_speaking = True
                    self._silence_chunk_count = 0
                    self._buffer.append(chunk)
                else:
                    # Still in the noise-filter window; buffer in case
                    # this turns into real speech.
                    self._buffer.append(chunk)
            else:
                self._silence_chunk_count = 0
                self._buffer.append(chunk)

        elif self._is_speaking:
            # Trailing silence — keep buffering until threshold
            self._buffer.append(chunk)
            self._silence_chunk_count += 1

            if self._silence_chunk_count >= self._max_silence_chunks:
                # End of utterance — enqueue for transcription
                utterance = np.concatenate(list(self._buffer))
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._queue.put(utterance), self._loop
                    )
                self._buffer.clear()
                self._is_speaking = False
                self._silence_chunk_count = 0
                self._voiced_chunk_count = 0
        else:
            # Idle silence — clear noise filter buffer if it had any chunks
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
    """Play audio arrays to speaker.

    In half-duplex mode (default), playback runs to completion — there
    is no barge-in. The agent pauses the mic before calling play() and
    resumes after.

    The legacy `interrupt()` method remains available for full-duplex
    mode but is unused in the default agent flow.
    """

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

    def interrupt(self) -> None:
        """Signal interrupt (only honored in full-duplex mode)."""
        self._interrupted = True

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        blocking: bool = True,
    ) -> bool:
        """Play audio array. Returns True if completed, False if interrupted."""
        if len(audio) == 0:
            return True

        sr = sample_rate or self._sample_rate
        chunk_samples = int((self.config.playback_chunk_ms / 1000) * sr)

        self._interrupted = False
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
