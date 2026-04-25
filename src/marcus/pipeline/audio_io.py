"""Real-time audio I/O: microphone capture with VAD and interruptible playback.

Architecture:
- AudioCapture: InputStream callback → energy-based VAD → async queue of utterances
- AudioPlayer: OutputStream with chunk-level barge-in checks

Barge-in state machine:
    IDLE → LISTENING → PROCESSING → SPEAKING
                ↑                        |
                └────── (barge-in) ──────┘

When the user starts speaking while Marcus is playing audio:
1. AudioCapture detects speech energy above threshold
2. It calls player.interrupt() to set the cancellation flag
3. AudioPlayer stops writing chunks on the next iteration
4. The agent loop sees the interrupted state and transitions back to LISTENING
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
    """Capture microphone audio with voice activity detection (VAD).

    Buffers audio chunks. When speech ends (silence detected after speech),
    the complete utterance is pushed to an async queue for the agent loop.

    Simultaneously monitors for barge-in: if Marcus is speaking and the
    user starts talking, player.interrupt() is called immediately.
    """

    def __init__(self, config: AudioConfig, player: "AudioPlayer | None" = None) -> None:
        self.config = config
        self.player = player
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._buffer: deque[np.ndarray] = deque()
        self._is_speaking = False
        self._silence_chunk_count = 0
        self._chunk_samples = int(config.chunk_duration * config.sample_rate)
        # How many silent chunks before we consider the utterance done
        self._max_silence_chunks = int(
            config.silence_duration / config.chunk_duration
        )

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice InputStream callback (runs in audio thread, not async)."""
        chunk = indata[:, 0].copy().astype(np.float32)
        rms = float(np.sqrt(np.mean(chunk**2)))

        is_voiced = rms > self.config.silence_threshold

        # Barge-in: user speaks while Marcus is playing
        if is_voiced and self.player and self.player.is_playing:
            self.player.interrupt()

        if is_voiced:
            if not self._is_speaking:
                self._is_speaking = True
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
    """Play audio arrays to speaker with chunk-level barge-in support.

    The key property: audio is written in small chunks (~100ms each).
    Between each chunk, we check the _interrupted flag. If set,
    playback stops immediately — giving sub-150ms barge-in response.
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
        """Signal barge-in — stops playback on the next chunk boundary."""
        self._interrupted = True

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        blocking: bool = True,
    ) -> bool:
        """Play audio array. Returns True if completed, False if interrupted.

        Args:
            audio: Float32 numpy array.
            sample_rate: Override sample rate (uses TTS rate if different from mic).
            blocking: If True, block until audio finishes or is interrupted.
        """
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
        """Alias for play() — emphasizes barge-in capability."""
        return self.play(audio, sample_rate=sample_rate)
