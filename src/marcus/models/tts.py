"""Text-to-Speech wrapper using mlx-audio (Kokoro by default).

Kokoro is an 82M-parameter MLX-native TTS model with built-in preset voices.
We use `bm_george` (British male, deep baritone) as our default Marcus voice —
no voice cloning needed, no reference audio file to manage.

For true voice cloning (e.g. record your own voice as Marcus), see
`MarcusTTS_Chatterbox` below — but that adds complexity and ~500MB of weights.

Output: 24kHz mono float32 numpy arrays.
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from rich.console import Console

from marcus.config import TTSConfig

console = Console()

# Sentence boundary characters for streaming segmentation
SENTENCE_ENDINGS = {".", "!", "?", ":", ";"}


class MarcusTTS:
    """Kokoro TTS wrapper via mlx-audio.

    Usage:
        tts = MarcusTTS(config.tts)
        audio = tts.synthesize("Your mind is a fortress.")
        for chunk in tts.synthesize_stream(long_text):
            player.play(chunk)
    """

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._model = None
        self._loaded = False
        self._sample_rate = config.sample_rate

    def _load(self) -> None:
        if self._loaded:
            return

        try:
            from mlx_audio.tts.utils import load_model
        except ImportError:
            raise ImportError(
                "mlx-audio not installed. Run: uv pip install mlx-audio"
            )

        console.print(f"[cyan]Loading TTS:[/cyan] {self.config.model_id}")
        self._model = load_model(self.config.model_id)
        self._loaded = True
        console.print(f"[green]TTS ready:[/green] voice={self.config.voice}, lang={self.config.lang_code}")

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to a float32 audio array.

        Args:
            text: Text to synthesize.

        Returns:
            Float32 numpy array of audio samples (mono, sample rate matches config).
        """
        self._load()

        text = text.strip()
        if not text:
            return np.array([], dtype=np.float32)

        # Kokoro returns a generator of GenerationResult objects with .audio
        results = self._model.generate(
            text=text,
            voice=self.config.voice,
            speed=self.config.speed,
            lang_code=self.config.lang_code,
        )

        audio_chunks = []
        for result in results:
            audio = result.audio
            if hasattr(audio, "numpy"):
                audio = np.asarray(audio)
            elif hasattr(audio, "tolist"):
                audio = np.asarray(audio)
            audio_chunks.append(np.asarray(audio, dtype=np.float32).flatten())

        if not audio_chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(audio_chunks)

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio chunks at sentence boundaries for low-latency playback.

        This enables the pipeline to start playing audio before the LLM has
        finished generating — each sentence is synthesized and played as it
        arrives, rather than waiting for the full response.

        Yields:
            Float32 numpy arrays, one per sentence (or per Kokoro chunk).
        """
        sentences = _split_sentences(text)
        for sentence in sentences:
            if sentence.strip():
                audio = self.synthesize(sentence)
                if len(audio) > 0:
                    yield audio

    def unload(self) -> None:
        """Free model memory."""
        self._model = None
        self._loaded = False
        import gc
        gc.collect()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-level chunks for streaming synthesis."""
    sentences = []
    current = ""

    for char in text:
        current += char
        if char in SENTENCE_ENDINGS:
            stripped = current.strip()
            if stripped:
                sentences.append(stripped)
            current = ""

    if current.strip():
        sentences.append(current.strip())

    return sentences
