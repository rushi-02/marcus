"""Automatic Speech Recognition wrapper using mlx-audio (Whisper).

Provides utterance-level transcription. Real-time streaming from the mic
is handled by AudioCapture in pipeline/audio_io.py, which buffers audio
until silence, then calls transcribe() on the complete utterance.
"""

from __future__ import annotations

import numpy as np
from rich.console import Console

from marcus.config import ASRConfig

console = Console()


class MarcusASR:
    """Whisper ASR wrapper via mlx-audio.

    Usage:
        asr = MarcusASR(config.asr)
        text = asr.transcribe(audio_array, sample_rate=16000)
    """

    def __init__(self, config: ASRConfig) -> None:
        self.config = config
        self._model = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            from mlx_audio.stt.utils import load as load_stt
            console.print(f"[cyan]Loading ASR model:[/cyan] {self.config.model_id}")
            self._model = load_stt(self.config.model_id)
            self._loaded = True
            console.print("[green]ASR model loaded.[/green]")
        except ImportError:
            raise ImportError(
                "mlx-audio not installed. Run: uv pip install mlx-audio"
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a complete audio segment to text.

        Args:
            audio: Float32 numpy array, shape (N,), values in [-1, 1].
            sample_rate: Sample rate of the audio (default 16kHz).

        Returns:
            Transcribed text string (stripped, may be empty if silence).
        """
        self._load()

        if audio is None or len(audio) == 0:
            return ""

        # Ensure float32 mono
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)

        # mlx-audio Whisper accepts numpy arrays or file paths
        result = self._model.generate(
            audio,
            language=self.config.language,
        )

        # Result may be a string or object with .text attribute
        if isinstance(result, str):
            text = result
        elif hasattr(result, "text"):
            text = result.text
        else:
            text = str(result)

        return text.strip()

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file. Useful for testing without a microphone."""
        self._load()
        result = self._model.generate(audio_path, language=self.config.language)
        if isinstance(result, str):
            return result.strip()
        return result.text.strip()

    def unload(self) -> None:
        """Free model memory (useful when running all 3 models concurrently)."""
        self._model = None
        self._loaded = False
        import gc
        gc.collect()
