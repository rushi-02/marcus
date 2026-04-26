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
            from mlx_audio.stt.utils import load_model
        except ImportError:
            raise ImportError(
                "mlx-audio not installed. Run: uv pip install mlx-audio"
            )

        console.print(f"[cyan]Loading ASR model:[/cyan] {self.config.model_id}")
        self._model = load_model(self.config.model_id)
        self._loaded = True
        console.print("[green]ASR model loaded.[/green]")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a complete audio segment to text.

        Args:
            audio: Float32 numpy array, shape (N,), values in [-1, 1].
            sample_rate: Sample rate of the audio (must be 16kHz; resample upstream).

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

        # Resample if needed
        if sample_rate != 16000:
            from mlx_audio.stt.utils import resample_audio
            audio = resample_audio(audio, sample_rate, 16000).astype(np.float32)

        # whisper.generate returns STTOutput with .text and .segments
        result = self._model.generate(
            audio,
            language=self.config.language,
            verbose=False,
        )

        if isinstance(result, str):
            return result.strip()
        if hasattr(result, "text"):
            return result.text.strip()
        return str(result).strip()

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file. Useful for testing without a microphone."""
        self._load()
        result = self._model.generate(
            audio_path, language=self.config.language, verbose=False,
        )
        if isinstance(result, str):
            return result.strip()
        return result.text.strip() if hasattr(result, "text") else str(result).strip()

    def unload(self) -> None:
        """Free model memory (useful when running all 3 models concurrently)."""
        self._model = None
        self._loaded = False
        import gc
        gc.collect()
