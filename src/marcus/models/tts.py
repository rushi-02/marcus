"""Text-to-Speech wrapper using F5-TTS-MLX with voice cloning.

F5-TTS supports zero-shot voice cloning from a short reference audio clip.
We use this to give Marcus a consistent deep baritone voice.

Primary: lucasnewman/f5-tts-mlx  (Apache-friendly, MLX-native, voice cloning)
Fallback: mlx-audio Qwen3-TTS    (faster, <100ms, but cloning quality varies)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from rich.console import Console

from marcus.config import TTSConfig

console = Console()

# Sentence boundary characters — used for streaming segmentation
SENTENCE_ENDINGS = {".", "!", "?", ":", ";"}


class MarcusTTS:
    """F5-TTS-MLX wrapper with voice cloning.

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
        self._voice_ref: str | None = None

    def _load(self) -> None:
        if self._loaded:
            return

        voice_path = Path(self.config.voice_ref_path)
        if not voice_path.exists():
            console.print(
                f"[yellow]Warning:[/yellow] Voice reference not found at "
                f"'{voice_path}'. Run 'marcus record-ref' to create one, "
                "or place a WAV file there. Using default voice."
            )
            self._voice_ref = None
        else:
            self._voice_ref = str(voice_path)

        try:
            from f5_tts_mlx.generate import generate as f5_generate  # noqa: F401
            self._backend = "f5_tts_mlx"
            console.print(f"[cyan]Loading TTS:[/cyan] f5-tts-mlx")
        except ImportError:
            try:
                from mlx_audio.tts.utils import load as load_tts  # noqa: F401
                self._backend = "mlx_audio"
                console.print(f"[cyan]Loading TTS:[/cyan] mlx-audio (fallback)")
            except ImportError:
                raise ImportError(
                    "No TTS backend found. Install one:\n"
                    "  uv pip install f5-tts-mlx   # recommended\n"
                    "  uv pip install mlx-audio     # fallback"
                )

        self._loaded = True
        console.print("[green]TTS ready.[/green]")

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to a float32 audio array.

        Args:
            text: Text to synthesize.

        Returns:
            Float32 numpy array of audio samples.
        """
        self._load()

        if not text.strip():
            return np.array([], dtype=np.float32)

        if self._backend == "f5_tts_mlx":
            return self._synthesize_f5(text)
        else:
            return self._synthesize_mlx_audio(text)

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio chunks by splitting text at sentence boundaries.

        This enables the pipeline to start playing audio before the LLM
        has finished generating — each sentence is synthesized and played
        as it arrives, rather than waiting for the full response.

        Yields:
            Float32 numpy arrays, one per sentence.
        """
        sentences = _split_sentences(text)
        for sentence in sentences:
            if sentence.strip():
                audio = self.synthesize(sentence)
                if len(audio) > 0:
                    yield audio

    def _synthesize_f5(self, text: str) -> np.ndarray:
        """Synthesize using F5-TTS-MLX."""
        from f5_tts_mlx.generate import generate

        result = generate(
            text=text,
            ref_audio_path=self._voice_ref,
            # ref_text is the transcript of the reference audio (optional)
            ref_text=None,
        )
        # F5-TTS returns (audio, sample_rate) or just audio depending on version
        if isinstance(result, tuple):
            audio, _ = result
        else:
            audio = result

        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        return np.asarray(audio, dtype=np.float32)

    def _synthesize_mlx_audio(self, text: str) -> np.ndarray:
        """Fallback: synthesize using mlx-audio TTS."""
        from mlx_audio.tts.utils import load as load_tts

        if not hasattr(self, "_mlx_tts_model"):
            self._mlx_tts_model = load_tts(self.config.model_id)

        result = self._mlx_tts_model.generate(
            text=text,
            voice=self._voice_ref,
        )
        audio = result.audio if hasattr(result, "audio") else result
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        return np.asarray(audio, dtype=np.float32)

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
