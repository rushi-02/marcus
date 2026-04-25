"""Configuration system for Marcus — Pydantic models with YAML overrides.

Usage:
    from marcus.config import load_config
    config = load_config()  # loads configs/default.yaml
    config = load_config("configs/custom.yaml")
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class ASRConfig(BaseSettings):
    """Automatic Speech Recognition (Whisper via mlx-audio)."""

    model_id: str = "mlx-community/whisper-small"
    language: str = "en"
    stream: bool = True


class LLMConfig(BaseSettings):
    """Language Model (Llama 3.2 3B via mlx-lm)."""

    model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    adapter_path: str | None = None  # LoRA adapter after fine-tuning
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt_path: str = "configs/system_prompt.txt"


class TTSConfig(BaseSettings):
    """Text-to-Speech with voice cloning (mlx-audio)."""

    model_id: str = "lucasnewman/f5-tts-mlx"
    voice_ref_path: str = "data/reference_voice/marcus_voice.wav"
    sample_rate: int = 24000
    streaming: bool = True


class AudioConfig(BaseSettings):
    """Audio I/O settings for mic capture and speaker playback."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 0.5  # seconds per audio chunk
    device_input: int | None = None
    device_output: int | None = None
    silence_threshold: float = 0.01  # RMS energy threshold for VAD
    silence_duration: float = 1.5  # seconds of silence to end utterance
    playback_chunk_ms: int = 100  # ms per playback chunk (for barge-in checks)


class TrainingConfig(BaseSettings):
    """Fine-tuning hyperparameters for SFT and GRPO."""

    # SFT (LoRA)
    sft_epochs: int = 3
    sft_lr: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_layers: int = 16
    batch_size: int = 4
    max_seq_length: int = 2048

    # GRPO
    grpo_group_size: int = 4
    grpo_lr: float = 5e-5
    grpo_epochs: int = 1
    min_feedback_samples: int = 50  # min samples before GRPO retraining


class DataConfig(BaseSettings):
    """Data pipeline settings."""

    data_dir: Path = Path("data")
    synthesis_model: str = "claude-sonnet-4-20250514"
    pairs_per_passage: int = 1
    target_pairs: int = 1500
    val_ratio: float = 0.1


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class MarcusConfig(BaseSettings):
    """Root configuration for Marcus voice agent."""

    asr: ASRConfig = Field(default_factory=ASRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    model_config = {"env_prefix": "MARCUS_"}


def load_config(path: str | Path = "configs/default.yaml") -> MarcusConfig:
    """Load config from a YAML file with Pydantic validation.

    YAML values override defaults. Missing keys use defaults.
    """
    path = Path(path)
    if path.exists():
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
    else:
        overrides = {}

    return MarcusConfig(**overrides)
