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
    """Automatic Speech Recognition (Whisper via mlx-audio).

    Default uses whisper-large-v3-turbo (~798M params, 2.5 GB peak memory,
    ~9x real-time on M2 Pro). For tighter memory budgets, fall back to
    `mlx-community/whisper-tiny` (~75 MB) or `mlx-community/whisper-base`.
    """

    model_id: str = "mlx-community/whisper-large-v3-turbo"
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
    """Text-to-Speech via mlx-audio (Kokoro by default).

    Kokoro voices follow the format `<accent><gender>_<name>`:
    - bm_george, bm_lewis, bm_fable    (British male, deep voices)
    - am_michael, am_adam, am_eric    (American male)
    - bf_emma, bf_isabella            (British female)
    - af_bella, af_sarah, af_nicole   (American female)

    `bm_george` is the default — deep British baritone, well-suited to Marcus.
    """

    model_id: str = "prince-canuma/Kokoro-82M"
    voice: str = "bm_george"
    lang_code: str = "b"  # 'a'=American, 'b'=British, 'e'=es, 'f'=fr, 'h'=hi, etc.
    speed: float = 0.95   # slightly slower for deliberate Stoic delivery
    sample_rate: int = 24000
    voice_ref_path: str | None = None  # not used by Kokoro; kept for future cloning backends
    streaming: bool = True


class AudioConfig(BaseSettings):
    """Audio I/O settings for mic capture and speaker playback."""

    sample_rate: int = 16000
    channels: int = 1
    # Smaller chunks → finer-grained VAD; 200ms is a good balance of latency
    # and CPU cost for the energy-based VAD callback.
    chunk_duration: float = 0.2
    device_input: int | None = None
    device_output: int | None = None
    # RMS energy threshold for VAD. 0.015 leaves quiet rooms in the silence
    # bucket while still catching normal speech. Bump higher (0.02-0.03) in
    # noisy environments; lower (0.005-0.01) for soft-spoken users.
    silence_threshold: float = 0.015
    # How long of continuous silence before we declare end-of-utterance.
    # 2.0s gives the user breathing room mid-thought without dragging too long.
    silence_duration: float = 2.0
    playback_chunk_ms: int = 100


class TrainingConfig(BaseSettings):
    """Fine-tuning hyperparameters for SFT and GRPO."""

    # SFT (LoRA)
    # NOTE: with ~93 train examples, 1 epoch ≈ 23 iters. Past iter 25, val
    # loss climbs sharply (severe overfit). Keep epochs low.
    sft_epochs: int = 1
    sft_lr: float = 1e-4
    lora_rank: int = 8
    lora_alpha: int = 16
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
