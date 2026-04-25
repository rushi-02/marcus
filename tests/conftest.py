"""Shared pytest fixtures for Marcus test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marcus.config import MarcusConfig, ASRConfig, LLMConfig, TTSConfig, AudioConfig, TrainingConfig, DataConfig


@pytest.fixture
def config() -> MarcusConfig:
    """Minimal MarcusConfig for testing (no real model paths needed for unit tests)."""
    return MarcusConfig()


@pytest.fixture
def audio_config() -> AudioConfig:
    return AudioConfig()


@pytest.fixture
def sample_audio() -> np.ndarray:
    """1 second of 16kHz sine wave audio (440 Hz) for testing ASR."""
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def silence_audio() -> np.ndarray:
    """1 second of silence."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def sample_messages() -> list[dict]:
    """Sample conversation messages for LLM testing."""
    return [
        {
            "role": "system",
            "content": "You are Marcus Aurelius, Stoic philosopher.",
        },
        {
            "role": "user",
            "content": "I feel overwhelmed at work and don't know how to cope.",
        },
    ]


@pytest.fixture
def sample_stoic_response() -> str:
    return (
        "My friend, consider what is within your power. The weight you feel is "
        "not from the work itself, but from your judgment of it. Remember the "
        "dichotomy of control: focus only on what you can act upon, and let the "
        "rest pass as clouds across the sky. Virtue lies in your response, not "
        "in the circumstances."
    )


@pytest.fixture
def sample_instruction_pair() -> dict:
    return {
        "user": "I feel overwhelmed at work.",
        "marcus": "My friend, consider what lies within your power...",
    }


@pytest.fixture
def system_prompt(tmp_path) -> str:
    prompt = "You are Marcus Aurelius, Stoic philosopher."
    prompt_file = tmp_path / "system_prompt.txt"
    prompt_file.write_text(prompt)
    return prompt


@pytest.fixture
def data_dir(tmp_path) -> Path:
    """Temporary data directory for pipeline tests."""
    d = tmp_path / "data"
    d.mkdir()
    (d / "raw").mkdir()
    (d / "processed").mkdir()
    (d / "synthetic").mkdir()
    (d / "training").mkdir()
    return d
