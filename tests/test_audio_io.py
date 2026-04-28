"""Tests for the audio I/O barge-in logic.

We exercise `AudioCapture._audio_callback` directly with synthetic audio
chunks rather than driving a real sounddevice stream — this lets us test
the VAD state machine deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest

from marcus.config import AudioConfig
from marcus.pipeline.audio_io import AudioCapture, AudioPlayer


def silent_chunk(audio_config: AudioConfig) -> np.ndarray:
    """A 2D float32 chunk of silence shaped (N, 1) — mirrors sounddevice's input."""
    n = int(audio_config.chunk_duration * audio_config.sample_rate)
    return np.zeros((n, 1), dtype=np.float32)


def voiced_chunk(audio_config: AudioConfig, rms: float = 0.05) -> np.ndarray:
    """A 2D float32 chunk with the requested RMS level."""
    n = int(audio_config.chunk_duration * audio_config.sample_rate)
    # White noise scaled to target RMS — keeps things simple and deterministic.
    rng = np.random.default_rng(seed=42)
    samples = rng.standard_normal(n).astype(np.float32)
    samples = samples / (np.sqrt(np.mean(samples**2)) + 1e-9) * rms
    return samples.reshape(-1, 1)


@pytest.fixture
def cfg():
    return AudioConfig(
        chunk_duration=0.2,
        silence_threshold=0.01,
        silence_duration=2.0,
        barge_in=True,
        barge_in_threshold_multiplier=3.0,
        barge_in_min_duration=0.6,
    )


class FakePlayer:
    """Stand-in for AudioPlayer that exposes is_playing / interrupt."""

    def __init__(self, is_playing: bool = False):
        self._is_playing = is_playing
        self.interrupt_count = 0

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def interrupt(self) -> None:
        self.interrupt_count += 1


class TestBargeInDetection:
    def test_brief_loud_blip_does_not_trigger(self, cfg):
        """One single loud chunk during playback shouldn't fire barge-in."""
        player = FakePlayer(is_playing=True)
        capture = AudioCapture(cfg, player=player)

        # One loud chunk (above 3x threshold), then silence
        capture._audio_callback(voiced_chunk(cfg, rms=0.05), 0, None, None)
        capture._audio_callback(silent_chunk(cfg), 0, None, None)

        assert player.interrupt_count == 0

    def test_sustained_loud_voice_triggers(self, cfg):
        """≥600ms of sustained loud voice during playback fires barge-in."""
        player = FakePlayer(is_playing=True)
        capture = AudioCapture(cfg, player=player)

        # 600ms / 200ms per chunk = 3 chunks of sustained voice
        for _ in range(4):
            capture._audio_callback(voiced_chunk(cfg, rms=0.05), 0, None, None)

        assert player.interrupt_count >= 1

    def test_quiet_speech_during_playback_ignored(self, cfg):
        """Speech below the 3x threshold (echo level) doesn't trigger."""
        player = FakePlayer(is_playing=True)
        capture = AudioCapture(cfg, player=player)

        # Below 3x threshold (0.03) — typical echo bleed-back energy
        for _ in range(10):
            capture._audio_callback(voiced_chunk(cfg, rms=0.02), 0, None, None)

        assert player.interrupt_count == 0

    def test_dynamic_threshold_overrides_static_multiplier(self, cfg):
        """When agent calibrates, the dynamic threshold takes effect."""
        player = FakePlayer(is_playing=True)
        capture = AudioCapture(cfg, player=player)

        # Static threshold would be: 0.01 * 3.0 = 0.03
        # Dynamic threshold from calibration: 0.05 (high bleed → high gate)
        capture.set_barge_in_threshold(0.05)

        # rms=0.04 is above static (0.03) but below dynamic (0.05) → no fire
        for _ in range(5):
            capture._audio_callback(voiced_chunk(cfg, rms=0.04), 0, None, None)
        assert player.interrupt_count == 0

        # rms=0.08 clears the dynamic threshold for sustained period → fire
        for _ in range(5):
            capture._audio_callback(voiced_chunk(cfg, rms=0.08), 0, None, None)
        assert player.interrupt_count >= 1

    def test_no_barge_in_when_disabled(self, cfg):
        """barge_in=False must never fire interrupt regardless of audio."""
        cfg = cfg.model_copy(update={"barge_in": False})
        player = FakePlayer(is_playing=True)
        capture = AudioCapture(cfg, player=player)

        for _ in range(10):
            capture._audio_callback(voiced_chunk(cfg, rms=0.1), 0, None, None)

        assert player.interrupt_count == 0


class TestNormalVAD:
    def test_idle_silence_buffers_nothing(self, cfg):
        capture = AudioCapture(cfg, player=FakePlayer(is_playing=False))
        for _ in range(5):
            capture._audio_callback(silent_chunk(cfg), 0, None, None)
        assert len(capture._buffer) == 0
        assert not capture._is_speaking

    def test_sustained_voice_starts_utterance(self, cfg):
        """≥300ms of sustained voice flips _is_speaking on."""
        capture = AudioCapture(cfg, player=FakePlayer(is_playing=False))
        # min_voiced_chunks_to_start = 0.3 / 0.2 = 2 chunks
        capture._audio_callback(voiced_chunk(cfg, rms=0.03), 0, None, None)
        capture._audio_callback(voiced_chunk(cfg, rms=0.03), 0, None, None)
        assert capture._is_speaking
        assert len(capture._buffer) > 0

    def test_paused_drops_input(self, cfg):
        capture = AudioCapture(cfg, player=FakePlayer(is_playing=False))
        capture.pause()
        for _ in range(10):
            capture._audio_callback(voiced_chunk(cfg, rms=0.1), 0, None, None)
        assert not capture._is_speaking
        assert len(capture._buffer) == 0


class TestAudioPlayer:
    def test_interrupt_flag_persists_until_reset(self, cfg):
        player = AudioPlayer(cfg)
        assert not player.was_interrupted
        player.interrupt()
        assert player.was_interrupted
        # Subsequent play() calls return immediately
        result = player.play(np.ones(1000, dtype=np.float32))
        assert result is False
        # Reset clears the flag
        player.reset_interrupt()
        assert not player.was_interrupted

    def test_empty_audio_returns_true(self, cfg):
        player = AudioPlayer(cfg)
        assert player.play(np.array([], dtype=np.float32)) is True
