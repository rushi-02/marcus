"""Tests for the TTS wrapper — unit tests that mock mlx-audio / f5-tts-mlx."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from marcus.config import TTSConfig
from marcus.models.tts import MarcusTTS, _split_sentences


@pytest.fixture
def tts_config(tmp_path):
    voice_ref = tmp_path / "voice.wav"
    voice_ref.write_bytes(b"RIFF" + b"\x00" * 36)  # minimal WAV stub
    return TTSConfig(
        model_id="lucasnewman/f5-tts-mlx",
        voice_ref_path=str(voice_ref),
    )


class TestSplitSentences:
    def test_splits_on_period(self):
        sentences = _split_sentences("Hello world. How are you?")
        assert len(sentences) == 2
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"

    def test_splits_on_multiple_endings(self):
        text = "One. Two! Three? Four:"
        sentences = _split_sentences(text)
        assert len(sentences) == 4

    def test_empty_returns_empty(self):
        assert _split_sentences("") == []

    def test_no_punctuation_returns_single(self):
        text = "A sentence without ending punctuation"
        sentences = _split_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == text


class TestMarcusTTSUnit:
    def test_synthesize_returns_ndarray(self, tts_config):
        fake_audio = np.zeros(24000, dtype=np.float32)
        mock_generate = MagicMock(return_value=(fake_audio, 24000))

        with patch("marcus.models.tts.MarcusTTS._load"):
            tts = MarcusTTS(tts_config)
            tts._loaded = True
            tts._backend = "f5_tts_mlx"
            tts._voice_ref = str(tts_config.voice_ref_path)

        with patch("marcus.models.tts.MarcusTTS._synthesize_f5", return_value=fake_audio):
            result = tts.synthesize("Remember, virtue is its own reward.")

        assert isinstance(result, np.ndarray)

    def test_empty_text_returns_empty(self, tts_config):
        with patch("marcus.models.tts.MarcusTTS._load"):
            tts = MarcusTTS(tts_config)
            tts._loaded = True
            tts._backend = "f5_tts_mlx"

        result = tts.synthesize("")
        assert len(result) == 0

    def test_synthesize_stream_yields_chunks(self, tts_config):
        fake_audio = np.ones(24000, dtype=np.float32)

        with patch("marcus.models.tts.MarcusTTS._load"):
            tts = MarcusTTS(tts_config)
            tts._loaded = True
            tts._backend = "f5_tts_mlx"

        with patch.object(tts, "synthesize", return_value=fake_audio):
            chunks = list(tts.synthesize_stream("First sentence. Second sentence!"))

        assert len(chunks) == 2
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)
