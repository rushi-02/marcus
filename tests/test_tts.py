"""Tests for the TTS wrapper — Kokoro via mlx-audio."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from marcus.config import TTSConfig
from marcus.models.tts import MarcusTTS, _split_sentences


@pytest.fixture
def tts_config():
    return TTSConfig(
        model_id="prince-canuma/Kokoro-82M",
        voice="bm_george",
        lang_code="b",
        speed=0.95,
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

        # Mock the loaded Kokoro model: model.generate() yields GenerationResult-like objects
        mock_result = MagicMock(audio=fake_audio)
        mock_model = MagicMock()
        mock_model.generate.return_value = iter([mock_result])

        tts = MarcusTTS(tts_config)
        tts._model = mock_model
        tts._loaded = True

        result = tts.synthesize("Remember, virtue is its own reward.")
        assert isinstance(result, np.ndarray)
        assert len(result) == 24000

    def test_empty_text_returns_empty(self, tts_config):
        tts = MarcusTTS(tts_config)
        tts._loaded = True
        # Don't even need a model since empty text should short-circuit
        result = tts.synthesize("")
        assert len(result) == 0

    def test_synthesize_stream_yields_chunks(self, tts_config):
        fake_audio = np.ones(24000, dtype=np.float32)

        tts = MarcusTTS(tts_config)
        tts._loaded = True

        with patch.object(tts, "synthesize", return_value=fake_audio):
            chunks = list(tts.synthesize_stream("First sentence. Second sentence!"))

        assert len(chunks) == 2
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)
