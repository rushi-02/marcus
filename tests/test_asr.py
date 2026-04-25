"""Tests for the ASR wrapper — unit tests that mock mlx-audio."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from marcus.config import ASRConfig
from marcus.models.asr import MarcusASR


@pytest.fixture
def asr_config():
    return ASRConfig(model_id="mlx-community/whisper-small", language="en")


class TestMarcusASRUnit:
    def test_transcribe_returns_string(self, asr_config, sample_audio):
        mock_model = MagicMock()
        mock_model.generate.return_value = "I feel overwhelmed."

        with patch("marcus.models.asr.MarcusASR._load"):
            asr = MarcusASR(asr_config)
            asr._model = mock_model
            asr._loaded = True

        result = asr.transcribe(sample_audio, sample_rate=16000)
        assert isinstance(result, str)
        assert result == "I feel overwhelmed."

    def test_empty_audio_returns_empty(self, asr_config):
        with patch("marcus.models.asr.MarcusASR._load"):
            asr = MarcusASR(asr_config)
            asr._loaded = True

        result = asr.transcribe(np.array([]), sample_rate=16000)
        assert result == ""

    def test_result_with_text_attr(self, asr_config, sample_audio):
        """Handles result objects with .text attribute."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "  wisdom awaits  "
        mock_model.generate.return_value = mock_result

        with patch("marcus.models.asr.MarcusASR._load"):
            asr = MarcusASR(asr_config)
            asr._model = mock_model
            asr._loaded = True

        result = asr.transcribe(sample_audio)
        assert result == "wisdom awaits"

    def test_stereo_audio_converted_to_mono(self, asr_config):
        mock_model = MagicMock()
        mock_model.generate.return_value = "hello"

        stereo = np.random.randn(16000, 2).astype(np.float32)

        with patch("marcus.models.asr.MarcusASR._load"):
            asr = MarcusASR(asr_config)
            asr._model = mock_model
            asr._loaded = True

        result = asr.transcribe(stereo)
        # Should not raise; model was called with mono
        called_audio = mock_model.generate.call_args[0][0]
        assert called_audio.ndim == 1

    def test_unload_resets_state(self, asr_config):
        with patch("marcus.models.asr.MarcusASR._load"):
            asr = MarcusASR(asr_config)
            asr._model = MagicMock()
            asr._loaded = True

        asr.unload()
        assert asr._model is None
        assert not asr._loaded
