"""Tests for the LLM wrapper — unit tests that mock mlx-lm.

We mock mlx-lm symbols at their import site (mlx_lm.*) since llm.py
imports them lazily inside method bodies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from marcus.config import LLMConfig
from marcus.models.llm import MarcusLLM


@pytest.fixture
def llm_config():
    return LLMConfig(
        model_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_tokens=128,
        temperature=0.7,
    )


class TestMarcusLLMUnit:
    """Unit tests that mock mlx-lm to avoid downloading models."""

    def test_generate_calls_mlx_generate(self, llm_config, sample_messages):
        """Verify generate() calls mlx_lm.generate with correct args."""
        llm = MarcusLLM(llm_config)
        llm._model = MagicMock()
        llm._tokenizer = MagicMock()
        llm._tokenizer.apply_chat_template.return_value = "formatted prompt"
        llm._loaded = True

        with patch("mlx_lm.generate", return_value="My friend, consider...") as mock_gen:
            result = llm.generate(sample_messages)

        assert result == "My friend, consider..."
        mock_gen.assert_called_once()

    def test_stream_generate_yields_tokens(self, llm_config, sample_messages):
        """Verify stream_generate() yields token strings."""
        llm = MarcusLLM(llm_config)
        llm._model = MagicMock()
        llm._tokenizer = MagicMock()
        llm._tokenizer.apply_chat_template.return_value = "formatted prompt"
        llm._loaded = True

        token_objects = [MagicMock(text=t) for t in ["My ", "friend, ", "consider."]]

        with patch("mlx_lm.stream_generate", return_value=iter(token_objects)):
            tokens = list(llm.stream_generate(sample_messages))

        assert tokens == ["My ", "friend, ", "consider."]

    def test_missing_adapter_warns_not_raises(self, llm_config):
        """Missing adapter path should warn but not crash."""
        config = llm_config.model_copy(update={"adapter_path": "/nonexistent/adapter"})
        llm = MarcusLLM(config)

        with patch("mlx_lm.load") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock())
            llm._load()

        # Should load without adapter_path argument (since it didn't exist)
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get("adapter_path") is None

    def test_unload_clears_model(self, llm_config):
        llm = MarcusLLM(llm_config)
        llm._model = MagicMock()
        llm._tokenizer = MagicMock()
        llm._loaded = True

        llm.unload()
        assert llm._model is None
        assert llm._tokenizer is None
        assert not llm._loaded
