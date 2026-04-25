"""Tests for the LLM wrapper — unit tests that mock mlx-lm.

Integration tests (actual model inference) require mlx-lm and ~3GB download.
Run with: uv run pytest tests/test_llm.py -k integration --run-integration
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
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        with patch("marcus.models.llm.MarcusLLM._load"):
            llm = MarcusLLM(llm_config)
            llm._model = mock_model
            llm._tokenizer = mock_tokenizer
            llm._loaded = True

            with patch("marcus.models.llm.generate", return_value="My friend, consider...") as mock_gen:
                result = llm.generate(sample_messages)

        assert result == "My friend, consider..."
        mock_gen.assert_called_once()

    def test_stream_generate_yields_tokens(self, llm_config, sample_messages):
        """Verify stream_generate() yields token strings."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        token_objects = [MagicMock(text=t) for t in ["My ", "friend, ", "consider."]]

        with patch("marcus.models.llm.MarcusLLM._load"):
            llm = MarcusLLM(llm_config)
            llm._model = mock_model
            llm._tokenizer = mock_tokenizer
            llm._loaded = True

            with patch("marcus.models.llm.stream_generate", return_value=iter(token_objects)):
                tokens = list(llm.stream_generate(sample_messages))

        assert tokens == ["My ", "friend, ", "consider."]

    def test_missing_adapter_warns_not_raises(self, llm_config, capsys):
        """Missing adapter path should warn but not crash."""
        config = llm_config.model_copy(update={"adapter_path": "/nonexistent/adapter"})

        with patch("marcus.models.llm.load") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock())
            llm = MarcusLLM(config)
            llm._load()

        # Should load without adapter_path argument
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get("adapter_path") is None

    def test_unload_clears_model(self, llm_config):
        with patch("marcus.models.llm.MarcusLLM._load"):
            llm = MarcusLLM(llm_config)
            llm._model = MagicMock()
            llm._tokenizer = MagicMock()
            llm._loaded = True

        llm.unload()
        assert llm._model is None
        assert llm._tokenizer is None
        assert not llm._loaded
