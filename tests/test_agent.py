"""Tests for MarcusAgent — unit tests using mocked components."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from marcus.config import load_config


class TestMarcusAgentTextMode:
    """Test text_chat() mode which doesn't require audio hardware."""

    @pytest.mark.asyncio
    async def test_text_chat_adds_to_conversation(self):
        """Verify text_chat() correctly routes user input through conversation."""
        with (
            patch("marcus.models.asr.MarcusASR._load"),
            patch("marcus.models.llm.MarcusLLM._load"),
            patch("marcus.models.tts.MarcusTTS._load"),
            patch("marcus.models.llm.generate", return_value="My friend, consider..."),
        ):
            from marcus.pipeline.agent import MarcusAgent
            config = load_config()
            agent = MarcusAgent(config)
            agent.llm._loaded = True
            agent.llm._model = MagicMock()
            agent.llm._tokenizer = MagicMock()
            agent.llm._tokenizer.apply_chat_template.return_value = "prompt"

            # Simulate one text exchange
            agent.conversation.add_user("I feel anxious about exams.")
            response = agent.llm.generate(agent.conversation.get_messages())
            agent.conversation.add_assistant(response)

        assert agent.conversation.turn_count == 1
        assert agent.conversation.last_user_message == "I feel anxious about exams."

    def test_feedback_recording(self, tmp_path):
        """Verify feedback is persisted after record_feedback() call."""
        with (
            patch("marcus.models.asr.MarcusASR._load"),
            patch("marcus.models.llm.MarcusLLM._load"),
            patch("marcus.models.tts.MarcusTTS._load"),
        ):
            from marcus.pipeline.agent import MarcusAgent
            from marcus.ui.feedback import FeedbackCollector
            config = load_config()
            agent = MarcusAgent(config)
            agent.feedback = FeedbackCollector(log_path=tmp_path / "feedback.jsonl")

            agent.conversation.add_user("I'm stressed.")
            agent.conversation.add_assistant("Consider what you control.")

            agent.record_feedback(thumbs_up=True)

        assert agent.feedback.count() == 1
        entries = agent.feedback.load_all()
        assert entries[0]["thumbs_up"] is True
        assert entries[0]["reward"] == 1.0
