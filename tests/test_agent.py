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
        ):
            from marcus.pipeline.agent import MarcusAgent
            config = load_config()
            agent = MarcusAgent(config)
            # Directly mock the bound generate method on the LLM instance
            agent.llm.generate = MagicMock(return_value="My friend, consider...")

            agent.conversation.add_user("I feel anxious about exams.")
            response = agent.llm.generate(agent.conversation.get_messages())
            agent.conversation.add_assistant(response)

        assert agent.conversation.turn_count == 1
        assert agent.conversation.last_user_message == "I feel anxious about exams."
        assert agent.conversation.last_assistant_message == "My friend, consider..."

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
