"""Tests for the data pipeline — runnable without models or network."""

import json
from pathlib import Path

import pytest

from marcus.data.clean import normalize_text, chunk_into_passages, clean_source
from marcus.data.download import strip_gutenberg_header_footer
from marcus.data.format import pairs_to_chat_format, train_val_split, load_system_prompt
from marcus.pipeline.conversation import ConversationManager


class TestGutenbergStripping:
    def test_strips_header(self):
        text = "Some preamble\n*** START OF THE PROJECT GUTENBERG EBOOK\nActual content here.\n"
        result = strip_gutenberg_header_footer(text)
        assert "preamble" not in result
        assert "Actual content" in result

    def test_strips_footer(self):
        text = "Good content.\n*** END OF THE PROJECT GUTENBERG EBOOK\nLegal boilerplate."
        result = strip_gutenberg_header_footer(text)
        assert "Good content" in result
        assert "Legal boilerplate" not in result

    def test_no_markers_returns_stripped(self):
        text = "  Plain text without markers.  "
        result = strip_gutenberg_header_footer(text)
        assert result == "Plain text without markers."


class TestNormalizeText:
    def test_removes_fancy_quotes(self):
        result = normalize_text("“Hello”")
        assert '"Hello"' in result

    def test_collapses_multiple_newlines(self):
        result = normalize_text("Line 1\n\n\n\n\nLine 2")
        assert result.count("\n") == 2  # exactly one blank line

    def test_strips_trailing_whitespace(self):
        result = normalize_text("Line 1   \nLine 2   ")
        assert "Line 1" in result
        assert "   " not in result


class TestChunkIntoPassages:
    def test_produces_passages_in_word_range(self):
        text = "This is a sentence. " * 100
        passages = chunk_into_passages(text, min_words=30, max_words=300)
        for p in passages:
            words = len(p.split())
            assert words >= 20  # some slack for edge cases

    def test_empty_text_returns_empty(self):
        assert chunk_into_passages("") == []

    def test_short_text_below_minimum_ignored(self):
        passages = chunk_into_passages("Hi there.", min_words=30, max_words=300)
        assert len(passages) == 0


class TestFormatPipeline:
    def test_pairs_to_chat_format(self):
        pairs = [{"user": "I am stressed.", "marcus": "Consider what you control."}]
        formatted = pairs_to_chat_format(pairs, system_prompt="You are Marcus.")
        assert len(formatted) == 1
        msgs = formatted[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "Consider what you control."

    def test_train_val_split_ratio(self):
        data = [{"x": i} for i in range(100)]
        train, val = train_val_split(data, val_ratio=0.1)
        assert len(train) == 90
        assert len(val) == 10
        assert len(train) + len(val) == len(data)

    def test_train_val_no_overlap(self):
        data = [{"x": i} for i in range(100)]
        train, val = train_val_split(data, val_ratio=0.2)
        train_set = {d["x"] for d in train}
        val_set = {d["x"] for d in val}
        assert train_set.isdisjoint(val_set)

    def test_load_system_prompt(self, tmp_path):
        prompt = "You are Marcus Aurelius."
        (tmp_path / "system_prompt.txt").write_text(prompt)
        loaded = load_system_prompt(str(tmp_path / "system_prompt.txt"))
        assert loaded == prompt


class TestConversationManager:
    def test_get_messages_includes_system(self):
        conv = ConversationManager(system_prompt="You are Marcus.")
        conv.add_user("Hello.")
        messages = conv.get_messages()
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are Marcus."

    def test_turn_alternation(self):
        conv = ConversationManager(system_prompt="You are Marcus.")
        conv.add_user("Question 1.")
        conv.add_assistant("Answer 1.")
        conv.add_user("Question 2.")
        messages = conv.get_messages()
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]

    def test_rolling_window_trims(self):
        conv = ConversationManager(system_prompt="S", max_turns=2)
        for i in range(5):
            conv.add_user(f"Q{i}")
            conv.add_assistant(f"A{i}")
        # max_turns=2 → keep last 4 messages (2 complete turns)
        messages = conv.get_messages()
        # system + 4 history messages = 5 exactly
        assert len(messages) == 5
        # History starts with user (no orphaned assistant message)
        assert messages[1]["role"] == "user"

    def test_clear_resets_history(self):
        conv = ConversationManager(system_prompt="S")
        conv.add_user("Hello")
        conv.add_assistant("World")
        conv.clear()
        assert conv.get_messages() == [{"role": "system", "content": "S"}]

    def test_last_messages(self):
        conv = ConversationManager(system_prompt="S")
        conv.add_user("My question.")
        conv.add_assistant("My answer.")
        assert conv.last_user_message == "My question."
        assert conv.last_assistant_message == "My answer."
