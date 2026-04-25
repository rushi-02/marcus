"""Tests for reward functions — runnable without models."""

import pytest

from marcus.rewards.stoic_alignment import stoic_alignment_score
from marcus.rewards.coherence import persona_consistency_score, length_reward, no_anachronism_score
from marcus.rewards.composite import composite_reward


class TestStoicAlignment:
    def test_stoic_response_scores_high(self):
        response = (
            "My friend, remember that death is natural and within the order of "
            "the cosmos. Virtue is your only true possession; all external things "
            "are beyond our control. Accept fate with equanimity."
        )
        score = stoic_alignment_score(response)
        assert score >= 0.6

    def test_generic_response_scores_low(self):
        response = "Just chill out and don't worry about it. Things will be fine."
        score = stoic_alignment_score(response)
        assert score < 0.3

    def test_score_bounded(self):
        for text in ["", "hello", "virtue wisdom death control accept fate reason"]:
            score = stoic_alignment_score(text)
            assert 0.0 <= score <= 1.0

    def test_empty_returns_zero(self):
        assert stoic_alignment_score("") == 0.0


class TestPersonaConsistency:
    def test_marcus_voice_scores_high(self):
        response = "My friend, consider what lies within your power. Reflect upon this."
        score = persona_consistency_score(response)
        assert score > 0.5

    def test_modern_slang_scores_low(self):
        response = "Lol yeah totally, no worries! Honestly speaking, just chill."
        score = persona_consistency_score(response)
        assert score < 0.5

    def test_score_bounded(self):
        for text in ["", "hello", "my friend consider reflect remember"]:
            score = persona_consistency_score(text)
            assert 0.0 <= score <= 1.0


class TestLengthReward:
    def test_target_range_scores_one(self):
        words = " ".join(["word"] * 80)
        assert length_reward(words) == 1.0

    def test_too_short_penalized(self):
        words = " ".join(["word"] * 10)
        assert length_reward(words) < 1.0

    def test_too_long_penalized(self):
        words = " ".join(["word"] * 300)
        assert length_reward(words) < 1.0

    def test_score_bounded(self):
        for n in [0, 10, 80, 150, 300, 1000]:
            score = length_reward(" ".join(["word"] * n))
            assert 0.0 <= score <= 1.0


class TestNoAnachronism:
    def test_clean_response_scores_one(self):
        response = "Remember that virtue is the only true good. Control your judgment."
        assert no_anachronism_score(response) == 1.0

    def test_modern_terms_penalized(self):
        response = "Post about it on Instagram and check your smartphone notifications."
        score = no_anachronism_score(response)
        assert score < 1.0

    def test_multiple_violations_penalized_more(self):
        single = "Check your smartphone."
        multiple = "Check your smartphone, go on YouTube, use the internet."
        assert no_anachronism_score(multiple) < no_anachronism_score(single)


class TestCompositeReward:
    def test_good_response_scores_above_point5(self, sample_stoic_response):
        score = composite_reward(sample_stoic_response)
        assert score > 0.5

    def test_thumbs_up_improves_score(self, sample_stoic_response):
        base = composite_reward(sample_stoic_response)
        with_feedback = composite_reward(sample_stoic_response, user_feedback=1.0)
        assert with_feedback >= base

    def test_thumbs_down_lowers_score(self, sample_stoic_response):
        base = composite_reward(sample_stoic_response)
        with_feedback = composite_reward(sample_stoic_response, user_feedback=-1.0)
        assert with_feedback <= base

    def test_score_bounded(self, sample_stoic_response):
        for feedback in [None, -1.0, 0.0, 1.0]:
            score = composite_reward(sample_stoic_response, user_feedback=feedback)
            assert 0.0 <= score <= 1.0
