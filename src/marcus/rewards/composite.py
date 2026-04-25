"""Composite reward function for GRPO training.

Combines rule-based signals with optional human feedback.
Used by the GRPO trainer as the reward function for policy optimization.
"""

from __future__ import annotations

from marcus.rewards.coherence import (
    length_reward,
    no_anachronism_score,
    persona_consistency_score,
)
from marcus.rewards.stoic_alignment import stoic_alignment_score


def composite_reward(
    response: str,
    user_feedback: float | None = None,
) -> float:
    """Compute composite reward for a Marcus Aurelius response.

    Components and weights:
        - Stoic alignment (is it actually Stoic?):    30%
        - Persona consistency (does it sound like Marcus?): 20%
        - No anachronisms (no modern slang/references):    15%
        - Response length (50-150 words for speech):       15%
        - Human feedback (thumbs up/down, normalized):     20% (if provided)
          or distributed equally across the above if absent

    Args:
        response: The generated response text.
        user_feedback: Optional float in [-1, 1] from human rating.
            +1.0 = thumbs up, -1.0 = thumbs down, None = no feedback.

    Returns:
        Float in [0, 1]. Higher is better.
    """
    rule_score = (
        0.375 * stoic_alignment_score(response)
        + 0.250 * persona_consistency_score(response)
        + 0.1875 * no_anachronism_score(response)
        + 0.1875 * length_reward(response)
    )

    if user_feedback is not None:
        # Normalize from [-1, 1] → [0, 1]
        feedback_score = (user_feedback + 1.0) / 2.0
        return 0.80 * rule_score + 0.20 * feedback_score
    else:
        return rule_score


def batch_rewards(responses: list[str], user_feedbacks: list[float | None] | None = None) -> list[float]:
    """Compute composite rewards for a batch of responses.

    Args:
        responses: List of response strings.
        user_feedbacks: Optional list of feedback floats (same length as responses).

    Returns:
        List of reward floats.
    """
    feedbacks = user_feedbacks or [None] * len(responses)
    return [
        composite_reward(r, fb)
        for r, fb in zip(responses, feedbacks)
    ]
