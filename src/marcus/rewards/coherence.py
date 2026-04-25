"""Coherence reward: scores response quality and Marcus persona consistency."""

from __future__ import annotations

# Voice markers indicating Marcus Aurelius persona
PERSONA_MARKERS = [
    "my friend", "consider", "reflect", "remember", "observe",
    "know this", "understand", "think upon", "recall", "bear in mind",
    "ask yourself", "it is", "we must", "one must",
]

# Anti-markers that break the Marcus Aurelius persona
ANTI_MARKERS = [
    "lol", "haha", "yeah", "nope", "sure thing", "no worries", "awesome",
    "cool", "hey there", "absolutely", "totally", "tbh", "tbf",
    "honestly speaking", "to be frank", "let me tell you",
    "as an ai", "as a language model",
]


def persona_consistency_score(response: str) -> float:
    """Score persona consistency for the Marcus Aurelius voice.

    Returns float in [0, 1]:
    - Starts at 0.5 (baseline)
    - +0.1 per persona marker found (up to 0.5 bonus)
    - -0.2 per anti-marker found (floored at 0.0)
    """
    response_lower = response.lower()

    score = 0.5
    for marker in PERSONA_MARKERS:
        if marker in response_lower:
            score += 0.1
    for anti in ANTI_MARKERS:
        if anti in response_lower:
            score -= 0.2

    return max(0.0, min(1.0, score))


def length_reward(response: str, target_min: int = 50, target_max: int = 150) -> float:
    """Score response length suitability for spoken dialogue.

    Target: 50-150 words. Shorter or longer responses score proportionally lower.
    """
    word_count = len(response.split())

    if target_min <= word_count <= target_max:
        return 1.0
    elif word_count < target_min:
        return word_count / target_min
    else:
        excess = word_count - target_max
        return max(0.0, 1.0 - excess / target_max)


def no_anachronism_score(response: str) -> float:
    """Penalize modern references that break historical immersion.

    Returns 1.0 if clean, 0.0 if anachronisms detected.
    """
    modern_terms = [
        "smartphone", "internet", "computer", "social media", "email",
        "linkedin", "instagram", "twitter", "facebook", "youtube",
        "netflix", "google", "amazon", "ai", "machine learning",
        "algorithm", "app", "online", "website", "password",
        "stress leave", "work from home", "zoom", "podcast",
    ]
    response_lower = response.lower()
    violations = [t for t in modern_terms if t in response_lower]

    if not violations:
        return 1.0
    return max(0.0, 1.0 - 0.3 * len(violations))
