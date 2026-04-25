"""Stoic alignment reward: scores how well a response reflects Stoic philosophy."""

from __future__ import annotations

# Stoic concepts and their associated keywords
STOIC_CONCEPTS: dict[str, list[str]] = {
    "dichotomy_of_control": [
        "control", "power", "within our", "up to us", "our own", "choice",
        "not in our", "external", "cannot control",
    ],
    "amor_fati": [
        "fate", "accept", "embrace", "love of fate", "what is", "as it is",
        "nature", "necessity", "providence",
    ],
    "memento_mori": [
        "death", "mortal", "finite", "fleeting", "impermanent", "brief",
        "transient", "perishable", "pass away", "end",
    ],
    "virtue": [
        "virtue", "virtuous", "wisdom", "courage", "justice", "temperance",
        "moderation", "right", "duty", "honour", "honorable",
    ],
    "logos": [
        "reason", "rational", "logos", "universal", "nature", "order",
        "law", "cosmos", "harmony", "providence",
    ],
    "present_moment": [
        "present", "now", "this moment", "at hand", "today", "here",
        "what is before", "immediate",
    ],
    "equanimity": [
        "calm", "tranquil", "equanimity", "serenity", "peace", "undisturbed",
        "untroubled", "composure", "stillness",
    ],
    "sympatheia": [
        "together", "connected", "one", "whole", "humanity", "fellow",
        "community", "brotherhood", "bond", "kinship",
    ],
}


def stoic_alignment_score(response: str) -> float:
    """Score a response on Stoic philosophical alignment.

    Returns a float in [0, 1] based on how many distinct Stoic concepts
    the response references. Full score requires 3+ distinct concepts.
    """
    response_lower = response.lower()
    concepts_found = sum(
        any(kw in response_lower for kw in keywords)
        for keywords in STOIC_CONCEPTS.values()
    )
    # Normalize: 3 concepts = 1.0, fewer = proportional
    return min(concepts_found / 3.0, 1.0)
