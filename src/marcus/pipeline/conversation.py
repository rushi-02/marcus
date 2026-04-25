"""Conversation history manager with rolling window and system prompt."""

from __future__ import annotations

from pathlib import Path


class ConversationManager:
    """Manages multi-turn conversation history for the Marcus agent.

    Maintains a rolling window of the last N turns so the LLM context
    window doesn't overflow during long sessions.

    Usage:
        conv = ConversationManager(system_prompt="You are Marcus Aurelius...")
        conv.add_user("I feel overwhelmed.")
        conv.add_assistant("My friend, consider...")
        messages = conv.get_messages()  # pass to llm.generate()
    """

    def __init__(self, system_prompt: str, max_turns: int = 10) -> None:
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_user(self, text: str) -> None:
        """Append a user message to history."""
        self._history.append({"role": "user", "content": text.strip()})
        self._trim()

    def add_assistant(self, text: str) -> None:
        """Append an assistant (Marcus) message to history."""
        self._history.append({"role": "assistant", "content": text.strip()})
        self._trim()

    def clear(self) -> None:
        """Reset conversation history (keep system prompt)."""
        self._history = []

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_messages(self) -> list[dict]:
        """Return full messages list including system prompt."""
        return [
            {"role": "system", "content": self.system_prompt},
            *self._history,
        ]

    @property
    def turn_count(self) -> int:
        """Number of complete user+assistant turns."""
        return len(self._history) // 2

    @property
    def last_user_message(self) -> str | None:
        for msg in reversed(self._history):
            if msg["role"] == "user":
                return msg["content"]
        return None

    @property
    def last_assistant_message(self) -> str | None:
        for msg in reversed(self._history):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """Keep only the last max_turns * 2 messages (user + assistant pairs)."""
        max_messages = self.max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]


def load_system_prompt(path: str = "configs/system_prompt.txt") -> str:
    """Load the Marcus Aurelius system prompt from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"System prompt not found at '{path}'. "
            "Check configs/system_prompt.txt exists."
        )
    return p.read_text(encoding="utf-8").strip()
