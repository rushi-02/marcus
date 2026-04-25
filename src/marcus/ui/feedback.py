"""User feedback collection for the GRPO RL loop.

Records thumbs up/down after each Marcus response.
Feedback is appended to data/feedback_log.jsonl and used to trigger
GRPO retraining when enough samples have been collected.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class FeedbackCollector:
    """Logs user feedback (thumbs up/down) for RL training.

    Usage:
        collector = FeedbackCollector()
        collector.record(user_message="...", assistant_message="...", thumbs_up=True)
    """

    def __init__(
        self,
        log_path: str | Path = "data/feedback_log.jsonl",
        session_id: str | None = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())[:8]

    def record(
        self,
        user_message: str,
        assistant_message: str,
        thumbs_up: bool,
    ) -> None:
        """Append a feedback entry to the log.

        reward: +1.0 for thumbs up, -1.0 for thumbs down.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "user": user_message,
            "assistant": assistant_message,
            "thumbs_up": thumbs_up,
            "reward": 1.0 if thumbs_up else -1.0,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def count(self) -> int:
        """Return total number of feedback entries collected."""
        if not self.log_path.exists():
            return 0
        with open(self.log_path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def load_all(self) -> list[dict]:
        """Load all feedback entries."""
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
