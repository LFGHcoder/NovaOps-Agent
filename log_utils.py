"""Lightweight structured logging utility."""

import json
from datetime import datetime, timezone


def log_event(stage: str, task_id: str, message: str) -> None:
    """Print a single structured JSON log line."""
    payload = {
        "stage": stage,
        "task_id": task_id,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload))
