"""Lightweight structured logging utility."""

import json
from datetime import datetime, timezone
from typing import Any, Optional


def log_event(stage: str, task_id: str, message: str) -> None:
    """Print a single structured JSON log line."""
    payload = {
        "stage": stage,
        "task_id": task_id,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload))


def log_step(
    agent_name: str,
    message: str,
    optional_data: Optional[dict[str, Any]] = None,
) -> None:
    """
    Print a readable step log for multi-agent demo.
    Format:
      [Agent Name]
      Message
      Result: key=value, ...  (if optional_data provided)
    """
    print(f"\n[{agent_name}]")
    print(message)
    if optional_data is not None and optional_data:
        parts = ", ".join(f"{k}={v}" for k, v in optional_data.items())
        print(f"Result: {parts}")
