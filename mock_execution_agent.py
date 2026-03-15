"""Mock execution agent: simulates hiring dashboard actions."""

from datetime import datetime, timezone

from task import Task, TaskStatus


def _log_entry(action: str, result: str = "success") -> dict[str, str]:
    """Build a single execution_logs entry."""
    return {
        "action": action,
        "result": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class MockExecutionAgent:
    """
    Simulates execution: login to hiring dashboard, update candidate score,
    generate summary report. Appends to execution_logs and sets status to verified.
    """

    def __init__(self) -> None:
        pass

    def execute(self, task: Task) -> Task:
        """
        Simulate the three actions, append execution_logs, set status = verified.
        """
        logs = list(task.execution_logs or [])

        logs.append(_log_entry("login_hiring_dashboard"))
        logs.append(_log_entry("update_candidate_score"))
        logs.append(_log_entry("generate_summary_report"))

        return task.model_copy(
            update={
                "execution_logs": logs,
                "status": TaskStatus.verified,
            },
            deep=True,
        )
