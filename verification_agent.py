"""Verification agent: validates task state before marking complete."""

from task import Task, TaskStatus
from log_utils import log_step


def _is_valid(task: Task) -> tuple[bool, str]:
    """Return (valid, error_message). Error message empty if valid."""
    if not task.evaluation:
        return False, "evaluation is missing or empty"
    if not task.execution_logs:
        return False, "execution_logs is empty"
    for entry in task.execution_logs:
        if isinstance(entry, dict) and entry.get("result") in ("failure", "failed"):
            return False, "execution_logs contains failure entry"
    return True, ""


class VerificationAgent:
    """Validates evaluation and execution_logs; sets status to complete or failed."""

    def verify(self, task: Task) -> Task:
        """If valid: status = complete. Else: mark_failed and return updated task."""
        log_step("Verification Agent", "Validating execution results")
        valid, msg = _is_valid(task)
        if valid:
            log_step("Verification Agent", "Validation passed", {"status": "completed"})
            return task.model_copy(update={"status": TaskStatus.complete}, deep=True)
        log_step("Verification Agent", "Validation failed", {"reason": msg})
        return task.model_copy(
            update={"status": TaskStatus.failed, "error": msg},
            deep=True,
        )
