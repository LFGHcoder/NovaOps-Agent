"""Workflow manager: orchestrates the HR automation pipeline."""

import logging
from typing import Any, Optional, Protocol

from task import Task, TaskStatus, Rubric
from planner_agent import PlannerAgent
from interview_agent import InterviewAgent
from evaluation_agent import EvaluationAgent
from mock_execution_agent import MockExecutionAgent
from verification_agent import VerificationAgent
from log_utils import log_step


logger = logging.getLogger(__name__)

BANNER = (
    "==================================================\n"
    "NovaOps Agent - AI Recruiting Workflow Automation\n"
    "=================================================="
)


# Canonical result keys for consistent JSON output.
RESULT_CANDIDATE = "candidate"
RESULT_OVERALL_SCORE = "overall_score"
RESULT_RECOMMENDATION = "recommendation"
RESULT_STATUS = "status"
RESULT_ERROR = "error"
RESULT_CATEGORY_SCORES = "category_scores"


def result_to_json(task: Task) -> dict[str, Any]:
    """
    Return a consistent JSON-serializable result from a task.
    Always includes: candidate, overall_score, recommendation, status, error.
    Includes category_scores when evaluation has rubric-based category scores.
    """
    evaluation = task.evaluation or {}
    out: dict[str, Any] = {
        RESULT_CANDIDATE: task.candidate_data.get("name") if isinstance(task.candidate_data, dict) else None,
        RESULT_OVERALL_SCORE: evaluation.get("overall_score"),
        RESULT_RECOMMENDATION: evaluation.get("recommendation"),
        RESULT_STATUS: task.status.value,
        RESULT_ERROR: task.error,
    }
    if "category_scores" in evaluation and evaluation["category_scores"]:
        out[RESULT_CATEGORY_SCORES] = evaluation["category_scores"]
    return out


class ExecutionAgentProtocol(Protocol):
    """Protocol for execution agents. Allows swapping MockExecutionAgent for NovaActExecutionAgent."""

    def execute(self, task: Task) -> Task:
        ...


class WorkflowManager:
    """
    Orchestrates the full pipeline: plan → interview → evaluate → execute → verify.
    Controls status transitions and handles errors. No agent calls another agent.
    """

    def __init__(
        self,
        execution_agent: Optional[ExecutionAgentProtocol] = None,
    ) -> None:
        self._planner = PlannerAgent()
        self._interview_agent = InterviewAgent()
        self._evaluation_agent = EvaluationAgent()
        self._execution_agent = execution_agent if execution_agent is not None else MockExecutionAgent()
        self._verification_agent = VerificationAgent()

    def _ensure_status(self, task: Task, expected: TaskStatus, stage: str) -> Task:
        """If task is failed or has unexpected status, return failed task; else return task."""
        if task.status == TaskStatus.failed:
            return task
        if task.status != expected:
            return task.model_copy(
                update={
                    "status": TaskStatus.failed,
                    "error": f"{stage}: expected status {expected.value}, got {task.status.value}",
                },
                deep=True,
            )
        return task

    def _mark_failed(self, task: Task, stage: str, message: str) -> Task:
        """Return a copy of task with status=failed and error set to a consistent string."""
        return task.model_copy(
            update={
                "status": TaskStatus.failed,
                "error": f"[{stage}] {message}",
            },
            deep=True,
        )

    def run(
        self,
        job_description: str,
        candidate_data: dict[str, Any],
        rubric: Optional[dict[str, Any]] = None,
    ) -> Task:
        """
        Run the full pipeline. On any failure the task is marked failed and returned.
        Errors are caught per step and as a top-level fallback; error message is always set when failed.
        """
        print(BANNER)
        task: Task
        try:
            rubric_model = None
            if rubric is not None:
                try:
                    rubric_model = Rubric.model_validate(rubric)
                except Exception as e:  # noqa: BLE001
                    logger.warning("rubric validation failed: %s", e)
                    task = Task(
                        job_description=job_description,
                        candidate_data=candidate_data,
                        rubric=None,
                    )
                    log_step("Workflow Manager", "Pipeline failed", {"error": f"Invalid rubric: {e!s}"})
                    return self._mark_failed(task, "init", f"Invalid rubric: {e!s}")
            task = Task(
                job_description=job_description,
                candidate_data=candidate_data,
                rubric=rubric_model,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("task init failed: %s", e)
            task = Task(job_description=job_description, candidate_data=candidate_data)
            log_step("Workflow Manager", "Pipeline failed", {"error": str(e)})
            return self._mark_failed(task, "init", str(e))

        log_step("Workflow Manager", "Starting pipeline")
        logger.info("stage=create task_id=%s status=%s", task.task_id, task.status.value)

        try:
            task = self._planner.plan(task)
        except Exception as e:  # noqa: BLE001
            logger.exception("stage=plan task_id=%s error=%s", task.task_id, e)
            log_step("Workflow Manager", "Pipeline failed", {"error": str(e)})
            return self._mark_failed(task, "plan", str(e))
        task = self._ensure_status(task, TaskStatus.planned, "plan")
        if task.status == TaskStatus.failed:
            logger.warning("stage=plan task_id=%s status=failed error=%s", task.task_id, task.error)
            log_step("Workflow Manager", "Pipeline failed", {"error": task.error})
            return task
        logger.info("stage=plan task_id=%s status=%s", task.task_id, task.status.value)

        try:
            task = self._interview_agent.conduct_interview(task)
        except Exception as e:  # noqa: BLE001
            logger.exception("stage=interview task_id=%s error=%s", task.task_id, e)
            log_step("Workflow Manager", "Pipeline failed", {"error": str(e)})
            return self._mark_failed(task, "interview", str(e))
        task = self._ensure_status(task, TaskStatus.interviewing, "interview")
        if task.status == TaskStatus.failed:
            logger.warning("stage=interview task_id=%s status=failed error=%s", task.task_id, task.error)
            log_step("Workflow Manager", "Pipeline failed", {"error": task.error})
            return task
        logger.info("stage=interview task_id=%s status=%s", task.task_id, task.status.value)

        try:
            task = self._evaluation_agent.evaluate(task)
        except Exception as e:  # noqa: BLE001
            logger.exception("stage=evaluate task_id=%s error=%s", task.task_id, e)
            log_step("Workflow Manager", "Pipeline failed", {"error": str(e)})
            return self._mark_failed(task, "evaluate", str(e))
        task = self._ensure_status(task, TaskStatus.evaluated, "evaluate")
        if task.status == TaskStatus.failed:
            logger.warning("stage=evaluate task_id=%s status=failed error=%s", task.task_id, task.error)
            log_step("Workflow Manager", "Pipeline failed", {"error": task.error})
            return task
        logger.info("stage=evaluate task_id=%s status=%s", task.task_id, task.status.value)

        try:
            task = self._execution_agent.execute(task)
        except Exception as e:  # noqa: BLE001
            logger.exception("stage=execute task_id=%s error=%s", task.task_id, e)
            log_step("Workflow Manager", "Pipeline failed", {"error": str(e)})
            return self._mark_failed(task, "execute", str(e))
        task = self._ensure_status(task, TaskStatus.verified, "execute")
        if task.status == TaskStatus.failed:
            logger.warning("stage=execute task_id=%s status=failed error=%s", task.task_id, task.error)
            log_step("Workflow Manager", "Pipeline failed", {"error": task.error})
            return task
        logger.info("stage=execute task_id=%s status=%s", task.task_id, task.status.value)

        try:
            task = self._verification_agent.verify(task)
        except Exception as e:  # noqa: BLE001
            logger.exception("stage=verify task_id=%s error=%s", task.task_id, e)
            log_step("Workflow Manager", "Pipeline failed", {"error": str(e)})
            return self._mark_failed(task, "verify", str(e))
        if task.status == TaskStatus.failed:
            logger.warning("stage=verify task_id=%s status=failed error=%s", task.task_id, task.error)
            log_step("Workflow Manager", "Pipeline failed", {"error": task.error})
            return task
        logger.info("stage=verify task_id=%s status=%s", task.task_id, task.status.value)

        log_step("Workflow Manager", "Pipeline completed")
        return task
