"""Workflow manager: orchestrates the HR automation pipeline."""

import logging
from typing import Any, Optional, Protocol

from task import Task, TaskStatus, Rubric
from planner_agent import PlannerAgent
from interview_agent import InterviewAgent
from evaluation_agent import EvaluationAgent
from mock_execution_agent import MockExecutionAgent
from verification_agent import VerificationAgent


logger = logging.getLogger(__name__)


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

    def run(
        self,
        job_description: str,
        candidate_data: dict[str, Any],
        rubric: Optional[dict[str, Any]] = None,
    ) -> Task:
        """
        Run the full pipeline. On any failure the task is marked failed and returned.
        """
        task = Task(
            job_description=job_description,
            candidate_data=candidate_data,
            rubric=Rubric.model_validate(rubric) if rubric else None,
        )

        try:
            logger.info("stage=create task_id=%s status=%s", task.task_id, task.status.value)

            task = self._planner.plan(task)
            task = self._ensure_status(task, TaskStatus.planned, "plan")
            if task.status == TaskStatus.failed:
                logger.warning("stage=plan task_id=%s status=failed error=%s", task.task_id, task.error)
                return task
            logger.info("stage=plan task_id=%s status=%s", task.task_id, task.status.value)

            task = self._interview_agent.conduct_interview(task)
            task = self._ensure_status(task, TaskStatus.interviewing, "interview")
            if task.status == TaskStatus.failed:
                logger.warning("stage=interview task_id=%s status=failed error=%s", task.task_id, task.error)
                return task
            logger.info("stage=interview task_id=%s status=%s", task.task_id, task.status.value)

            task = self._evaluation_agent.evaluate(task)
            task = self._ensure_status(task, TaskStatus.evaluated, "evaluate")
            if task.status == TaskStatus.failed:
                logger.warning("stage=evaluate task_id=%s status=failed error=%s", task.task_id, task.error)
                return task
            logger.info("stage=evaluate task_id=%s status=%s", task.task_id, task.status.value)

            task = self._execution_agent.execute(task)
            task = self._ensure_status(task, TaskStatus.verified, "execute")
            if task.status == TaskStatus.failed:
                logger.warning("stage=execute task_id=%s status=failed error=%s", task.task_id, task.error)
                return task
            logger.info("stage=execute task_id=%s status=%s", task.task_id, task.status.value)

            task = self._verification_agent.verify(task)
            if task.status == TaskStatus.failed:
                logger.warning("stage=verify task_id=%s status=failed error=%s", task.task_id, task.error)
                return task
            logger.info("stage=verify task_id=%s status=%s", task.task_id, task.status.value)

            return task

        except Exception as e:  # noqa: BLE001
            logger.exception("stage=workflow task_id=%s error=%s", task.task_id, e)
            return task.model_copy(
                update={
                    "status": TaskStatus.failed,
                    "error": str(e),
                },
                deep=True,
            )
