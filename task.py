"""Production-ready Pydantic Task system for AI HR Automation Agent."""

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Lifecycle status of a task."""

    created = "created"
    planned = "planned"
    interviewing = "interviewing"
    evaluated = "evaluated"
    executing = "executing"
    verified = "verified"
    complete = "completed"
    failed = "failed"


class Step(BaseModel):
    """A single step in a task plan."""

    step_id: int
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class RubricCategory(BaseModel):
    """One category in a recruiter rubric."""

    name: str
    weight: int
    criteria: list[str] = Field(default_factory=list)


class Rubric(BaseModel):
    """Custom recruiter rubric: categories and scoring scale."""

    categories: list[RubricCategory] = Field(default_factory=list)
    scoring_scale: int = 100

    def validate_rubric(self) -> bool:
        """Return True if sum of category weights equals scoring_scale. Else raise ValueError."""
        total = sum(c.weight for c in self.categories)
        if total != self.scoring_scale:
            raise ValueError(
                f"Rubric total weights ({total}) do not equal scoring_scale ({self.scoring_scale})"
            )
        return True


class Task(BaseModel):
    """HR automation task with plan, execution state, and lifecycle."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    job_description: str = ""
    candidate_data: dict[str, Any] = Field(default_factory=dict)
    interview_questions: list[str] = Field(default_factory=list)
    candidate_answers: list[str] = Field(default_factory=list)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    plan: list[Step] = Field(default_factory=list)
    current_step: int = 0
    execution_logs: list[dict[str, Any]] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.created
    error: Optional[str] = None
    rubric: Optional[Rubric] = None

    def mark_failed(self, message: str) -> None:
        """Set task status to failed and store the error message."""
        self.status = TaskStatus.failed
        self.error = message

    def advance_step(self) -> None:
        """Increment current_step by one."""
        self.current_step += 1
