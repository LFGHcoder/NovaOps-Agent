"""Interview agent: runs screening questions and collects candidate answers."""

from task import Task, TaskStatus
from log_utils import log_step

# Mock answers used when candidate_data["answers"] is not provided.
_DEFAULT_MOCK_ANSWERS = [
    "We used a versioned REST API with backward compatibility; we chose consistency over brevity in payloads.",
    "I start by reproducing locally, add logging or metrics, then narrow with binary search or bisect on deploys.",
    "Junction table for many-to-many; I use it when both sides can have multiple associations and I need extra attributes on the link.",
    "Idempotency key in the request, stored with outcome; we reject duplicates and return the same result for the same key.",
    "We timebox refactors, track tech debt in the backlog, and align with the team on a sustainable pace.",
]


class InterviewAgent:
    """
    Conducts the interview phase: asks interview_questions and collects answers.
    Uses candidate_data["answers"] if present; otherwise mocks answers.
    Controlled by WorkflowManager. No UI logic.
    """

    def __init__(self) -> None:
        pass

    def conduct_interview(self, task: Task) -> Task:
        """
        Simulate asking task.interview_questions and record answers.
        Answers come from task.candidate_data["answers"] or are mocked.
        Returns updated task with candidate_answers set and status = interviewing.
        """
        log_step("Interview Agent", "Conducting technical screening (Q&A)")
        questions = task.interview_questions or []
        provided = task.candidate_data.get("answers") if isinstance(task.candidate_data.get("answers"), list) else None

        if provided is not None and len(provided) >= len(questions):
            answers = list(provided[: len(questions)])
        else:
            # Mock one answer per question, reusing defaults if needed
            answers = []
            for i in range(len(questions)):
                answers.append(_DEFAULT_MOCK_ANSWERS[i % len(_DEFAULT_MOCK_ANSWERS)])
        log_step("Interview Agent", "Screening complete", {"answers_collected": len(answers), "questions_answered": len(questions)})

        return task.model_copy(
            update={
                "candidate_answers": answers,
                "status": TaskStatus.interviewing,
            },
            deep=True,
        )
