import json
from pathlib import Path

from workflow_manager import WorkflowManager, result_to_json


# Select which resume to evaluate (file path relative to project root).
resume_file = "resumes/strong_candidate.txt"

JOB_DESCRIPTION = "Hiring backend Python engineer with system design experience"

RUBRIC = {
    "categories": [
        {"name": "Python", "weight": 40, "criteria": ["OOP", "Async", "Clean code"]},
        {"name": "System Design", "weight": 40, "criteria": ["Scalability", "Trade-offs"]},
        {"name": "Communication", "weight": 20, "criteria": ["Clarity", "Structure"]},
    ],
    "scoring_scale": 10,
}

DEFAULT_ANSWERS = [
    "Designed REST API with pagination and caching trade-offs.",
    "Used structured logging and incremental isolation to debug.",
    "Used junction table with proper indexing for many-to-many.",
    "Implemented idempotency keys with Redis and unique constraints.",
    "Balanced speed with code reviews and technical debt tracking.",
]


def load_resume(path: str) -> str:
    """Load resume text from a file. Returns empty string if file not found."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / path
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


if __name__ == "__main__":
    resume_text = load_resume(resume_file)

    candidate_data = {
        "name": None,
        "resume": resume_text,
        "answers": DEFAULT_ANSWERS,
    }

    workflow = WorkflowManager()
    task = workflow.run(
        job_description=JOB_DESCRIPTION,
        candidate_data=candidate_data,
        rubric=RUBRIC,
        resume_file=resume_file,
    )

    result = result_to_json(task)
    print("\nFinal Result:")
    print(json.dumps(result, indent=4))
