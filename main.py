import json
from workflow_manager import WorkflowManager


if __name__ == "__main__":
    workflow = WorkflowManager()

    task = workflow.run(
        job_description="Hiring backend Python engineer with system design experience",
        candidate_data={
            "name": "John Doe",
            "answers": [
                "Designed REST API with pagination and caching trade-offs.",
                "Used structured logging and incremental isolation to debug.",
                "Used junction table with proper indexing for many-to-many.",
                "Implemented idempotency keys with Redis and unique constraints.",
                "Balanced speed with code reviews and technical debt tracking."
            ]
        },
        rubric={
            "categories": [
                {
                    "name": "Python",
                    "weight": 40,
                    "criteria": ["OOP", "Async", "Clean code"]
                },
                {
                    "name": "System Design",
                    "weight": 40,
                    "criteria": ["Scalability", "Trade-offs"]
                },
                {
                    "name": "Communication",
                    "weight": 20,
                    "criteria": ["Clarity", "Structure"]
                }
            ],
            "scoring_scale": 10
        }
    )

    # ---- CLEAN DEMO OUTPUT ----
    print("\n" + "=" * 50)
    print("AI HR AUTOMATION RESULT")
    print("=" * 50)

    result = {
        "candidate": task.candidate_data.get("name"),
        "overall_score": task.evaluation.get("overall_score"),
        "recommendation": task.evaluation.get("recommendation"),
        "status": task.status.value
    }

    print(json.dumps(result, indent=4))

    # Optional: show category breakdown
    if "category_scores" in task.evaluation:
        print("\nCategory Breakdown:")
        print(json.dumps(task.evaluation["category_scores"], indent=4))

    print("\n" + "=" * 50)  