import json
from workflow_manager import WorkflowManager, result_to_json


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

    # Consistent JSON result (includes error when failed, category_scores when present)
    result = result_to_json(task)
    print("\nFinal Result:")
    print(json.dumps(result, indent=4))  