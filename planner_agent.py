"""Planner agent: converts job description into structured interview plan."""

import json
import re
from typing import Any

from task import Step, Task, TaskStatus


def _mock_nova_lite(prompt: str) -> str:
    """Mock Nova 2 Lite API call. Returns structured JSON string."""
    # Simulated response with required shape for parsing
    return json.dumps({
        "required_skills": ["Python", "REST APIs", "SQL", "Problem solving", "Communication"],
        "interview_questions": [
            "Describe a time you designed or improved an API. What trade-offs did you consider?",
            "How do you approach debugging a production issue with limited logs?",
            "Explain how you would model a many-to-many relationship and when to use a junction table.",
            "Walk through how you would implement idempotency for a critical background job.",
            "How do you balance speed of delivery with code quality and technical debt?",
        ],
        "evaluation_criteria": {
            "technical_depth": "Understanding of concepts and trade-offs",
            "practical_experience": "Relevant examples and outcomes",
            "communication": "Clarity and structure of answers",
            "problem_solving": "Approach and reasoning",
        },
        "plan": [
            {"step_id": 1, "action": "extract_skills", "parameters": {"source": "job_description"}, "reasoning": "Identify required skills from JD."},
            {"step_id": 2, "action": "prepare_screening_questions", "parameters": {"count": 5}, "reasoning": "Generate technical screening questions."},
            {"step_id": 3, "action": "define_criteria", "parameters": {}, "reasoning": "Set evaluation criteria for answers."},
            {"step_id": 4, "action": "conduct_screening", "parameters": {"questions": 5}, "reasoning": "Run technical screening with candidate."},
            {"step_id": 5, "action": "evaluate_responses", "parameters": {}, "reasoning": "Score answers against criteria."},
        ],
    })


def _parse_json_strict(raw: str) -> dict[str, Any]:
    """Parse string as JSON. Raises ValueError if malformed."""
    # Strip markdown code blocks if present
    text = raw.strip()
    if text.startswith("```"):
        match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return json.loads(text)


def _validate_plan_payload(data: dict[str, Any]) -> None:
    """Ensure payload has required keys and types. Raises ValueError if invalid."""
    if not isinstance(data.get("interview_questions"), list):
        raise ValueError("interview_questions must be a list")
    questions = data["interview_questions"]
    if len(questions) != 5:
        raise ValueError("interview_questions must contain exactly 5 questions")
    if not all(isinstance(q, str) for q in questions):
        raise ValueError("All interview_questions must be strings")
    if not isinstance(data.get("plan"), list):
        raise ValueError("plan must be a list")
    for i, item in enumerate(data["plan"]):
        if not isinstance(item, dict):
            raise ValueError(f"plan[{i}] must be an object")
        for key in ("step_id", "action", "parameters", "reasoning"):
            if key not in item:
                raise ValueError(f"plan[{i}] missing required key: {key}")
        if not isinstance(item["parameters"], dict):
            raise ValueError(f"plan[{i}].parameters must be a dict")


class PlannerAgent:
    """
    Converts job_description into a structured interview plan.
    Uses Nova 2 Lite (mock). Returns updated Task with plan and questions.
    """

    def __init__(self) -> None:
        pass

    def _call_nova_lite(self, prompt: str) -> str:
        """Call Nova 2 Lite API. Mock implementation."""
        return _mock_nova_lite(prompt)

    def _build_prompt(self, job_description: str) -> str:
        """Build prompt for plan generation."""
        return (
            "Convert the following job description into a structured interview plan.\n\n"
            "Return a single JSON object with exactly these keys:\n"
            '- "required_skills": list of strings (skills to assess)\n'
            '- "interview_questions": list of exactly 5 strings (technical screening questions)\n'
            '- "evaluation_criteria": object (criteria name -> description)\n'
            '- "plan": list of step objects, each with "step_id" (int), "action" (str), '
            '"parameters" (object), "reasoning" (str)\n\n'
            "Job description:\n"
            f"{job_description}\n\n"
            "Respond with only the JSON object, no other text."
        )

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response as JSON. Retries once on malformed JSON."""
        last_error: Exception | None = None
        text = response.strip()
        for attempt in range(2):
            try:
                data = _parse_json_strict(text)
                _validate_plan_payload(data)
                return data
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                # Retry once: strip markdown more aggressively
                if attempt == 0 and "```" in text:
                    text = re.sub(r"^```\w*\s*", "", text)
                    text = re.sub(r"\s*```\s*$", "", text).strip()
        raise last_error  # type: ignore[misc]

    def plan(self, task: Task) -> Task:
        """
        Produce interview plan from task.job_description.
        Fills interview_questions, plan (Step objects), and sets status to planned.
        """
        prompt = self._build_prompt(task.job_description)
        response = self._call_nova_lite(prompt)
        data = self._parse_response(response)

        steps = [Step(**s) for s in data["plan"]]
        interview_questions = list(data["interview_questions"])

        return task.model_copy(
            update={
                "interview_questions": interview_questions,
                "plan": steps,
                "status": TaskStatus.planned,
                "evaluation": {
                    **task.evaluation,
                    "required_skills": data.get("required_skills", []),
                    "evaluation_criteria": data.get("evaluation_criteria", {}),
                },
            },
            deep=True,
        )
