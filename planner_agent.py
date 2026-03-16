"""Planner agent: converts job description into structured interview plan."""

import json
import re
from typing import Any

from task import Step, Task, TaskStatus
from log_utils import log_step
from nova_client import call_nova


PLANNER_PROMPT_TEMPLATE = """
You are an AI recruiting workflow planner.

Analyze the job description and generate a hiring workflow plan.

Return ONLY JSON in this format:

{{
  "required_skills": ["skill1", "skill2"],
  "interview_questions": ["q1", "q2", "q3", "q4", "q5"],
  "evaluation_criteria": {{
    "Python": 40,
    "System Design": 40,
    "Communication": 20
  }},
  "plan": [
    {{"step_id": 1, "action": "extract_skills", "parameters": {{}}, "reasoning": "Identify required skills"}},
    {{"step_id": 2, "action": "prepare_screening_questions", "parameters": {{"count": 5}}, "reasoning": "Generate interview questions"}},
    {{"step_id": 3, "action": "define_criteria", "parameters": {{}}, "reasoning": "Define evaluation criteria"}},
    {{"step_id": 4, "action": "conduct_screening", "parameters": {{"questions": 5}}, "reasoning": "Run screening"}},
    {{"step_id": 5, "action": "evaluate_responses", "parameters": {{}}, "reasoning": "Score candidate answers"}}
  ]
}}

Job Description:
{job_description}

Respond ONLY with JSON.
"""


def _parse_json_strict(raw: str) -> dict[str, Any]:
    """Parse string as JSON. Raises ValueError if malformed."""
    text = raw.strip()

    if text.startswith("```"):
        match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    return json.loads(text)


def _validate_plan_payload(data: dict[str, Any]) -> None:
    """Ensure payload has required keys and types."""

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
    Uses Amazon Nova 2 Lite.
    """

    def __init__(self) -> None:
        pass

    def _build_prompt(self, job_description: str) -> str:
        """Build Nova prompt."""
        return PLANNER_PROMPT_TEMPLATE.format(
            job_description=job_description or ""
        )

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse Nova response safely."""
        last_error: Exception | None = None
        text = response.strip()

        for attempt in range(2):
            try:
                data = _parse_json_strict(text)
                _validate_plan_payload(data)
                return data
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e

                if attempt == 0 and "```" in text:
                    text = re.sub(r"^```\w*\s*", "", text)
                    text = re.sub(r"\s*```\s*$", "", text).strip()

        raise last_error  # type: ignore

    def plan(self, task: Task) -> Task:
        """Generate interview plan using Nova."""

        log_step("Planner Agent", "Generating ATS workflow plan")

        prompt = self._build_prompt(task.job_description)

        print("Planner calling Nova...")

        response = call_nova(prompt)

        print("Planner Nova response:", response)

        data = self._parse_response(response)

        steps = [Step(**s) for s in data["plan"]]

        interview_questions = list(data["interview_questions"])

        log_step(
            "Planner Agent",
            "Plan generated",
            {
                "questions": len(interview_questions),
                "plan_steps": len(steps),
                "matched_skills": len(data.get("required_skills", [])),
            },
        )

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