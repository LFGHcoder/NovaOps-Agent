"""Rubric agent: generates custom recruiter rubric from job description."""

import json
import re
from typing import Any


def _mock_nova_lite_rubric(prompt: str) -> str:
    """Mock Nova 2 Lite API call. Returns structured rubric JSON."""
    return json.dumps({
        "categories": [
            {"name": "Technical depth", "weight": 30, "criteria": ["Concept clarity", "Trade-off reasoning", "Relevant examples"]},
            {"name": "Problem solving", "weight": 25, "criteria": ["Structured approach", "Debugging method", "Edge cases"]},
            {"name": "Communication", "weight": 20, "criteria": ["Clarity", "Conciseness", "Structure"]},
            {"name": "Experience", "weight": 15, "criteria": ["Past projects", "Outcomes", "Learnings"]},
            {"name": "Culture fit", "weight": 10, "criteria": ["Values alignment", "Collaboration", "Growth mindset"]},
        ],
        "scoring_scale": 10,
    })


def _parse_json_strict(raw: str) -> dict[str, Any]:
    """Parse string as JSON. Raises ValueError if malformed."""
    text = raw.strip()
    if text.startswith("```"):
        match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return json.loads(text)


def _validate_rubric_payload(data: dict[str, Any]) -> None:
    """Ensure rubric has 3–5 categories, weights sum to 100, scoring_scale. Raises ValueError if invalid."""
    if "categories" not in data:
        raise ValueError("categories is required")
    cats = data["categories"]
    if not isinstance(cats, list):
        raise ValueError("categories must be a list")
    if len(cats) < 3 or len(cats) > 5:
        raise ValueError("categories must contain 3 to 5 items")
    total_weight = 0
    for i, item in enumerate(cats):
        if not isinstance(item, dict):
            raise ValueError(f"categories[{i}] must be an object")
        for key in ("name", "weight", "criteria"):
            if key not in item:
                raise ValueError(f"categories[{i}] missing required key: {key}")
        if not isinstance(item["name"], str):
            raise ValueError(f"categories[{i}].name must be a string")
        if not isinstance(item["weight"], int):
            raise ValueError(f"categories[{i}].weight must be an int")
        if not isinstance(item["criteria"], list):
            raise ValueError(f"categories[{i}].criteria must be a list")
        if not all(isinstance(c, str) for c in item["criteria"]):
            raise ValueError(f"categories[{i}].criteria must be list of strings")
        total_weight += item["weight"]
    if total_weight != 100:
        raise ValueError(f"category weights must sum to 100, got {total_weight}")
    if "scoring_scale" not in data:
        raise ValueError("scoring_scale is required")
    if not isinstance(data["scoring_scale"], int):
        raise ValueError("scoring_scale must be an int")


def _strip_markdown_retry(text: str) -> str:
    """Strip markdown code fences for retry parse."""
    out = re.sub(r"^```\w*\s*", "", text)
    return re.sub(r"\s*```\s*$", "", out).strip()


class RubricAgent:
    """
    Generates a structured rubric from a job description.
    Uses Nova 2 Lite (mock). Returns dict compatible with Task.rubric.
    """

    def __init__(self) -> None:
        pass

    def _call_nova_lite(self, prompt: str) -> str:
        """Call Nova 2 Lite API. Mock implementation."""
        return _mock_nova_lite_rubric(prompt)

    def _build_prompt(self, job_description: str) -> str:
        """Build prompt for rubric generation."""
        return (
            "Extract 3 to 5 skill categories from the job description. "
            "Assign each category a weight (integers); weights must sum to 100. "
            "Add criteria (list of strings) per category.\n\n"
            "Return a single JSON object with:\n"
            '- "categories": list of objects, each with "name" (str), "weight" (int), "criteria" (list[str])\n'
            '- "scoring_scale": 10\n\n'
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
                _validate_rubric_payload(data)
                return data
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt == 0 and "```" in text:
                    text = _strip_markdown_retry(text)
        raise last_error  # type: ignore[misc]

    def generate_rubric(self, job_description: str) -> dict[str, Any]:
        """
        Generate a rubric with 3–5 categories, weights summing to 100, scoring_scale 10.
        Returns dict suitable for Task.rubric (Rubric.model_validate(result)).
        """
        prompt = self._build_prompt(job_description)
        response = self._call_nova_lite(prompt)
        data = self._parse_response(response)
        return {
            "categories": [
                {
                    "name": c["name"],
                    "weight": c["weight"],
                    "criteria": list(c["criteria"]),
                }
                for c in data["categories"]
            ],
            "scoring_scale": data["scoring_scale"],
        }
