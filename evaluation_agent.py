"""Evaluation agent: scores candidate answers and produces recommendation."""

import json
import re
from typing import Any, Callable

from task import Rubric, RubricCategory, Task, TaskStatus


def _mock_nova_lite_evaluate(prompt: str) -> str:
    """Mock Nova 2 Lite API call for evaluation. Returns structured JSON string."""
    return json.dumps({
        "overall_score": 78,
        "strengths": [
            "Strong API design awareness and trade-off reasoning",
            "Structured debugging approach",
            "Clear explanation of data modeling",
        ],
        "weaknesses": [
            "Idempotency example could be more concrete",
            "Tech debt answer was brief",
        ],
        "recommendation": "shortlist",
    })


def _mock_nova_lite_category(prompt: str, scoring_scale: int) -> str:
    """Mock Nova 2 Lite API call for one rubric category. Returns score and reasoning."""
    return json.dumps({
        "score": min(78 + hash(prompt) % 20, scoring_scale),
        "reasoning": "Candidate addressed the criteria with relevant examples.",
    })


def _mock_nova_lite_recommendation(prompt: str) -> str:
    """Mock Nova 2 Lite API call for recommendation. Returns shortlist or reject."""
    return json.dumps({"recommendation": "shortlist"})


def _parse_json_strict(raw: str) -> dict[str, Any]:
    """Parse string as JSON. Raises ValueError if malformed."""
    text = raw.strip()
    if text.startswith("```"):
        match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return json.loads(text)


def _strip_markdown_retry(text: str) -> str:
    """Strip markdown code fences for retry parse."""
    out = re.sub(r"^```\w*\s*", "", text)
    return re.sub(r"\s*```\s*$", "", out).strip()


def _parse_response(
    response: str,
    validator: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Parse LLM response as JSON. Retries once on malformed JSON."""
    last_error: Exception | None = None
    text = response.strip()
    for attempt in range(2):
        try:
            data = _parse_json_strict(text)
            validator(data)
            return data
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt == 0 and "```" in text:
                text = _strip_markdown_retry(text)
    raise last_error  # type: ignore[misc]


def _validate_evaluation_payload(data: dict[str, Any]) -> None:
    """Ensure payload has required keys and types. Raises ValueError if invalid."""
    if "overall_score" not in data:
        raise ValueError("overall_score is required")
    if not isinstance(data["overall_score"], int):
        raise ValueError("overall_score must be an int")
    if "strengths" not in data:
        raise ValueError("strengths is required")
    if not isinstance(data["strengths"], list):
        raise ValueError("strengths must be a list")
    if "weaknesses" not in data:
        raise ValueError("weaknesses is required")
    if not isinstance(data["weaknesses"], list):
        raise ValueError("weaknesses must be a list")
    if "recommendation" not in data:
        raise ValueError("recommendation is required")
    rec = data["recommendation"]
    if rec not in ("shortlist", "reject"):
        raise ValueError('recommendation must be "shortlist" or "reject"')


def _validate_category_score_payload(
    data: dict[str, Any],
    scoring_scale: int,
) -> None:
    """Ensure category score payload is valid. Raises ValueError if invalid."""
    if "score" not in data:
        raise ValueError("score is required")
    if not isinstance(data["score"], int):
        raise ValueError("score must be an int")
    s = data["score"]
    if s < 0 or s > scoring_scale:
        raise ValueError(f"score must be 0 to {scoring_scale}")
    if "reasoning" not in data:
        raise ValueError("reasoning is required")
    if not isinstance(data["reasoning"], str):
        raise ValueError("reasoning must be a str")


def _validate_recommendation_payload(data: dict[str, Any]) -> None:
    """Ensure recommendation payload is valid. Raises ValueError if invalid."""
    if "recommendation" not in data:
        raise ValueError("recommendation is required")
    if data["recommendation"] not in ("shortlist", "reject"):
        raise ValueError('recommendation must be "shortlist" or "reject"')


def _compute_weighted_score(
    category_scores: list[dict[str, Any]],
    scoring_scale: int,
) -> float:
    """
    Category scores are on 0–scoring_scale; normalize to 0–100 contribution.
    weighted_score = (category_score / 100) * category_weight  [with category_score normalized to 0–100]
    overall_score = sum(weighted_scores)  → 0–100 when weights sum to 100.
    """
    if scoring_scale <= 0:
        return 0.0
    # normalized contribution per category: (score/scale)*weight; weights sum to 100 → overall 0–100
    return sum((c["score"] / scoring_scale) * c["weight"] for c in category_scores)


def _score_to_recommendation(overall_score: float) -> str:
    """Recommendation from overall_score (0–100)."""
    if overall_score >= 75:
        return "Strong Hire"
    if overall_score >= 60:
        return "Consider"
    return "Reject"


class EvaluationAgent:
    """
    Scores candidate answers against required_skills and evaluation_criteria,
    or against task.rubric when present. Uses Nova 2 Lite (mock).
    Writes result to task.evaluation and sets status = evaluated.
    """

    def __init__(self) -> None:
        pass

    def _call_nova_lite(self, prompt: str) -> str:
        """Call Nova 2 Lite API. Mock implementation."""
        return _mock_nova_lite_evaluate(prompt)

    def _call_nova_lite_category(self, prompt: str, scoring_scale: int) -> str:
        """Call Nova 2 Lite for one rubric category. Mock implementation."""
        return _mock_nova_lite_category(prompt, scoring_scale)

    def _call_nova_lite_recommendation(self, prompt: str) -> str:
        """Call Nova 2 Lite for recommendation. Mock implementation."""
        return _mock_nova_lite_recommendation(prompt)

    def _build_prompt(self, task: Task) -> str:
        """Build prompt for evaluation using task context."""
        required_skills = task.evaluation.get("required_skills") or []
        criteria = task.evaluation.get("evaluation_criteria") or {}
        questions = task.interview_questions or []
        answers = task.candidate_answers or []

        return (
            "Evaluate the candidate's interview answers.\n\n"
            "Required skills to assess: " + json.dumps(required_skills) + "\n\n"
            "Evaluation criteria: " + json.dumps(criteria) + "\n\n"
            "Questions and answers:\n"
            + "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))
            + "\n\n"
            "Return a single JSON object with:\n"
            '- "overall_score": int (0-100)\n'
            '- "strengths": list of strings\n'
            '- "weaknesses": list of strings\n'
            '- "recommendation": "shortlist" or "reject"\n\n'
            "Respond with only the JSON object, no other text."
        )

    def _build_category_prompt(
        self,
        task: Task,
        category: RubricCategory,
        scoring_scale: int,
    ) -> str:
        """Build prompt for scoring one rubric category."""
        questions = task.interview_questions or []
        answers = task.candidate_answers or []
        qa_block = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))
        return (
            f"Score the candidate's answers for category: {category.name}\n"
            f"Criteria: {json.dumps(category.criteria)}\n"
            f"Scoring scale: 0 to {scoring_scale}\n\n"
            f"Questions and answers:\n{qa_block}\n\n"
            "Return a single JSON object with:\n"
            '- "score": int (0 to scoring_scale)\n'
            '- "reasoning": string\n\n'
            "Respond with only the JSON object, no other text."
        )

    def _evaluate_with_rubric(self, task: Task) -> dict[str, Any]:
        """Run rubric-based evaluation: per-category scores, weighted total, recommendation."""
        rubric = task.rubric
        if not rubric or not rubric.categories:
            raise ValueError("task.rubric must have categories")

        scoring_scale = rubric.scoring_scale
        total_weight = sum(c.weight for c in rubric.categories)
        if total_weight == 0:
            total_weight = 1

        category_scores: list[dict[str, Any]] = []

        for cat in rubric.categories:
            prompt = self._build_category_prompt(task, cat, scoring_scale)
            response = self._call_nova_lite_category(prompt, scoring_scale)
            validator = lambda d, sc=scoring_scale: _validate_category_score_payload(d, sc)
            data = _parse_response(response, validator)

            category_scores.append({
                "category": cat.name,
                "score": data["score"],
                "weight": cat.weight,
                "reasoning": data["reasoning"],
            })

        overall_score = _compute_weighted_score(category_scores, scoring_scale)
        recommendation = _score_to_recommendation(overall_score)

        return {
            "category_scores": category_scores,
            "overall_score": round(overall_score, 2),
            "recommendation": recommendation,
        }

    def _evaluate_fallback(self, task: Task) -> dict[str, Any]:
        """Existing AI-based evaluation when no rubric."""
        prompt = self._build_prompt(task)
        response = self._call_nova_lite(prompt)
        data = _parse_response(response, _validate_evaluation_payload)
        overall = data["overall_score"]
        return {
            "overall_score": overall,
            "strengths": list(data["strengths"]),
            "weaknesses": list(data["weaknesses"]),
            "recommendation": _score_to_recommendation(overall),
        }

    def evaluate(self, task: Task) -> Task:
        """
        If task.rubric exists: score by rubric categories, compute weighted score in Python,
        then recommendation. Else: fall back to existing AI evaluation.
        Merges result into task.evaluation and sets status = evaluated.
        """
        if task.rubric and task.rubric.categories:
            evaluation_result = self._evaluate_with_rubric(task)
        else:
            evaluation_result = self._evaluate_fallback(task)

        merged_evaluation = {
            **task.evaluation,
            **evaluation_result,
        }

        return task.model_copy(
            update={
                "evaluation": merged_evaluation,
                "status": TaskStatus.evaluated,
            },
            deep=True,
        )
