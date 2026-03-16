"""Evaluation agent: scores candidate answers and produces recommendation."""
import random
import json
import re
from typing import Any, Callable

from task import Rubric, RubricCategory, Task, TaskStatus
from log_utils import log_step
from nova_client import call_nova


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
    """Mock Nova category scoring."""
    score = random.randint(6, scoring_scale)

    return json.dumps({
        "score": score,
        "reasoning": "Candidate demonstrated relevant experience and problem solving ability."
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
    if rec not in ("shortlist", "review", "reject"):
        raise ValueError('recommendation must be "shortlist", "review", or "reject"')


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
    if data["recommendation"] not in ("shortlist", "review", "reject"):
        raise ValueError('recommendation must be "shortlist", "review", or "reject"')


def _compute_weighted_score(
    category_scores: list[dict[str, Any]],
    scoring_scale: int,
) -> float:
    """
    Category score is 0–scoring_scale per category; weight is percentage (0–100).
    total = sum(score * (weight/100)); overall_score = round(total * 10, 2) → scale 0–100.
    """
    total = 0.0
    for c in category_scores:
        weight = c["weight"] / 100.0
        total += c["score"] * weight
    return round(total * 10, 2)


def _score_to_recommendation(overall_score: float) -> str:
    """Only evaluation_agent sets final recommendation. Score 0–100."""
    if overall_score >= 80:
        return "Strong Hire"
    if overall_score >= 65:
        return "Shortlist"
    if overall_score >= 50:
        return "Review"
    return "Reject"


def _evaluate_with_nova(task: Task) -> dict[str, Any]:
    """
    Use Amazon Nova to evaluate candidate answers.

    Expected JSON:
    {
      "overall_score": 0-100,
      "strengths": [...],
      "weaknesses": [...],
      "recommendation": "shortlist" | "review" | "reject"
    }
    """
    required_skills = task.evaluation.get("required_skills") or []
    criteria = task.evaluation.get("evaluation_criteria") or {}
    questions = task.interview_questions or []
    answers = task.candidate_answers or []

    qa_block = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))

    prompt = (
        "You are an expert AI technical interviewer.\n\n"
        "Evaluate the candidate's interview answers against the required skills and evaluation criteria.\n\n"
        "Return ONLY valid JSON with the following fields:\n"
        '- \"overall_score\": number between 0 and 100\n'
        '- \"strengths\": array of strings\n'
        '- \"weaknesses\": array of strings\n'
        '- \"recommendation\": one of \"shortlist\", \"review\", \"reject\"\n\n'
        "Required skills:\n"
        f"{json.dumps(required_skills, indent=2)}\n\n"
        "Evaluation criteria:\n"
        f"{json.dumps(criteria, indent=2)}\n\n"
        "Questions and answers:\n"
        f"{qa_block}\n"
    )

    print("Evaluation Agent calling Nova...")
    response = call_nova(prompt)
    print("Evaluation Nova response:", response)

    data = _parse_json_strict(response)
    _validate_evaluation_payload(data)

    score = float(data["overall_score"])
    return {
        "overall_score": round(score, 2),
        "strengths": list(data["strengths"]),
        "weaknesses": list(data["weaknesses"]),
        # Normalize final recommendation based on score bands
        "recommendation": _score_to_recommendation(score),
    }


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
            "overall_score": overall_score,
            "recommendation": recommendation,
        }

    def _evaluate_fallback(self, task: Task) -> dict[str, Any]:
        """Simple fallback when Nova is unavailable and no rubric is present."""
        # Conservative baseline: no strengths/weaknesses, reject
        return {
            "overall_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "recommendation": "Reject",
        }

    def evaluate(self, task: Task) -> Task:
        """
        If task.rubric exists: score by rubric categories, compute weighted score in Python,
        then recommendation. Else: fall back to existing AI evaluation.
        Merges result into task.evaluation and sets status = evaluated.
        """
        log_step("Evaluation Agent", "Calculating weighted candidate score")
        try:
            # Primary path: use Nova for holistic evaluation
            evaluation_result = _evaluate_with_nova(task)
        except Exception as e:  # noqa: BLE001
            print("Evaluation Nova error:", e)
            # Fallback: use existing rubric-based scoring when available
            if task.rubric and task.rubric.categories:
                evaluation_result = self._evaluate_with_rubric(task)
            else:
                evaluation_result = self._evaluate_fallback(task)
        log_step(
            "Evaluation Agent",
            "Score computed",
            {
                "overall_score": evaluation_result.get("overall_score"),
                "recommendation": evaluation_result.get("recommendation"),
            },
        )

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
