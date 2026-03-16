"""Screening agent: LLM-based resume evaluation against job description with rule-based fallback."""

import json
import re
from typing import Any

from task import Task
from log_utils import log_step


SCREENING_PROMPT = """You are an expert AI recruiter.
Evaluate the candidate resume against the job description.

Return only valid JSON with:
- candidate_name (string)
- score (number 0-100)
- strengths (array of strings)
- weaknesses (array of strings)
- recommendation (one of: shortlist, review, reject)

Job description:
{job_description}

Resume:
{resume}

Return only the JSON object, no other text."""


def _mock_llm_screening(prompt: str) -> str:
    """Mock LLM call for resume screening. Replace with real Nova/API when available."""
    return json.dumps({
        "candidate_name": "Candidate",
        "score": 78,
        "strengths": ["Relevant experience", "Clear formatting", "Matching skills"],
        "weaknesses": ["Limited leadership examples"],
        "recommendation": "review",
    })


def _parse_json_safe(raw: str) -> dict[str, Any] | None:
    """Parse JSON, strip markdown if present. Return None on failure."""
    text = raw.strip()
    if text.startswith("```"):
        match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validate_screening_payload(data: dict[str, Any]) -> bool:
    """Return True if payload has required keys and valid types."""
    if not isinstance(data.get("candidate_name"), str):
        return False
    if "score" not in data or not isinstance(data["score"], (int, float)):
        return False
    s = float(data["score"])
    if s < 0 or s > 100:
        return False
    if not isinstance(data.get("strengths"), list):
        return False
    if not isinstance(data.get("weaknesses"), list):
        return False
    rec = data.get("recommendation")
    if rec not in ("shortlist", "review", "reject"):
        return False
    return True


def _rule_based_screening(resume: str, job_description: str) -> dict[str, Any]:
    """
    Fallback: simple keyword-based scoring so the pipeline never breaks.
    Returns the same structure as LLM output.
    """
    jd_lower = (job_description or "").lower()
    resume_lower = (resume or "").lower()
    # Extract likely skills/terms from JD (simple: split and take words 4+ chars)
    jd_terms = set(w for w in re.findall(r"[a-z0-9]+", jd_lower) if len(w) >= 4)
    jd_terms.discard("description")
    jd_terms.discard("experience")
    jd_terms.discard("candidate")
    if not jd_terms:
        jd_terms = {"python", "experience", "team", "skills"}
    matches = sum(1 for t in jd_terms if t in resume_lower)
    total = len(jd_terms)
    score = min(100, round((matches / total * 100) if total else 50))
    if score >= 75:
        recommendation = "shortlist"
    elif score >= 50:
        recommendation = "review"
    else:
        recommendation = "reject"
    matched = [t for t in list(jd_terms)[:5] if t in resume_lower]
    missing = [t for t in list(jd_terms)[:5] if t not in resume_lower][:3]
    return {
        "candidate_name": "Candidate",
        "score": score,
        "strengths": [f"Matches: {', '.join(matched)}"] if matched else ["Resume provided"],
        "weaknesses": [f"Missing or weak: {', '.join(missing)}"] if missing else ["Limited keyword match"],
        "recommendation": recommendation,
    }


def evaluate_with_llm(resume: str, job_description: str) -> dict[str, Any]:
    """
    Evaluate resume against job description using LLM.
    On LLM or parse failure, falls back to rule-based scoring.
    Always returns valid dict: candidate_name, score (0-100), strengths, weaknesses, recommendation (shortlist|review|reject).
    """
    result, _ = _evaluate_with_llm(resume, job_description)
    return result


def _evaluate_with_llm(resume: str, job_description: str) -> tuple[dict[str, Any], bool]:
    """Internal: returns (result, used_llm)."""
    try:
        prompt = SCREENING_PROMPT.format(
            job_description=job_description or "",
            resume=resume or "",
        )
        response = _mock_llm_screening(prompt)
        data = _parse_json_safe(response)
        if data is not None and _validate_screening_payload(data):
            return (
                {
                    "candidate_name": str(data["candidate_name"]),
                    "score": round(float(data["score"]), 2),
                    "strengths": list(data["strengths"]) if isinstance(data["strengths"], list) else [],
                    "weaknesses": list(data["weaknesses"]) if isinstance(data["weaknesses"], list) else [],
                    "recommendation": str(data["recommendation"]),
                },
                True,
            )
    except Exception:  # noqa: BLE001
        pass
    return (_rule_based_screening(resume, job_description), False)


class ScreeningAgent:
    """
    Screens candidate resume against job description via LLM.
    Falls back to rule-based scoring if LLM fails. Always returns valid structure.
    """

    def __init__(self) -> None:
        pass

    def screen(self, task: Task) -> Task:
        """
        Run resume screening: evaluate_with_llm(resume, job_description).
        Store result in task.evaluation["screening"]. Log with log_utils. Status unchanged.
        """
        log_step("Screening Agent", "Evaluating resume against job description")
        resume = ""
        if isinstance(task.candidate_data, dict):
            resume = task.candidate_data.get("resume") or task.candidate_data.get("resume_text") or ""
            if not isinstance(resume, str):
                resume = str(resume) if resume else ""
        job_description = task.job_description or ""

        result, used_llm = _evaluate_with_llm(resume, job_description)

        log_step(
            "Screening Agent",
            "Resume evaluation complete",
            {
                "score": result["score"],
                "recommendation": result["recommendation"],
                "source": "llm" if used_llm else "rule_based",
            },
        )

        evaluation = dict(task.evaluation or {})
        evaluation["screening"] = result
        if result.get("candidate_name") and isinstance(task.candidate_data, dict):
            candidate_data = dict(task.candidate_data)
            if not candidate_data.get("name"):
                candidate_data["name"] = result["candidate_name"]
            return task.model_copy(
                update={"evaluation": evaluation, "candidate_data": candidate_data},
                deep=True,
            )
        return task.model_copy(update={"evaluation": evaluation}, deep=True)
