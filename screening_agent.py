"""Screening agent: LLM-based resume evaluation against job description with rule-based fallback."""

import json
import re
from typing import Any

from task import Task
from log_utils import log_step

try:
    # Preferred import style
    from nova_client import call_nova
except ImportError:  # Fallback if direct import fails
    import nova_client  # type: ignore[import]

    def call_nova(prompt: str) -> str:
        return nova_client.call_nova(prompt)


SCREENING_PROMPT_TEMPLATE = """You are an AI recruiter.

Evaluate the candidate resume against the job description.

IMPORTANT:
Return ONLY valid JSON in the exact format below.
Do not include explanations, markdown, or extra text.

{{
  "score": number between 0 and 100,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "recommendation": "shortlist" | "review" | "reject"
}}

Resume:
{resume}

Job Description:
{job_description}
"""


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
    candidate_name = "Candidate"
    if resume and resume.strip():
        lines = resume.strip().split("\n")
        if lines and lines[0].strip():
            candidate_name = lines[0].strip()
    return {
        "candidate_name": candidate_name,
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
    # Escape braces in user content so .format() doesn't treat them as placeholders
    resume_safe = (resume or "").replace("{", "{{").replace("}", "}}")
    job_description_safe = (job_description or "").replace("{", "{{").replace("}", "}}")

    prompt = SCREENING_PROMPT_TEMPLATE.format(
        job_description=job_description_safe,
        resume=resume_safe,
    )

    try:
        print("Calling Nova...")

        response = call_nova(prompt)

        print("Nova raw response:", response)

        data = _parse_json_safe(response)

        if data is None:
            print("Nova response was not valid JSON")
        elif not _validate_screening_payload(data):
            print("Nova JSON failed validation:", data)
        else:
            return (
                {
                    "score": round(float(data["score"]), 2),
                    "strengths": list(data["strengths"]) if isinstance(data["strengths"], list) else [],
                    "weaknesses": list(data["weaknesses"]) if isinstance(data["weaknesses"], list) else [],
                    "recommendation": str(data["recommendation"]),
                },
                True,
            )

    except Exception as e:
        print("Nova error:", e)

    # fallback
    print("Using rule-based fallback scoring")
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

        # Extract candidate name from first line of resume for consistency
        if resume and resume.strip():
            lines = resume.strip().split("\n")
            if lines and lines[0].strip():
                result["candidate_name"] = lines[0].strip()

        log_step(
            "Screening Agent",
            "Resume evaluation complete",
            {
                "score": result["score"],
                "recommendation": result["recommendation"],
                "source": "nova" if used_llm else "rule_based",
            },
        )

        evaluation = dict(task.evaluation or {})
        evaluation["screening"] = result
        candidate_name = result.get("candidate_name") or "Candidate"
        if isinstance(task.candidate_data, dict):
            candidate_data = dict(task.candidate_data)
            candidate_data["name"] = candidate_name
            return task.model_copy(
                update={"evaluation": evaluation, "candidate_data": candidate_data},
                deep=True,
            )
        return task.model_copy(update={"evaluation": evaluation}, deep=True)
