"""Skill entity extraction based on the project skill dictionary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .keyword_extractor import (
    _difference_keywords,
    _extract_dict_keywords,
    _flatten_keywords,
    _intersect_keywords,
    load_skill_dict,
)


def extract_skills(
    resume_text: str,
    jd_text: str,
    skill_dict_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract categorized skill entities from resume and JD."""
    skill_dict = load_skill_dict(skill_dict_path)
    resume_by_category = _extract_dict_keywords(resume_text, skill_dict)
    jd_by_category = _extract_dict_keywords(jd_text, skill_dict)

    matched_by_category: dict[str, list[str]] = {}
    missing_by_category: dict[str, list[str]] = {}
    for category in skill_dict:
        matched_by_category[category] = _intersect_keywords(
            jd_by_category.get(category, []),
            resume_by_category.get(category, []),
        )
        missing_by_category[category] = _difference_keywords(
            jd_by_category.get(category, []),
            resume_by_category.get(category, []),
        )

    resume_skills = _flatten_keywords(resume_by_category)
    jd_skills = _flatten_keywords(jd_by_category)
    matched_skills = _flatten_keywords(matched_by_category)
    missing_skills = _flatten_keywords(missing_by_category)

    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_skills_by_category": resume_by_category,
        "jd_skills_by_category": jd_by_category,
        "matched_skills_by_category": matched_by_category,
        "missing_skills_by_category": missing_by_category,
    }

