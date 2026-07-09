"""Compatibility entry point for the current four-dimension matching pipeline."""

from __future__ import annotations

from typing import Any

from .baseline_tfidf import calculate_tfidf_baseline
from .evidence_matcher import match_evidence_with_status
from .preprocess import split_jd_requirements, split_resume_sections
from .scoring import calculate_scores
from .skill_extractor import extract_skills


def calculate_match(resume_text: str, jd_text: str) -> dict[str, Any]:
    """Run the same BGE evidence-matching pipeline used by the Streamlit app."""
    resume_sections = split_resume_sections(resume_text)
    jd_requirements = split_jd_requirements(jd_text)
    skill_result = extract_skills(resume_text, jd_text)
    evidence_result = match_evidence_with_status(jd_requirements, resume_sections)
    scores = calculate_scores(
        resume_text,
        jd_text,
        skill_result,
        evidence_result["matches"],
        resume_sections,
    )
    baseline = calculate_tfidf_baseline(resume_text, jd_text)

    return {
        **scores,
        "matched_keywords": skill_result["matched_skills"],
        "missing_keywords": skill_result["missing_skills"],
        "keywords_by_category": skill_result["matched_skills_by_category"],
        "resume_keywords": skill_result["resume_skills"],
        "jd_keywords": skill_result["jd_skills"],
        "skill_result": skill_result,
        "evidence_matches": evidence_result["matches"],
        "semantic_model": evidence_result["semantic_model"],
        "tfidf_baseline": baseline,
    }
