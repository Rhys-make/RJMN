"""Four-dimension scoring for BGE-based resume-JD matching."""

from __future__ import annotations

from typing import Any

from .evidence_matcher import semantic_similarity_pair
from .preprocess import clean_text


PROJECT_TERMS = [
    "项目",
    "实习",
    "开发",
    "系统",
    "平台",
    "接口",
    "数据处理",
    "模型",
    "部署",
    "测试",
    "优化",
    "自动化",
    "业务",
    "需求分析",
    "文档",
    "数据清洗",
    "文本处理",
]


def _clip_score(score: float) -> float:
    return round(max(0.0, min(100.0, float(score))), 2)


def _ratio_score(required_items: list[str], matched_items: list[str], default_score: float = 60.0) -> float:
    if not required_items:
        return default_score
    return _clip_score(len(matched_items) / len(required_items) * 100)


def _match_level(total_score: float) -> str:
    if total_score >= 85:
        return "高度匹配"
    if total_score >= 70:
        return "较高匹配"
    if total_score >= 50:
        return "一般匹配"
    return "匹配度较低"


def _best_evidence_average(evidence_matches: list[dict[str, Any]]) -> float:
    scores = [float(item.get("best_similarity", 0.0)) for item in evidence_matches]
    if not scores:
        return 0.0
    return _clip_score(sum(scores) / len(scores) * 100)


def _project_experience_score(resume_sections: dict[str, list[str]], jd_text: str) -> float:
    project_text = "\n".join(resume_sections.get("项目经历", []) + resume_sections.get("实习经历", []))
    if not project_text.strip():
        return 40.0

    similarity, _ = semantic_similarity_pair(project_text, jd_text)
    project_clean = clean_text(project_text).replace(" ", "")
    jd_clean = clean_text(jd_text).replace(" ", "")
    jd_terms = [term for term in PROJECT_TERMS if clean_text(term).replace(" ", "") in jd_clean]
    if jd_terms:
        hit_terms = [term for term in jd_terms if clean_text(term).replace(" ", "") in project_clean]
        term_score = len(hit_terms) / len(jd_terms) * 100
    else:
        term_score = 60.0
    return _clip_score(max(similarity * 100, term_score))


def calculate_scores(
    resume_text: str,
    jd_text: str,
    skill_result: dict[str, Any],
    evidence_matches: list[dict[str, Any]],
    resume_sections: dict[str, list[str]],
) -> dict[str, Any]:
    """Calculate the required four-dimension score."""
    hard_jd_skills: list[str] = []
    hard_matched_skills: list[str] = []
    for category, skills in skill_result.get("jd_skills_by_category", {}).items():
        if category == "软技能":
            continue
        hard_jd_skills.extend(skills)
        hard_matched_skills.extend(skill_result.get("matched_skills_by_category", {}).get(category, []))

    skill_entity_score = _ratio_score(hard_jd_skills, hard_matched_skills)
    semantic_evidence_score = _best_evidence_average(evidence_matches)
    project_score = _project_experience_score(resume_sections, jd_text)

    soft_jd = skill_result.get("jd_skills_by_category", {}).get("软技能", [])
    soft_matched = skill_result.get("matched_skills_by_category", {}).get("软技能", [])
    soft_score = _ratio_score(soft_jd, soft_matched)

    total_score = _clip_score(
        skill_entity_score * 0.30
        + semantic_evidence_score * 0.40
        + project_score * 0.20
        + soft_score * 0.10
    )

    return {
        "total_score": total_score,
        "match_level": _match_level(total_score),
        "dimension_scores": {
            "技能实体匹配": skill_entity_score,
            "语义证据匹配": semantic_evidence_score,
            "项目经历匹配": project_score,
            "软技能匹配": soft_score,
        },
    }

