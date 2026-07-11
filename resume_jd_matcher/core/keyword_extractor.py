"""Keyword extraction based on a skill dictionary and TF-IDF."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import clean_text, tokenize_text


DEFAULT_SKILL_DICT_PATH = Path(__file__).resolve().parents[1] / "data" / "skill_dict.json"

SKILL_ALIASES = {
    "SQL": ["mysql", "postgresql", "数据库查询", "数据库设计"],
    "自然语言处理": ["nlp", "文本处理", "语言处理", "意图识别"],
    "NLP": ["自然语言处理", "文本处理", "语言处理", "意图识别"],
    "文本分类": ["意图识别", "分类效果", "分类实验", "短文本分类"],
    "关键词提取": ["关键词抽取", "关键短语提取"],
    "语义匹配": ["相似句检索", "相似文本检索", "语义相似", "相似度匹配"],
    "向量检索": ["向量化表示", "相似句检索", "向量召回", "语义检索"],
    "模型评估": ["评估分类效果", "实验评估", "结果分析", "评估指标"],
    "数据清洗": ["清洗用户反馈语料", "清洗语料", "清洗文本", "数据预处理"],
    "文档撰写": ["文档", "实验记录", "报告撰写"],
    "沟通能力": ["沟通", "沟通需求", "沟通协作", "团队协作"],
    "团队协作": ["团队沟通", "沟通协作", "协作"],
}


def load_skill_dict(skill_dict_path: str | Path | None = None) -> dict[str, list[str]]:
    path = Path(skill_dict_path) if skill_dict_path else DEFAULT_SKILL_DICT_PATH
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _contains_skill(text: str, skill: str) -> bool:
    candidates = [skill, *SKILL_ALIASES.get(skill, [])]
    return any(_contains_skill_literal(text, candidate) for candidate in candidates)


def _contains_skill_literal(text: str, skill: str) -> bool:
    cleaned_text = clean_text(text)
    skill_text = clean_text(skill)
    if not cleaned_text or not skill_text:
        return False

    if _contains_chinese(skill_text):
        return skill_text.replace(" ", "") in cleaned_text.replace(" ", "")

    if skill_text in {"c++", "c#", ".net"}:
        return skill_text in cleaned_text

    if skill_text == "r":
        return re.search(r"(?<![a-z0-9])r(?![a-z0-9])", cleaned_text) is not None

    if any(symbol in skill_text for symbol in (" ", "+", "#", ".", "-", "/")):
        return skill_text in cleaned_text

    pattern = rf"(?<![a-z0-9]){re.escape(skill_text)}(?![a-z0-9])"
    return re.search(pattern, cleaned_text) is not None


def _extract_dict_keywords(text: str, skill_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for category, skills in skill_dict.items():
        result[category] = [skill for skill in skills if _contains_skill(text, skill)]
    return result


def _flatten_keywords(keywords_by_category: dict[str, list[str]]) -> list[str]:
    flattened: list[str] = []
    seen = set()
    for skills in keywords_by_category.values():
        for skill in skills:
            key = clean_text(skill)
            if key not in seen:
                flattened.append(skill)
                seen.add(key)
    return flattened


def _intersect_keywords(left: list[str], right: list[str]) -> list[str]:
    right_set = {clean_text(keyword) for keyword in right}
    return [keyword for keyword in left if clean_text(keyword) in right_set]


def _difference_keywords(left: list[str], right: list[str]) -> list[str]:
    right_set = {clean_text(keyword) for keyword in right}
    return [keyword for keyword in left if clean_text(keyword) not in right_set]


def _tfidf_keywords_for_docs(resume_text: str, jd_text: str, top_k: int = 10) -> dict[str, list[str]]:
    docs = [" ".join(tokenize_text(resume_text)), " ".join(tokenize_text(jd_text))]
    if not docs[0].strip() and not docs[1].strip():
        return {"resume": [], "jd": [], "overall": []}

    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=80)
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return {"resume": [], "jd": [], "overall": []}

    feature_names = vectorizer.get_feature_names_out()

    def top_terms(row_index: int) -> list[str]:
        row = matrix[row_index].toarray().ravel()
        ranked = row.argsort()[::-1]
        terms = [feature_names[index] for index in ranked if row[index] > 0]
        return terms[:top_k]

    total_scores = matrix.sum(axis=0).A1
    overall_ranked = total_scores.argsort()[::-1]
    overall = [feature_names[index] for index in overall_ranked if total_scores[index] > 0][:top_k]
    return {
        "resume": top_terms(0),
        "jd": top_terms(1),
        "overall": overall,
    }


def extract_keywords(
    resume_text: str,
    jd_text: str,
    skill_dict_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract dictionary keywords and supplementary TF-IDF terms."""
    skill_dict = load_skill_dict(skill_dict_path)
    resume_by_category = _extract_dict_keywords(resume_text, skill_dict)
    jd_by_category = _extract_dict_keywords(jd_text, skill_dict)

    resume_keywords = _flatten_keywords(resume_by_category)
    jd_keywords = _flatten_keywords(jd_by_category)
    matched_keywords = _intersect_keywords(jd_keywords, resume_keywords)
    missing_keywords = _difference_keywords(jd_keywords, resume_keywords)

    category_detail: dict[str, dict[str, list[str]]] = {}
    for category in skill_dict:
        resume_items = resume_by_category.get(category, [])
        jd_items = jd_by_category.get(category, [])
        category_detail[category] = {
            "resume": resume_items,
            "jd": jd_items,
            "matched": _intersect_keywords(jd_items, resume_items),
            "missing": _difference_keywords(jd_items, resume_items),
        }

    return {
        "resume_keywords": resume_keywords,
        "jd_keywords": jd_keywords,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "keywords_by_category": category_detail,
        "tfidf_keywords": _tfidf_keywords_for_docs(resume_text, jd_text),
    }
