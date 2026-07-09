"""Text cleaning and tokenization helpers."""

from __future__ import annotations

import re

try:
    import jieba
except ImportError:  # The app can still run basic English tokenization without jieba.
    jieba = None


STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "在",
    "对",
    "中",
    "为",
    "是",
    "有",
    "熟悉",
    "掌握",
    "负责",
    "具备",
    "能够",
    "相关",
    "进行",
    "使用",
    "基于",
    "the",
    "and",
    "or",
    "with",
    "for",
    "to",
    "in",
    "of",
    "a",
    "an",
}


def clean_text(text: str) -> str:
    """Normalize text while preserving Chinese, English, digits, and tech symbols."""
    text = str(text or "")
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\+\#\./_\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize_text(text: str) -> list[str]:
    """Tokenize mixed Chinese and English text."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    tokens: list[str] = []
    if jieba is not None:
        tokens.extend(jieba.lcut(cleaned))

    tech_pattern = r"\.?[a-zA-Z][a-zA-Z0-9]*(?:\+\+|#|[._\-][a-zA-Z0-9]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+"
    tokens.extend(re.findall(tech_pattern, cleaned))

    normalized_tokens: list[str] = []
    seen = set()
    for token in tokens:
        token = token.strip().lower()
        token = token.strip("._-/")
        if not token or token in STOPWORDS:
            continue
        if len(token) == 1 and token not in {"c", "r"} and not token.isdigit():
            continue
        if token not in seen:
            normalized_tokens.append(token)
            seen.add(token)
    return normalized_tokens


def preprocess_text(text: str) -> dict[str, object]:
    """Return cleaned text and tokens in one structured result."""
    cleaned = clean_text(text)
    return {
        "cleaned_text": cleaned,
        "tokens": tokenize_text(cleaned),
    }


def _split_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in str(text or "").replace("\r\n", "\n").split("\n"):
        line = re.sub(r"\s+", " ", line).strip(" -；;。")
        if line:
            lines.append(line)
    return lines


def _split_sentences(text: str) -> list[str]:
    pieces = re.split(r"[\n。；;.!?！？]+", str(text or ""))
    return [re.sub(r"\s+", " ", piece).strip(" -:：") for piece in pieces if piece.strip()]


def split_resume_sections(resume_text: str) -> dict[str, list[str]]:
    """Roughly split a resume into common sections for evidence matching."""
    section_aliases = {
        "教育背景": ("教育", "教育背景", "学历", "学校", "主修课程"),
        "专业技能": ("技能", "技能概述", "专业技能", "技术栈", "能力"),
        "项目经历": ("项目", "项目经历", "项目经验"),
        "实习经历": ("实习", "实习经历", "工作经历", "工作经验"),
        "其他经历": ("个人特点", "自我评价", "获奖", "证书", "其他"),
    }
    sections: dict[str, list[str]] = {name: [] for name in section_aliases}
    current_section = "其他经历"

    for line in _split_lines(resume_text):
        compact = line.replace(" ", "")
        matched_section = None
        for section, aliases in section_aliases.items():
            if any(alias in compact for alias in aliases) and len(compact) <= 24:
                matched_section = section
                break

        if matched_section:
            current_section = matched_section
            continue

        for sentence in _split_sentences(line):
            if len(sentence) >= 4:
                sections.setdefault(current_section, []).append(sentence)

    if not any(sections.values()):
        sections["其他经历"] = _split_sentences(resume_text)

    return {section: items for section, items in sections.items() if items}


def split_jd_requirements(jd_text: str) -> list[dict[str, str]]:
    """Split JD into individual requirement items with a coarse requirement type."""
    category_aliases = {
        "岗位职责": ("岗位职责", "工作职责", "职责描述", "工作内容"),
        "任职要求": ("任职要求", "职位要求", "岗位要求", "任职资格"),
        "技能要求": ("技能要求", "技术要求", "能力要求"),
        "加分项": ("加分项", "优先", "加分", "优先考虑"),
    }
    current_type = "任职要求"
    requirements: list[dict[str, str]] = []

    for line in _split_lines(jd_text):
        compact = line.replace(" ", "")
        if compact.startswith(("岗位名称", "职位名称", "岗位名", "职位名")):
            continue
        matched_type = None
        for category, aliases in category_aliases.items():
            if any(alias in compact for alias in aliases) and len(compact) <= 28:
                matched_type = category
                break
        if matched_type:
            current_type = matched_type
            continue

        line = re.sub(r"^[0-9一二三四五六七八九十]+[、.)）]\s*", "", line)
        for sentence in _split_sentences(line):
            if len(sentence) >= 4:
                requirements.append({"type": current_type, "text": sentence})

    if not requirements:
        requirements = [{"type": "任职要求", "text": sentence} for sentence in _split_sentences(jd_text)]

    return requirements
