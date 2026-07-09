"""Rule-based analysis and interview question generation."""

from __future__ import annotations

from typing import Any


def _join_keywords(keywords: list[str], limit: int = 6) -> str:
    return "、".join(keywords[:limit])


def _ensure_length(items: list[str], fallback: list[str], min_count: int = 3, max_count: int = 5) -> list[str]:
    for item in fallback:
        if len(items) >= min_count:
            break
        if item not in items:
            items.append(item)
    return items[:max_count]


def generate_advantages(match_result: dict[str, Any]) -> list[str]:
    matched = match_result.get("matched_keywords", [])
    scores = match_result.get("dimension_scores", {})
    advantages: list[str] = []

    if matched:
        advantages.append(f"简历中体现了 {_join_keywords(matched)} 等岗位要求关键词，说明候选人具备一定岗位相关基础。")
    if scores.get("技能实体匹配", 0) >= 70:
        advantages.append("核心技能覆盖度较高，简历中的技术栈与岗位要求具有较明显交集。")
    if scores.get("语义证据匹配", 0) >= 60:
        advantages.append("简历证据与 JD 要求在语义层面较接近，经历表达方向与岗位需求较一致。")
    if scores.get("项目经历匹配", 0) >= 70:
        advantages.append("项目或实习经历中包含开发、数据处理、系统优化等内容，能够支撑岗位职责要求。")
    if scores.get("软技能匹配", 0) >= 70:
        advantages.append("简历体现出沟通协作、文档撰写或学习执行等软技能，有利于团队协作类岗位。")

    fallback = [
        "简历具备基础技术经历，可通过进一步补充项目细节提升说服力。",
        "候选人经历中包含可迁移能力，适合围绕岗位需求继续强化表达。",
        "当前简历已经提供了部分技能和项目线索，便于面试官继续追问。",
    ]
    return _ensure_length(advantages, fallback)


def generate_shortcomings(match_result: dict[str, Any]) -> list[str]:
    missing = match_result.get("missing_keywords", [])
    scores = match_result.get("dimension_scores", {})
    shortcomings: list[str] = []

    if missing:
        shortcomings.append(f"JD 中提到 {_join_keywords(missing)} 等要求，但简历中体现不足。")
    if scores.get("技能实体匹配", 0) < 70:
        shortcomings.append("岗位技能关键词覆盖度仍有提升空间，需要更明确地展示与岗位相关的技术能力。")
    if scores.get("语义证据匹配", 0) < 55:
        shortcomings.append("简历证据与岗位职责的语义匹配度一般，可能存在经历描述不够聚焦的问题。")
    if scores.get("项目经历匹配", 0) < 70:
        shortcomings.append("项目经历对需求分析、数据处理、部署测试或业务结果的描述还不够充分。")
    if scores.get("软技能匹配", 0) < 70:
        shortcomings.append("沟通协作、文档撰写、责任心等软技能表达较少，建议结合团队项目补充。")

    fallback = [
        "简历中部分项目结果缺少量化指标，面试时可能难以体现实际贡献。",
        "建议补充关键技术的应用场景，避免只罗列技能名称。",
        "对岗位所需能力的证明材料还可以更具体，例如项目产出、指标提升或工具链使用。",
    ]
    return _ensure_length(shortcomings, fallback)


def generate_suggestions(match_result: dict[str, Any]) -> list[str]:
    missing = match_result.get("missing_keywords", [])
    suggestions: list[str] = []

    if missing:
        suggestions.append(f"建议在项目经历或技能部分补充 {_join_keywords(missing)} 等岗位关键词，并说明具体应用场景。")
    suggestions.extend(
        [
            "建议使用“任务-方法-结果”的结构优化项目描述，突出负责内容、技术路线和最终产出。",
            "建议增加量化结果，例如准确率、处理数据量、接口响应时间、报表效率提升比例等。",
            "建议将与 JD 最相关的技能和项目放在简历靠前位置，提高筛选阶段的信息命中率。",
            "建议为每段项目经历补充工具链，例如 Python、SQL、FastAPI、数据库、可视化平台或文本处理方法。",
        ]
    )
    return suggestions[:5]


def generate_interview_questions(match_result: dict[str, Any]) -> list[dict[str, str]]:
    matched = match_result.get("matched_keywords", [])
    missing = match_result.get("missing_keywords", [])
    primary_skill = matched[0] if matched else "Python"
    weak_skill = missing[0] if missing else "岗位核心技能"

    return [
        {
            "type": "基础技术问题",
            "question": "你如何理解 NLP 中的语义匹配？",
            "answer_hint": "可以从文本向量化、TF-IDF、余弦相似度、关键词匹配与语义匹配的区别等方面回答。",
        },
        {
            "type": "基础技术问题",
            "question": f"你在项目中如何使用 {primary_skill} 解决实际问题？",
            "answer_hint": "建议说明任务背景、使用该技术的原因、关键实现步骤和最终效果。",
        },
        {
            "type": "项目深挖问题",
            "question": "请介绍一个你做过的数据处理或文本处理项目。",
            "answer_hint": "按背景、数据来源、清洗流程、核心算法、结果评估和个人贡献展开。",
        },
        {
            "type": "项目深挖问题",
            "question": "如果项目数据质量较差，你会如何清洗和验证数据？",
            "answer_hint": "可以回答缺失值处理、重复数据去除、异常值检查、人工抽样验证和日志记录。",
        },
        {
            "type": "岗位匹配问题",
            "question": f"JD 中提到 {weak_skill}，你目前掌握到什么程度？后续如何补齐？",
            "answer_hint": "建议诚实说明已有基础，结合课程、项目实践和可落地学习计划展示成长性。",
        },
        {
            "type": "岗位匹配问题",
            "question": "你为什么认为自己的经历适合这个岗位？",
            "answer_hint": "从已匹配技能、相关项目经历、学习能力、协作经验和对岗位职责的理解进行回答。",
        },
    ]


def generate_analysis(match_result: dict[str, Any]) -> dict[str, Any]:
    """Generate all rule-based textual outputs for the UI and JSON export."""
    return {
        "strengths": generate_advantages(match_result),
        "weaknesses": generate_shortcomings(match_result),
        "resume_suggestions": generate_suggestions(match_result),
        "interview_questions": generate_interview_questions(match_result),
        "final_advice": "建议围绕 JD 关键词补充项目证据，并在面试中用任务-方法-结果结构说明个人贡献。",
        "generation_mode": "template_fallback",
    }
