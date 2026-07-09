"""Export helpers for JSON and Markdown reports."""

from __future__ import annotations

import json
from typing import Any


def build_json_report(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_markdown_report(payload: dict[str, Any]) -> str:
    scores = payload["scores"]
    skill_result = payload["skill_result"]
    analysis = payload["generation_analysis"]
    lines = [
        "# 简历岗位匹配度分析报告",
        "",
        f"- 综合匹配度：{scores['total_score']} / 100",
        f"- 匹配等级：{scores['match_level']}",
        f"- 生成模式：{analysis.get('generation_mode', '')}",
        f"- 生成模型：{analysis.get('model_name', '')}",
        "",
        "## 维度得分",
    ]
    for name, score in scores["dimension_scores"].items():
        lines.append(f"- {name}: {score}")
    lines.extend(
        [
            "",
            "## 命中技能",
            "、".join(skill_result.get("matched_skills", [])) or "暂无",
            "",
            "## 缺失技能",
            "、".join(skill_result.get("missing_skills", [])) or "暂无",
            "",
            "## 候选人优势分析",
        ]
    )
    lines.extend([f"- {item}" for item in analysis.get("strengths", [])])
    lines.append("\n## 能力短板分析")
    lines.extend([f"- {item}" for item in analysis.get("weaknesses", [])])
    lines.append("\n## 简历优化建议")
    lines.extend([f"- {item}" for item in analysis.get("resume_suggestions", [])])
    lines.append("\n## 面试问题")
    for item in analysis.get("interview_questions", []):
        lines.append(f"- [{item.get('type', '')}] {item.get('question', '')}")
        lines.append(f"  回答思路：{item.get('answer_hint', '')}")
    lines.append("\n## 最终建议")
    lines.append(str(analysis.get("final_advice", "")))
    return "\n".join(lines)

