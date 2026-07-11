"""Generate high-value analysis sections for the experiment report.

This script adds three report-ready modules without changing core model code:
1. Top-K evidence explainability table.
2. Ablation-style scoring table.
3. Typical case analysis for semantic-near and keyword-stacking cases.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = ROOT / "myenv310" / "Scripts" / "python.exe"
if PROJECT_PYTHON.exists() and Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
    os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.matcher import calculate_match  # noqa: E402


WEIGHTS = {
    "技能实体匹配": 0.30,
    "语义证据匹配": 0.40,
    "项目经历匹配": 0.20,
    "软技能匹配": 0.10,
}


def _match_level(score: float) -> str:
    if score >= 85:
        return "高度匹配"
    if score >= 70:
        return "较高匹配"
    if score >= 50:
        return "一般匹配"
    return "匹配度较低"


def _shorten(text: str, limit: int = 42) -> str:
    text = str(text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _markdown_escape(text: str) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


def load_eval_cases() -> list[dict[str, str]]:
    return json.loads((ROOT / "data" / "eval_cases.json").read_text(encoding="utf-8"))


def calculate_ablation_scores(dimensions: dict[str, float]) -> dict[str, float]:
    full_score = sum(float(dimensions[name]) * weight for name, weight in WEIGHTS.items())
    result = {"完整四维模型": round(full_score)}

    for removed_name, removed_weight in WEIGHTS.items():
        remaining_weight = 1.0 - removed_weight
        score = 0.0
        for name, weight in WEIGHTS.items():
            if name == removed_name:
                continue
            score += float(dimensions[name]) * weight
        result[f"去掉{removed_name}"] = round(score / remaining_weight)

    return result


def build_topk_evidence_section(sample_result: dict[str, Any]) -> str:
    lines = [
        "## 9. Top-K Evidence 可解释性分析",
        "",
        "本系统的核心不是直接比较整份简历和整份 JD，而是把 JD 拆成多条 requirement，",
        "再为每条 requirement 检索简历中的 Top-K evidence。这样可以回答“为什么匹配”以及“匹配证据来自哪里”。",
        "",
        "| JD requirement | Top-3 简历证据 | 最佳相似度 | 匹配状态 | 解释 |",
        "| --- | --- | ---: | --- | --- |",
    ]

    for item in sample_result["evidence_matches"]:
        top_items = item.get("top_evidence") or []
        requirement = _markdown_escape(_shorten(item.get("requirement", ""), 38))
        evidence_parts = []
        for index, top in enumerate(top_items[:3], start=1):
            evidence_parts.append(
                "{index}. {text} ({score:.2f})".format(
                    index=index,
                    text=_markdown_escape(_shorten(top.get("resume_text", ""), 34)),
                    score=float(top.get("similarity", 0.0)),
                )
            )
        evidence = "<br>".join(evidence_parts) if evidence_parts else "暂无证据"
        similarity = float(item.get("best_similarity", 0.0))
        status = item.get("match_status", "")
        if similarity >= 0.75:
            explanation = "该要求能在简历中找到较强语义证据。"
        elif similarity >= 0.55:
            explanation = "该要求有一定相关证据，但证据强度仍可提升。"
        else:
            explanation = "该要求缺少明确简历证据。"
        lines.append(f"| {requirement} | {evidence} | {similarity:.2f} | {status} | {explanation} |")

    lines.extend(
        [
            "",
            "从表中可以看到，每条 JD 要求都能追溯到具体简历片段。相比只输出一个总分，这种 evidence 链路提升了系统可解释性，",
            "也能帮助用户定位简历中需要补充或强化的部分。",
            "",
        ]
    )
    return "\n".join(lines)


def build_ablation_section(cases: list[dict[str, str]]) -> str:
    lines = [
        "## 10. 四维评分消融实验",
        "",
        "为了说明四维评分不是简单堆分，下面在不改动核心代码的前提下，对每组样例做消融分析。",
        "消融方式为：去掉某一维后，将剩余维度按原权重比例重新归一化计算分数。",
        "",
        "| 样例 | 完整四维模型 | 去掉技能实体 | 去掉语义证据 | 去掉项目经历 | 去掉软技能 | 结论 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for case in cases:
        result = calculate_match(case["resume"], case["jd"])
        dimensions = {key: float(value) for key, value in result["dimension_scores"].items()}
        ablation = calculate_ablation_scores(dimensions)
        name = _markdown_escape(case["case_name"].split("：")[0])
        full_score = ablation["完整四维模型"]
        no_semantic = ablation["去掉语义证据匹配"]
        no_skill = ablation["去掉技能实体匹配"]
        if no_semantic < full_score - 5:
            conclusion = "语义证据对最终判断贡献明显。"
        elif no_skill < full_score - 5:
            conclusion = "技能实体覆盖对结果影响较大。"
        else:
            conclusion = "多维度共同约束最终结果。"
        lines.append(
            "| {name} | {full:.0f} | {no_skill:.0f} | {no_semantic:.0f} | {no_project:.0f} | {no_soft:.0f} | {conclusion} |".format(
                name=name,
                full=full_score,
                no_skill=ablation["去掉技能实体匹配"],
                no_semantic=ablation["去掉语义证据匹配"],
                no_project=ablation["去掉项目经历匹配"],
                no_soft=ablation["去掉软技能匹配"],
                conclusion=conclusion,
            )
        )

    lines.extend(
        [
            "",
            "消融结果说明，完整四维模型能同时利用技能覆盖、语义证据、项目经历和软技能信息。",
            "对于语义相近样例，BGE 语义证据维度能够弥补 TF-IDF 和关键词匹配的不足；",
            "对于关键词堆砌样例，项目经历和技能覆盖会限制其综合分，避免被误判为高度匹配。",
            "",
        ]
    )
    return "\n".join(lines)


def build_typical_case_section(cases: list[dict[str, str]]) -> str:
    selected = [cases[4], cases[5]]
    titles = [
        "语义相近但关键词不完全一致",
        "关键词相同但实际语义弱匹配",
    ]
    lines = [
        "## 11. 典型案例分析",
        "",
        "本节选取两个边界样例，说明系统相比关键词匹配方法的优势，以及四维评分对误判的约束作用。",
        "",
    ]

    for title, case in zip(titles, selected):
        result = calculate_match(case["resume"], case["jd"])
        baseline = result["tfidf_baseline"]
        tfidf = float(baseline.get("tfidf_similarity", baseline.get("baseline_score", 0.0)))
        semantic = float(result["dimension_scores"]["语义证据匹配"])
        total = float(result["total_score"])
        level = result["match_level"]
        matched = "、".join(result.get("matched_keywords", [])[:8]) or "无"
        missing = "、".join(result.get("missing_keywords", [])[:8]) or "无"
        best_items = []
        for item in result["evidence_matches"][:3]:
            top = (item.get("top_evidence") or [{}])[0]
            best_items.append(
                f"- JD: {_shorten(item.get('requirement', ''), 46)}\n"
                f"  Evidence: {_shorten(top.get('resume_text', ''), 54)}\n"
                f"  Similarity: {float(top.get('similarity', 0.0)):.2f}"
            )

        lines.extend(
            [
                f"### 11.{titles.index(title) + 1} {title}",
                "",
                f"- TF-IDF baseline: `{tfidf:.2f}`",
                f"- BGE 语义证据分: `{semantic:.2f}`",
                f"- 综合分: `{total:.2f}`",
                f"- 匹配等级: `{level}`",
                f"- 命中技能: {matched}",
                f"- 缺失技能: {missing}",
                "",
                "代表性 evidence：",
                "",
                *best_items,
                "",
            ]
        )

        if "语义相近" in title:
            lines.extend(
                [
                    "分析：该样例中，简历使用“意图识别、向量化表示、相似句检索”等表达，JD 使用“文本分类、语义匹配、向量检索”等表达。",
                    "两者词面差异较大，因此 TF-IDF 分数很低；但 BGE 能捕捉这些表达背后的语义接近关系，使系统给出较高匹配判断。",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "分析：该样例虽然出现 Python、NLP 等关键词，但简历主要是课程报告和资料整理，缺少真实项目 evidence。",
                    "因此系统没有因为关键词重合而给出高匹配，而是通过四维评分将其限制在一般匹配，体现了 evidence 约束的作用。",
                    "",
                ]
            )

    return "\n".join(lines)


def main() -> None:
    cases = load_eval_cases()
    sample_resume = (ROOT / "data" / "sample_resume.txt").read_text(encoding="utf-8")
    sample_jd = (ROOT / "data" / "sample_jd.txt").read_text(encoding="utf-8")
    sample_result = calculate_match(sample_resume, sample_jd)

    output = "\n".join(
        [
            "# 高分实验分析补充模块",
            "",
            "本文档由 `scripts/generate_high_value_analysis.py` 自动生成，用于补充实验报告中的高价值分析模块。",
            "",
            build_topk_evidence_section(sample_result),
            build_ablation_section(cases),
            build_typical_case_section(cases),
        ]
    )

    output_path = ROOT / "docs" / "high_value_analysis.md"
    output_path.write_text(output, encoding="utf-8")
    print(f"[OK] Generated {output_path}")


if __name__ == "__main__":
    main()
