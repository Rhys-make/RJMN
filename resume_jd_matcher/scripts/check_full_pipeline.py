"""Run one end-to-end sample including BGE scoring and Qwen generation."""

from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = ROOT / "myenv310" / "Scripts" / "python.exe"
if PROJECT_PYTHON.exists() and Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
    os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evidence_matcher import match_evidence_with_status  # noqa: E402
from core.llm_generator import generate_analysis  # noqa: E402
from core.preprocess import split_jd_requirements, split_resume_sections  # noqa: E402
from core.scoring import calculate_scores  # noqa: E402
from core.skill_extractor import extract_skills  # noqa: E402


def main() -> None:
    resume_text = (ROOT / "data" / "sample_resume.txt").read_text(encoding="utf-8")
    jd_text = (ROOT / "data" / "sample_jd.txt").read_text(encoding="utf-8")

    resume_sections = split_resume_sections(resume_text)
    jd_requirements = split_jd_requirements(jd_text)
    skill_result = extract_skills(resume_text, jd_text)
    evidence_result = match_evidence_with_status(jd_requirements, resume_sections)
    scores = calculate_scores(resume_text, jd_text, skill_result, evidence_result["matches"], resume_sections)
    generation = generate_analysis(resume_text, jd_text, scores, skill_result, evidence_result["matches"])

    print("综合分:", scores["total_score"])
    print("匹配等级:", scores["match_level"])
    print("语义模型:", evidence_result["semantic_model"].get("mode"))
    print("生成模式:", generation.get("generation_mode"))
    print("优势数量:", len(generation.get("strengths", [])))
    print("建议数量:", len(generation.get("resume_suggestions", [])))
    print("面试题数量:", len(generation.get("interview_questions", [])))
    if generation.get("error"):
        print("生成错误:", generation["error"])


if __name__ == "__main__":
    main()
