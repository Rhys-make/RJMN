from __future__ import annotations

import inspect
import re
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import config as project_config
from core.baseline_tfidf import calculate_tfidf_baseline
from core.evidence_matcher import match_evidence_with_status
from core.file_reader import read_uploaded_file
from core.llm_generator import generate_analysis, get_generation_status
import core.preprocess as preprocess_module
from core.report_generator import build_json_report, build_markdown_report
from core.scoring import calculate_scores
from core.skill_extractor import extract_skills


EMBEDDING_MODEL_NAME = getattr(project_config, "EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
GENERATION_MODEL_NAME = getattr(project_config, "GENERATION_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _fallback_split_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in str(text or "").replace("\r\n", "\n").split("\n"):
        line = re.sub(r"\s+", " ", line).strip(" -；;。")
        if line:
            lines.append(line)
    return lines


def _fallback_split_sentences(text: str) -> list[str]:
    pieces = re.split(r"[\n。；;.!?！？]+", str(text or ""))
    return [re.sub(r"\s+", " ", piece).strip(" -:：") for piece in pieces if piece.strip()]


def _fallback_split_resume_sections(resume_text: str) -> dict[str, list[str]]:
    section_aliases = {
        "教育背景": ("教育", "教育背景", "学历", "学校", "主修课程"),
        "专业技能": ("技能", "技能概述", "专业技能", "技术栈", "能力"),
        "项目经历": ("项目", "项目经历", "项目经验"),
        "实习经历": ("实习", "实习经历", "工作经历", "工作经验"),
        "其他经历": ("个人特点", "自我评价", "获奖", "证书", "其他"),
    }
    sections: dict[str, list[str]] = {name: [] for name in section_aliases}
    current_section = "其他经历"

    for line in _fallback_split_lines(resume_text):
        compact = line.replace(" ", "")
        matched_section = None
        for section, aliases in section_aliases.items():
            if any(alias in compact for alias in aliases) and len(compact) <= 24:
                matched_section = section
                break
        if matched_section:
            current_section = matched_section
            continue
        sections.setdefault(current_section, []).extend(
            sentence for sentence in _fallback_split_sentences(line) if len(sentence) >= 4
        )

    if not any(sections.values()):
        sections["其他经历"] = _fallback_split_sentences(resume_text)
    return {section: items for section, items in sections.items() if items}


def _fallback_split_jd_requirements(jd_text: str) -> list[dict[str, str]]:
    category_aliases = {
        "岗位职责": ("岗位职责", "工作职责", "职责描述", "工作内容"),
        "任职要求": ("任职要求", "职位要求", "岗位要求", "任职资格"),
        "技能要求": ("技能要求", "技术要求", "能力要求"),
        "加分项": ("加分项", "优先", "加分", "优先考虑"),
    }
    current_type = "任职要求"
    requirements: list[dict[str, str]] = []

    for line in _fallback_split_lines(jd_text):
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
        requirements.extend(
            {"type": current_type, "text": sentence}
            for sentence in _fallback_split_sentences(line)
            if len(sentence) >= 4
        )

    if not requirements:
        requirements = [{"type": "任职要求", "text": sentence} for sentence in _fallback_split_sentences(jd_text)]
    return requirements


split_resume_sections = getattr(preprocess_module, "split_resume_sections", _fallback_split_resume_sections)
split_jd_requirements = getattr(preprocess_module, "split_jd_requirements", _fallback_split_jd_requirements)


st.set_page_config(
    page_title="简历-JD智能匹配系统",
    page_icon="📄",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2.4rem; max-width: 1320px; }
    .hero-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
        border: 1px solid #dbeafe; border-radius: 14px; padding: 1.2rem 1.35rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06); margin-bottom: 1rem;
    }
    .hero-title { color: #0f172a; font-size: 1.75rem; font-weight: 780; line-height: 1.25; margin-bottom: 0.35rem; }
    .hero-subtitle { color: #475569; line-height: 1.7; font-size: 1rem; }
    .section-title { color: #111827; font-size: 1.06rem; font-weight: 730; margin: 0.3rem 0 0.55rem; }
    .section-note { color: #64748b; font-size: 0.92rem; line-height: 1.65; margin: 0.15rem 0 0.75rem; }
    .status-card, .score-card, .metric-card, .report-card, .list-card, .question-card, .empty-card {
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
    }
    .status-card { padding: 0.82rem 0.95rem; min-height: 96px; }
    .status-label, .metric-label { color: #64748b; font-size: 0.84rem; margin-bottom: 0.35rem; }
    .status-value { color: #0f172a; font-weight: 720; line-height: 1.45; word-break: break-word; }
    .status-extra { color: #64748b; font-size: 0.83rem; margin-top: 0.35rem; line-height: 1.45; }
    .score-card {
        padding: 1.1rem 1.18rem; background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        border-color: #bae6fd; min-height: 160px;
    }
    .score-value { color: #075985; font-size: 2.45rem; font-weight: 800; line-height: 1.1; margin: 0.2rem 0; }
    .metric-card { padding: 0.9rem 0.92rem; min-height: 122px; }
    .metric-value { color: #0f172a; font-size: 1.55rem; font-weight: 760; line-height: 1.2; }
    .metric-desc { color: #64748b; font-size: 0.82rem; margin-top: 0.36rem; line-height: 1.45; }
    .level-badge, .mode-badge {
        display: inline-block; border-radius: 999px; padding: 0.24rem 0.72rem;
        font-weight: 720; font-size: 0.92rem;
    }
    .level-high, .mode-ok { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .level-good { background: #e0f2fe; color: #075985; border: 1px solid #7dd3fc; }
    .level-normal, .mode-warn { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
    .level-low, .mode-fallback { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .chip {
        display: inline-block; border-radius: 999px; padding: 0.25rem 0.68rem;
        margin: 0.16rem 0.22rem 0.16rem 0; font-size: 0.88rem; line-height: 1.35;
    }
    .chip-matched { background: #ecfdf5; color: #166534; border: 1px solid #bbf7d0; }
    .chip-missing { background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }
    .chip-neutral { background: #f8fafc; color: #334155; border: 1px solid #cbd5e1; }
    .report-card { padding: 0.9rem 1rem; margin-bottom: 0.85rem; color: #334155; line-height: 1.65; }
    .list-card { padding: 0.82rem 0.95rem; margin-bottom: 0.65rem; color: #1f2937; line-height: 1.7; }
    .question-card { padding: 0.9rem 0.95rem; margin-bottom: 0.78rem; }
    .question-title { color: #111827; font-weight: 730; margin-bottom: 0.35rem; }
    .answer-idea { color: #475569; line-height: 1.65; font-size: 0.93rem; }
    .empty-card {
        background: #f8fafc; border: 1px dashed #cbd5e1; padding: 1.15rem 1.25rem;
        color: #475569; text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_session_state() -> None:
    st.session_state.setdefault("resume_text", "")
    st.session_state.setdefault("jd_text", "")
    st.session_state.setdefault("analysis_payload", None)
    st.session_state.setdefault("resume_file_id", "")
    st.session_state.setdefault("jd_file_id", "")


def _stretch_kwargs(streamlit_func) -> dict[str, str | bool]:
    try:
        parameters = inspect.signature(streamlit_func).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in parameters:
        return {"width": "stretch"}
    if "use_container_width" in parameters:
        return {"use_container_width": True}
    return {}


def _read_sample(filename: str) -> str:
    return (DATA_DIR / filename).read_text(encoding="utf-8")


def _load_sample_data() -> None:
    st.session_state["resume_text"] = _read_sample("sample_resume.txt")
    st.session_state["jd_text"] = _read_sample("sample_jd.txt")
    st.session_state["analysis_payload"] = None
    st.session_state["resume_file_id"] = ""
    st.session_state["jd_file_id"] = ""


def _file_identity(uploaded_file) -> str:
    return f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"


def _load_uploaded_text(uploaded_file, state_key: str, file_id_key: str, label: str) -> None:
    if uploaded_file is None:
        return
    file_id = _file_identity(uploaded_file)
    if st.session_state.get(file_id_key) == file_id:
        return
    text, error = read_uploaded_file(uploaded_file)
    if error:
        st.warning(f"{label}文件读取失败：{error}")
        return
    st.session_state[state_key] = text
    st.session_state[file_id_key] = file_id
    st.success(f"{label}文件已读取，可继续编辑文本内容。")


def _level_class(match_level: str) -> str:
    if match_level == "高度匹配":
        return "level-high"
    if match_level == "较高匹配":
        return "level-good"
    if match_level == "一般匹配":
        return "level-normal"
    return "level-low"


def _mode_class(mode: str, available: bool = False) -> str:
    if available and mode in {"qwen2.5-1.5b-instruct", "pretrained_embedding"}:
        return "mode-ok"
    if mode in {"template_fallback", "tfidf_fallback", "model_unavailable"}:
        return "mode-fallback"
    return "mode-warn"


def render_chips(keywords, color_type: str = "matched") -> None:
    if not keywords:
        st.markdown("<div class='section-note'>暂无相关关键词</div>", unsafe_allow_html=True)
        return
    css_class = {
        "matched": "chip-matched",
        "missing": "chip-missing",
        "neutral": "chip-neutral",
    }.get(color_type, "chip-neutral")
    html = "".join(f"<span class='chip {css_class}'>{escape(str(keyword))}</span>" for keyword in keywords)
    st.markdown(html, unsafe_allow_html=True)


def _render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">📄 基于预训练语言模型的简历岗位匹配度评估与面试辅助系统</div>
            <div class="hero-subtitle">
                以 BAAI/bge-small-zh-v1.5 语义向量为主方法，结合岗位要求-简历证据匹配、技能实体抽取、TF-IDF baseline 对比和 Qwen2.5 生成式分析，生成可解释的课程设计演示报告。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_card(label: str, value: str, extra: str = "", mode: str = "", available: bool = False) -> None:
    badge = ""
    if mode:
        badge = f"<div style='margin-top:0.45rem'><span class='mode-badge {_mode_class(mode, available)}'>{escape(mode)}</span></div>"
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{escape(label)}</div>
            <div class="status-value">{escape(value)}</div>
            {badge}
            <div class="status-extra">{escape(extra)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_model_status(payload: dict[str, Any] | None) -> None:
    st.markdown("<div class='section-title'>模型状态</div>", unsafe_allow_html=True)
    semantic_status = (payload or {}).get("semantic_model", {})
    generation = (payload or {}).get("generation_analysis", {})
    gen_status = get_generation_status()

    semantic_mode = semantic_status.get("mode", "pending")
    semantic_name = semantic_status.get("model_name", EMBEDDING_MODEL_NAME)
    semantic_available = bool(semantic_status.get("available", False))
    generation_mode = generation.get("generation_mode", "pending")
    generation_available = generation_mode == "qwen2.5-1.5b-instruct"
    device = generation.get("device") or gen_status.get("device", "unknown")
    semantic_extra = semantic_status.get("error") or f"加载方式：{semantic_status.get('load_mode', 'pending')}"
    generation_extra = generation.get("error") or f"加载方式：{generation.get('load_mode', gen_status.get('load_mode', 'pending'))}"

    cols = st.columns(4)
    with cols[0]:
        _render_status_card("当前语义模型", semantic_name, semantic_extra, semantic_mode, semantic_available)
    with cols[1]:
        _render_status_card("当前生成模型", generation.get("model_name", GENERATION_MODEL_NAME), generation_extra, generation_mode, generation_available)
    with cols[2]:
        _render_status_card("当前生成模式", generation_mode, "Qwen 不可用时显示 template_fallback", generation_mode, generation_available)
    with cols[3]:
        _render_status_card("设备信息", str(device).upper(), "CUDA 可用时自动使用 GPU，否则使用 CPU。", generation_mode, generation_available)


def _render_input_area() -> None:
    st.markdown("<div class='section-note'>支持直接粘贴文本，也支持上传 txt / pdf / docx 文件。</div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='section-title'>简历输入</div>", unsafe_allow_html=True)
        resume_file = st.file_uploader("上传简历文件", type=["txt", "docx", "pdf"], key="resume_upload")
        _load_uploaded_text(resume_file, "resume_text", "resume_file_id", "简历")
        st.text_area("简历文本", height=340, key="resume_text")
    with right:
        st.markdown("<div class='section-title'>岗位 JD 输入</div>", unsafe_allow_html=True)
        jd_file = st.file_uploader("上传 JD 文件", type=["txt", "docx", "pdf"], key="jd_upload")
        _load_uploaded_text(jd_file, "jd_text", "jd_file_id", "JD")
        st.text_area("岗位 JD 文本", height=340, key="jd_text")


def _run_analysis(resume_text: str, jd_text: str) -> dict[str, Any]:
    resume_sections = split_resume_sections(resume_text)
    jd_requirements = split_jd_requirements(jd_text)
    skill_result = extract_skills(resume_text, jd_text)
    evidence_result = match_evidence_with_status(jd_requirements, resume_sections)
    baseline = calculate_tfidf_baseline(resume_text, jd_text)
    scores = calculate_scores(resume_text, jd_text, skill_result, evidence_result["matches"], resume_sections)
    generation_analysis = generate_analysis(resume_text, jd_text, scores, skill_result, evidence_result["matches"])

    bge_score = scores["dimension_scores"]["语义证据匹配"]
    comparison = {
        "tfidf_baseline_score": baseline["baseline_score"],
        "bge_semantic_score": bge_score,
        "analysis": (
            "BGE 预训练语义模型能够利用上下文语义表示寻找 JD 要求与简历证据之间的相似关系；"
            "TF-IDF baseline 主要依赖字面重叠，因此在同义表达、描述顺序变化时稳定性较弱。"
        ),
    }
    return {
        "resume_sections": resume_sections,
        "jd_requirements": jd_requirements,
        "skill_result": skill_result,
        "evidence_matches": evidence_result["matches"],
        "semantic_model": evidence_result["semantic_model"],
        "baseline": baseline,
        "scores": scores,
        "generation_analysis": generation_analysis,
        "comparison": comparison,
    }


def _score_dataframe(scores: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"维度": name, "得分": float(score)} for name, score in scores["dimension_scores"].items()])


def _render_score_card(scores: dict[str, Any]) -> None:
    total_score = float(scores["total_score"])
    match_level = scores["match_level"]
    st.markdown(
        f"""
        <div class="score-card">
            <div class="metric-label">综合匹配度</div>
            <div class="score-value">{total_score:.1f} / 100</div>
            <span class="level-badge {_level_class(match_level)}">{escape(match_level)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(total_score / 100)


def _render_dimension_cards(scores: dict[str, Any]) -> None:
    descriptions = {
        "技能实体匹配": "JD 技能实体被简历覆盖的比例",
        "语义证据匹配": "逐条 JD 要求与简历证据的 BGE 相似度",
        "项目经历匹配": "JD 与项目/实习经历的语义相关性",
        "软技能匹配": "沟通协作、文档撰写等软技能覆盖",
    }
    columns = st.columns(len(scores["dimension_scores"]))
    for column, (name, score) in zip(columns, scores["dimension_scores"].items()):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{escape(name)}</div>
                    <div class="metric-value">{float(score):.2f}</div>
                    <div class="metric-desc">{escape(descriptions.get(name, ""))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_overview_tab(payload: dict[str, Any]) -> None:
    scores = payload["scores"]
    left, right = st.columns([1, 1.7])
    with left:
        _render_score_card(scores)
    with right:
        _render_dimension_cards(scores)
    st.markdown("<div class='section-title'>四维度得分柱状图</div>", unsafe_allow_html=True)
    score_df = _score_dataframe(scores)
    st.bar_chart(score_df.set_index("维度"), height=320)
    st.dataframe(score_df, hide_index=True, **_stretch_kwargs(st.dataframe))


def _render_evidence_tab(payload: dict[str, Any]) -> None:
    rows = []
    for item in payload["evidence_matches"]:
        top = item.get("top_evidence", [])
        best = top[0] if top else {}
        rows.append(
            {
                "JD要求": item.get("requirement", ""),
                "最匹配简历证据": best.get("resume_text", ""),
                "所属简历部分": best.get("section", ""),
                "相似度": round(float(item.get("best_similarity", 0.0)) * 100, 2),
                "匹配状态": item.get("match_status", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, **_stretch_kwargs(st.dataframe))
    with st.expander("查看每条 JD 要求的 Top-3 简历证据", expanded=False):
        for index, item in enumerate(payload["evidence_matches"], start=1):
            st.markdown(f"**{index}. JD 要求：** {item.get('requirement', '')}")
            evidence_rows = [
                {
                    "证据片段": evidence.get("resume_text", ""),
                    "所属部分": evidence.get("section", ""),
                    "相似度": round(float(evidence.get("similarity", 0.0)) * 100, 2),
                }
                for evidence in item.get("top_evidence", [])
            ]
            st.dataframe(pd.DataFrame(evidence_rows), hide_index=True, **_stretch_kwargs(st.dataframe))


def _render_skill_tab(payload: dict[str, Any]) -> None:
    skill_result = payload["skill_result"]
    hit_col, miss_col = st.columns(2)
    with hit_col:
        st.markdown("<div class='section-title'>命中技能</div>", unsafe_allow_html=True)
        render_chips(skill_result.get("matched_skills", []), "matched")
    with miss_col:
        st.markdown("<div class='section-title'>缺失技能</div>", unsafe_allow_html=True)
        render_chips(skill_result.get("missing_skills", []), "missing")

    rows = []
    for category in skill_result.get("jd_skills_by_category", {}).keys():
        rows.append(
            {
                "类别": category,
                "JD技能": "、".join(skill_result["jd_skills_by_category"].get(category, [])) or "-",
                "简历技能": "、".join(skill_result["resume_skills_by_category"].get(category, [])) or "-",
                "命中技能": "、".join(skill_result["matched_skills_by_category"].get(category, [])) or "-",
                "缺失技能": "、".join(skill_result["missing_skills_by_category"].get(category, [])) or "-",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, **_stretch_kwargs(st.dataframe))


def _render_list_cards(items: list[str], empty_text: str = "暂无内容") -> None:
    if not items:
        st.info(empty_text)
        return
    for index, item in enumerate(items, start=1):
        st.markdown(f"<div class='list-card'><strong>{index}.</strong> {escape(str(item))}</div>", unsafe_allow_html=True)


def _render_generation_tab(payload: dict[str, Any]) -> None:
    analysis = payload["generation_analysis"]
    mode = analysis.get("generation_mode", "template_fallback")
    badge_class = _mode_class(mode, mode == "qwen2.5-1.5b-instruct")
    st.markdown(
        f"<div class='report-card'>当前生成模式：<span class='mode-badge {badge_class}'>{escape(mode)}</span>"
        f"<br>模型：{escape(str(analysis.get('model_name', GENERATION_MODEL_NAME)))} ｜ 设备：{escape(str(analysis.get('device', 'unknown')).upper())}</div>",
        unsafe_allow_html=True,
    )
    if analysis.get("error"):
        st.warning(f"Qwen 生成模型不可用或输出异常，当前使用模板降级模式：{analysis['error']}")

    tabs = st.tabs(["候选人优势分析", "能力短板分析", "简历优化建议", "面试问题", "最终建议"])
    with tabs[0]:
        _render_list_cards(analysis.get("strengths", []))
    with tabs[1]:
        _render_list_cards(analysis.get("weaknesses", []))
    with tabs[2]:
        _render_list_cards(analysis.get("resume_suggestions", []))
    with tabs[3]:
        categories = ["基础技术问题", "项目深挖问题", "岗位匹配问题"]
        category_tabs = st.tabs(categories)
        questions = analysis.get("interview_questions", [])
        for tab, category in zip(category_tabs, categories):
            with tab:
                filtered = [item for item in questions if item.get("type") == category]
                if not filtered:
                    st.info("暂无该类别问题")
                for item in filtered:
                    st.markdown(
                        f"""
                        <div class="question-card">
                            <div class="question-title">问题：{escape(str(item.get('question', '')))}</div>
                            <div class="answer-idea">回答思路：{escape(str(item.get('answer_hint', '')))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    with tabs[4]:
        st.markdown(f"<div class='report-card'>{escape(str(analysis.get('final_advice', '暂无最终建议')))}</div>", unsafe_allow_html=True)


def _render_comparison_tab(payload: dict[str, Any]) -> None:
    comparison = payload["comparison"]
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">TF-IDF baseline 分数</div>
                <div class="metric-value">{float(comparison['tfidf_baseline_score']):.2f}</div>
                <div class="metric-desc">仅作为实验对比，不作为主方法。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">BGE 语义模型分数</div>
                <div class="metric-value">{float(comparison['bge_semantic_score']):.2f}</div>
                <div class="metric-desc">基于 JD requirement 与简历 evidence 的语义匹配。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    chart_df = pd.DataFrame(
        [
            {"方法": "TF-IDF baseline", "分数": float(comparison["tfidf_baseline_score"])},
            {"方法": "BGE 语义模型", "分数": float(comparison["bge_semantic_score"])},
        ]
    )
    st.bar_chart(chart_df.set_index("方法"), height=260)
    st.markdown(f"<div class='report-card'>{escape(comparison['analysis'])}</div>", unsafe_allow_html=True)


def _render_export_tab(payload: dict[str, Any]) -> None:
    json_data = build_json_report(payload)
    markdown_data = build_markdown_report(payload)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("导出 JSON 分析报告", data=json_data, file_name="resume_jd_match_report.json", mime="application/json", **_stretch_kwargs(st.download_button))
    with col2:
        st.download_button("导出 Markdown 分析报告", data=markdown_data, file_name="resume_jd_match_report.md", mime="text/markdown", **_stretch_kwargs(st.download_button))
    st.json(payload)


def _render_results(payload: dict[str, Any]) -> None:
    st.markdown("<div class='section-title'>分析报告</div>", unsafe_allow_html=True)
    st.markdown("<div class='report-card'>系统已完成结构化切分、技能实体抽取、BGE 语义证据匹配、TF-IDF baseline 对比和生成式分析。</div>", unsafe_allow_html=True)
    tabs = st.tabs(["匹配总览", "逐项证据匹配", "技能分析", "生成式分析", "对比实验", "导出报告"])
    with tabs[0]:
        _render_overview_tab(payload)
    with tabs[1]:
        _render_evidence_tab(payload)
    with tabs[2]:
        _render_skill_tab(payload)
    with tabs[3]:
        _render_generation_tab(payload)
    with tabs[4]:
        _render_comparison_tab(payload)
    with tabs[5]:
        _render_export_tab(payload)


def main() -> None:
    _init_session_state()
    _render_hero()
    _render_model_status(st.session_state.get("analysis_payload"))
    _render_input_area()

    left, middle, right = st.columns([1, 1, 1])
    with left:
        load_sample = st.button("加载示例数据", on_click=_load_sample_data, **_stretch_kwargs(st.button))
    with middle:
        analyze_clicked = st.button("开始分析", type="primary", **_stretch_kwargs(st.button))
    with right:
        st.markdown("<div class='report-card'>建议先加载示例数据，查看完整证据匹配和对比实验效果。</div>", unsafe_allow_html=True)

    if load_sample:
        st.success("示例数据已加载。")

    if analyze_clicked:
        resume_text = st.session_state.get("resume_text", "").strip()
        jd_text = st.session_state.get("jd_text", "").strip()
        if not resume_text or not jd_text:
            st.warning("请先输入或上传简历和岗位 JD。")
        else:
            with st.spinner("正在调用 BGE 语义匹配、证据检索和 Qwen/模板生成模块..."):
                st.session_state["analysis_payload"] = _run_analysis(resume_text, jd_text)

    st.divider()
    payload = st.session_state.get("analysis_payload")
    if payload:
        _render_results(payload)
    else:
        st.markdown("<div class='empty-card'>请在上方输入简历和岗位 JD，然后点击开始分析。</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
