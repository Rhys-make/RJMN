"""Generation module based on Hugging Face Qwen with template fallback."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import config as project_config


GENERATION_MODEL_NAME = getattr(project_config, "GENERATION_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
GENERATION_MODEL_LOCAL_PATH = getattr(project_config, "GENERATION_MODEL_LOCAL_PATH", None)
GENERATION_MAX_NEW_TOKENS = getattr(project_config, "GENERATION_MAX_NEW_TOKENS", 768)
USE_GENERATION_MODEL = getattr(project_config, "USE_GENERATION_MODEL", True)


_GENERATION_STATUS: dict[str, Any] = {
    "available": False,
    "model_name": GENERATION_MODEL_NAME,
    "generation_mode": "pending",
    "device": "unknown",
    "error": "生成模型尚未加载",
}
_FAILED_ERROR: str | None = None


def _usable_local_qwen_dir(path_value) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_dir():
        return None
    has_config = (path / "config.json").exists()
    has_tokenizer = (path / "tokenizer.json").exists() or (path / "tokenizer.model").exists()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    if has_config and has_tokenizer and has_weights:
        return path
    return None


def _join(items: list[str], limit: int = 6) -> str:
    return "、".join(items[:limit]) if items else "暂无"


def _compact_evidence(evidence_result: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    compact = []
    for item in evidence_result[:limit]:
        top = item.get("top_evidence") or []
        compact.append(
            {
                "requirement": item.get("requirement", ""),
                "match_status": item.get("match_status", ""),
                "best_similarity": item.get("best_similarity", 0),
                "best_evidence": top[0] if top else {},
            }
        )
    return compact


@lru_cache(maxsize=1)
def _load_generation_model():
    global _FAILED_ERROR, _GENERATION_STATUS
    if not USE_GENERATION_MODEL:
        _FAILED_ERROR = "配置 USE_GENERATION_MODEL=False，已关闭生成模型。"
        raise RuntimeError(_FAILED_ERROR)
    if _FAILED_ERROR:
        raise RuntimeError(_FAILED_ERROR)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        project_local_dir = _usable_local_qwen_dir(GENERATION_MODEL_LOCAL_PATH)
        if project_local_dir:
            tokenizer = AutoTokenizer.from_pretrained(str(project_local_dir), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(project_local_dir),
                torch_dtype=dtype,
                local_files_only=True,
            )
            load_mode = f"project_local:{project_local_dir}"
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    GENERATION_MODEL_NAME,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
                load_mode = "huggingface_cache"
            except Exception as local_exc:  # noqa: BLE001
                tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
                model = AutoModelForCausalLM.from_pretrained(
                    GENERATION_MODEL_NAME,
                    torch_dtype=dtype,
                )
                load_mode = f"online_download; local_cache_miss={local_exc}"
        model.to(device)
        model.eval()
        _GENERATION_STATUS = {
            "available": True,
            "model_name": GENERATION_MODEL_NAME,
            "generation_mode": "qwen2.5-1.5b-instruct",
            "device": device.upper(),
            "load_mode": load_mode,
            "error": "",
        }
        return tokenizer, model, torch, device
    except Exception as exc:  # noqa: BLE001
        _FAILED_ERROR = f"生成式模型 {GENERATION_MODEL_NAME} 加载失败：{exc}"
        _GENERATION_STATUS = {
            "available": False,
            "model_name": GENERATION_MODEL_NAME,
            "generation_mode": "template_fallback",
            "device": "unknown",
            "load_mode": "unavailable",
            "error": _FAILED_ERROR,
        }
        raise RuntimeError(_FAILED_ERROR) from exc


def get_generation_status() -> dict[str, Any]:
    return dict(_GENERATION_STATUS)


def _build_prompt(
    resume_text: str,
    jd_text: str,
    scoring_result: dict[str, Any],
    skill_result: dict[str, Any],
    evidence_result: list[dict[str, Any]],
) -> str:
    payload = {
        "resume_excerpt": resume_text[:1000],
        "jd_excerpt": jd_text[:1000],
        "scoring_result": scoring_result,
        "matched_skills": skill_result.get("matched_skills", []),
        "missing_skills": skill_result.get("missing_skills", []),
        "evidence_result": _compact_evidence(evidence_result),
    }
    return f"""
你是严谨的中文自然语言处理课程设计分析助手。
请只根据给定简历、JD、技能匹配结果和证据匹配结果进行分析，不要编造简历中没有的信息。
请输出严格 JSON，不要输出 Markdown，不要添加额外解释。

JSON 结构必须为：
{{
  "strengths": ["..."],
  "weaknesses": ["..."],
  "resume_suggestions": ["..."],
  "interview_questions": [
    {{"type": "基础技术问题", "question": "...", "answer_hint": "..."}},
    {{"type": "项目深挖问题", "question": "...", "answer_hint": "..."}},
    {{"type": "岗位匹配问题", "question": "...", "answer_hint": "..."}}
  ],
  "final_advice": "..."
}}

给定信息：
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def _parse_json(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _ensure_list(value: Any, fallback: list[Any], limit: int = 5, min_count: int = 3) -> list[Any]:
    if isinstance(value, list):
        items = [item for item in value if item]
        if items:
            for item in fallback:
                if len(items) >= min_count:
                    break
                if item not in items:
                    items.append(item)
            return items[:limit]
    if isinstance(value, str) and value.strip():
        pieces = [piece.strip(" -；;。") for piece in re.split(r"[\n；;]", value) if piece.strip()]
        items = pieces if pieces else [value.strip()]
        for item in fallback:
            if len(items) >= min_count:
                break
            if item not in items:
                items.append(item)
        return items[:limit]
    return fallback[:limit]


def _normalize_analysis_result(
    parsed: dict[str, Any],
    scoring_result: dict[str, Any],
    skill_result: dict[str, Any],
    evidence_result: list[dict[str, Any]],
) -> dict[str, Any]:
    fallback = template_fallback_analysis(scoring_result, skill_result, evidence_result)
    interview_items = parsed.get("interview_questions", [])
    normalized_questions: list[dict[str, str]] = []
    if isinstance(interview_items, list):
        for item in interview_items:
            if isinstance(item, dict):
                normalized_questions.append(
                    {
                        "type": str(item.get("type") or "面试问题"),
                        "question": str(item.get("question") or ""),
                        "answer_hint": str(item.get("answer_hint") or item.get("answer") or ""),
                    }
                )
            elif isinstance(item, str):
                normalized_questions.append({"type": "面试问题", "question": item, "answer_hint": "结合简历证据和岗位要求作答。"})
    normalized_questions = [item for item in normalized_questions if item["question"]]
    if not normalized_questions:
        normalized_questions = fallback["interview_questions"]

    return {
        "strengths": _ensure_list(parsed.get("strengths"), fallback["strengths"], min_count=3),
        "weaknesses": _ensure_list(parsed.get("weaknesses"), fallback["weaknesses"], min_count=3),
        "resume_suggestions": _ensure_list(parsed.get("resume_suggestions"), fallback["resume_suggestions"], min_count=4),
        "interview_questions": normalized_questions[:5],
        "final_advice": str(parsed.get("final_advice") or fallback["final_advice"]),
    }


def template_fallback_analysis(
    scoring_result: dict[str, Any],
    skill_result: dict[str, Any],
    evidence_result: list[dict[str, Any]],
    error: str = "",
) -> dict[str, Any]:
    matched = skill_result.get("matched_skills", [])
    missing = skill_result.get("missing_skills", [])
    weak_requirements = [
        item.get("requirement", "")
        for item in evidence_result
        if item.get("match_status") != "高度匹配"
    ][:3]

    strengths = [
        f"简历已覆盖 {_join(matched, 6)} 等岗位相关技能，说明候选人与 JD 存在明确技能交集。",
        f"综合匹配度为 {scoring_result['total_score']} 分，匹配等级为 {scoring_result['match_level']}，适合作为岗位初筛参考。",
    ]
    if evidence_result:
        best = evidence_result[0].get("top_evidence", [{}])[0]
        strengths.append(f"系统能够为 JD 要求找到简历证据，例如“{best.get('resume_text', '暂无证据')}”。")

    weaknesses = [f"JD 中的 {_join(missing, 6)} 等技能或能力在简历中体现不足。"]
    if weak_requirements:
        weaknesses.append("部分 JD 要求缺少强证据支撑，例如：" + "；".join(weak_requirements))
    weaknesses.append("建议进一步补充项目结果和量化指标，提升经历可信度。")

    suggestions = [
        "建议使用“任务-方法-结果”的结构重写项目经历。",
        f"建议在项目或技能部分补充 {_join(missing, 5)} 等 JD 关键词，并说明具体应用场景。",
        "建议增加准确率、召回率、F1、处理数据量或效率提升比例等量化结果。",
        "建议将最匹配目标岗位的 NLP 或数据处理项目放在简历靠前位置。",
    ]

    questions = [
        {
            "type": "基础技术问题",
            "question": "请解释 BGE 语义向量模型与 TF-IDF 在文本匹配中的区别。",
            "answer_hint": "可从词面匹配、上下文语义表示、向量相似度和泛化能力等角度回答。",
        },
        {
            "type": "项目深挖问题",
            "question": "请介绍一个你做过的数据清洗或 NLP 文本处理项目。",
            "answer_hint": "按数据来源、清洗规则、模型或算法、评估指标和个人贡献展开。",
        },
        {
            "type": "岗位匹配问题",
            "question": f"JD 中提到 {_join(missing, 2)}，你准备如何补齐？",
            "answer_hint": "建议结合课程学习、项目实践和短期可完成的实验计划回答。",
        },
    ]

    return {
        "strengths": strengths[:5],
        "weaknesses": weaknesses[:5],
        "resume_suggestions": suggestions[:5],
        "interview_questions": questions,
        "final_advice": "建议优先补充与 JD 直接相关的项目证据和量化结果，并准备围绕证据链进行面试表达。",
        "generation_mode": "template_fallback",
        "model_name": GENERATION_MODEL_NAME,
        "device": get_generation_status().get("device", "unknown"),
        "error": error,
        "raw_text": "",
    }


def generate_analysis(
    resume_text: str,
    jd_text: str,
    scoring_result: dict[str, Any],
    skill_result: dict[str, Any],
    evidence_result: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt = _build_prompt(resume_text, jd_text, scoring_result, skill_result, evidence_result)
    try:
        tokenizer, model, torch, device = _load_generation_model()

        def _generate_once(user_prompt: str, max_tokens: int) -> str:
            messages = [
                {"role": "system", "content": "你是严谨的中文 NLP 课程设计分析助手，只输出合法 JSON。"},
                {"role": "user", "content": user_prompt},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = user_prompt
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        max_new_tokens = min(GENERATION_MAX_NEW_TOKENS, 512 if device == "cpu" else GENERATION_MAX_NEW_TOKENS)
        raw_text = _generate_once(prompt, max_new_tokens)
        parsed = _parse_json(raw_text)
        if not parsed:
            repair_prompt = (
                "把下面文本整理为严格 JSON，只保留 strengths、weaknesses、resume_suggestions、"
                "interview_questions、final_advice 五个字段，不要输出 Markdown。\n"
                f"原始文本：\n{raw_text[:1800]}"
            )
            repaired_text = _generate_once(repair_prompt, 256)
            parsed = _parse_json(repaired_text)
            raw_text = raw_text + "\n\n[JSON_REPAIR_OUTPUT]\n" + repaired_text

        if parsed:
            parsed = _normalize_analysis_result(parsed, scoring_result, skill_result, evidence_result)
        else:
            parsed = template_fallback_analysis(scoring_result, skill_result, evidence_result)
            parsed["warning"] = "Qwen 输出不是合法 JSON，已使用结构化修复结果。"

        parsed.update(
            {
                "generation_mode": "qwen2.5-1.5b-instruct",
                "model_name": GENERATION_MODEL_NAME,
                "device": _GENERATION_STATUS.get("device", "unknown"),
                "error": "",
                "raw_text": raw_text,
            }
        )
        return parsed
    except Exception as exc:  # noqa: BLE001
        return template_fallback_analysis(scoring_result, skill_result, evidence_result, str(exc))
