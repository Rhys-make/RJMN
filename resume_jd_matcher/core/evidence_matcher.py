"""Match each JD requirement to resume evidence snippets."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config as project_config
from .embedding_model import cosine_similarity_matrix, encode_texts, get_embedding_status
from .preprocess import tokenize_text


EVIDENCE_TOP_K = getattr(project_config, "EVIDENCE_TOP_K", 3)
HIGH_MATCH_THRESHOLD = getattr(project_config, "HIGH_MATCH_THRESHOLD", 0.75)
MEDIUM_MATCH_THRESHOLD = getattr(project_config, "MEDIUM_MATCH_THRESHOLD", 0.55)


def _flatten_resume_sections(resume_sections: dict[str, list[str]]) -> list[dict[str, str]]:
    evidence_items: list[dict[str, str]] = []
    for section, items in resume_sections.items():
        for item in items:
            text = str(item or "").strip()
            if text:
                evidence_items.append({"section": section, "resume_text": text})
    return evidence_items


def _status_from_similarity(similarity: float) -> str:
    if similarity >= HIGH_MATCH_THRESHOLD:
        return "高度匹配"
    if similarity >= MEDIUM_MATCH_THRESHOLD:
        return "部分匹配"
    return "匹配不足"


def _tfidf_similarity_matrix(requirement_texts: list[str], evidence_texts: list[str]) -> np.ndarray:
    docs = requirement_texts + evidence_texts
    if not docs or not any(text.strip() for text in docs):
        return np.zeros((len(requirement_texts), len(evidence_texts)))
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(docs)
    req_matrix = matrix[: len(requirement_texts)]
    evidence_matrix = matrix[len(requirement_texts) :]
    tfidf_scores = cosine_similarity(req_matrix, evidence_matrix)
    overlap_scores = _token_overlap_matrix(requirement_texts, evidence_texts)
    return np.maximum(tfidf_scores, overlap_scores)


def _token_overlap_matrix(requirement_texts: list[str], evidence_texts: list[str]) -> np.ndarray:
    """Lexical overlap supplement used only when pretrained embeddings are unavailable."""
    matrix = np.zeros((len(requirement_texts), len(evidence_texts)))
    req_tokens = [set(tokenize_text(text)) for text in requirement_texts]
    evidence_tokens = [set(tokenize_text(text)) for text in evidence_texts]

    for i, req_set in enumerate(req_tokens):
        if not req_set:
            continue
        for j, evidence_set in enumerate(evidence_tokens):
            if not evidence_set:
                continue
            intersection = req_set & evidence_set
            if not intersection:
                continue
            coverage = len(intersection) / len(req_set)
            jaccard = len(intersection) / len(req_set | evidence_set)
            matrix[i][j] = min(0.95, coverage * 0.75 + jaccard * 0.25)
    return matrix


def semantic_similarity_pair(text_a: str, text_b: str) -> tuple[float, dict[str, Any]]:
    """Return semantic similarity in [0, 1] using embeddings, with TF-IDF fallback."""
    try:
        embeddings = encode_texts([text_a, text_b])
        score = float(cosine_similarity_matrix(embeddings[:1], embeddings[1:2])[0][0])
        status = get_embedding_status()
        status["mode"] = "pretrained_embedding"
        return max(0.0, min(1.0, score)), status
    except Exception as exc:  # noqa: BLE001
        matrix = _tfidf_similarity_matrix([text_a], [text_b])
        status = get_embedding_status()
        status.update(
            {
                "available": False,
                "mode": "tfidf_fallback",
                "error": str(exc),
            }
        )
        return float(matrix[0][0]) if matrix.size else 0.0, status


def match_evidence_with_status(
    jd_requirements: list[dict[str, str]],
    resume_sections: dict[str, list[str]],
    top_k: int = EVIDENCE_TOP_K,
) -> dict[str, Any]:
    """Return evidence matches plus semantic backend status."""
    evidence_items = _flatten_resume_sections(resume_sections)
    requirement_texts = [item["text"] for item in jd_requirements if item.get("text")]
    evidence_texts = [item["resume_text"] for item in evidence_items]

    if not requirement_texts or not evidence_texts:
        return {
            "matches": [],
            "semantic_model": {
                "available": False,
                "mode": "empty_input",
                "model_name": get_embedding_status().get("model_name", ""),
                "error": "JD 要求或简历证据为空",
            },
        }

    try:
        req_embeddings = encode_texts(requirement_texts)
        evidence_embeddings = encode_texts(evidence_texts)
        similarity = cosine_similarity_matrix(req_embeddings, evidence_embeddings)
        semantic_model = get_embedding_status()
        semantic_model["mode"] = "pretrained_embedding"
    except Exception as exc:  # noqa: BLE001 - keep the application available.
        similarity = _tfidf_similarity_matrix(requirement_texts, evidence_texts)
        semantic_model = get_embedding_status()
        semantic_model.update(
            {
                "available": False,
                "mode": "tfidf_fallback",
                "error": str(exc),
            }
        )

    matches: list[dict[str, Any]] = []
    for row_index, requirement in enumerate(requirement_texts):
        row = similarity[row_index]
        ranked_indices = row.argsort()[::-1][:top_k]
        top_evidence = []
        for index in ranked_indices:
            score = float(max(0.0, min(1.0, row[index])))
            evidence = evidence_items[int(index)]
            top_evidence.append(
                {
                    "resume_text": evidence["resume_text"],
                    "similarity": round(score, 4),
                    "section": evidence["section"],
                }
            )
        best_score = top_evidence[0]["similarity"] if top_evidence else 0.0
        req_meta = jd_requirements[row_index]
        matches.append(
            {
                "requirement": requirement,
                "requirement_type": req_meta.get("type", "任职要求"),
                "top_evidence": top_evidence,
                "best_similarity": best_score,
                "match_status": _status_from_similarity(best_score),
            }
        )

    return {"matches": matches, "semantic_model": semantic_model}


def match_evidence(
    jd_requirements: list[dict[str, str]],
    resume_sections: dict[str, list[str]],
    top_k: int = EVIDENCE_TOP_K,
) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only the evidence list."""
    return match_evidence_with_status(jd_requirements, resume_sections, top_k)["matches"]
