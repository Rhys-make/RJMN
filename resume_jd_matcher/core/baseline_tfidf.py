"""TF-IDF baseline for comparison with pretrained semantic matching."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import clean_text


def calculate_tfidf_baseline(resume_text: str, jd_text: str) -> dict[str, float | str]:
    """Calculate a simple TF-IDF cosine similarity baseline."""
    resume_cleaned = clean_text(resume_text)
    jd_cleaned = clean_text(jd_text)
    if not resume_cleaned or not jd_cleaned:
        return {"baseline_score": 0.0, "method": "TF-IDF char n-gram cosine"}

    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform([resume_cleaned, jd_cleaned])
        score = float(cosine_similarity(matrix[0], matrix[1])[0][0] * 100)
    except ValueError:
        score = 0.0
    return {
        "baseline_score": round(max(0.0, min(100.0, score)), 2),
        "method": "TF-IDF char n-gram cosine",
    }

