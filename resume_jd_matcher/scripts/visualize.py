"""Generate all visualization charts for the NLP course project report.

Usage:
    python scripts/visualize.py

Output: outputs/ directory with PNG images.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Chinese font support ---
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# --- Project imports ---
from core.embedding_model import encode_texts, cosine_similarity_matrix, get_embedding_status
from core.preprocess import split_resume_sections, split_jd_requirements, tokenize_text
from core.skill_extractor import extract_skills
from core.evidence_matcher import match_evidence_with_status
from core.scoring import calculate_scores
from core.baseline_tfidf import calculate_tfidf_baseline
from core.matcher import calculate_match


def load_sample_data() -> tuple[str, str]:
    resume = (ROOT / "data" / "sample_resume.txt").read_text(encoding="utf-8")
    jd = (ROOT / "data" / "sample_jd.txt").read_text(encoding="utf-8")
    return resume, jd


def load_eval_cases() -> list[dict]:
    return json.loads((ROOT / "data" / "eval_cases.json").read_text(encoding="utf-8"))


# ============================================================
# Chart 1: Word Cloud
# ============================================================
def generate_wordcloud(resume_text: str, jd_text: str) -> None:
    """Generate word cloud images for resume and JD."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[SKIP] wordcloud not installed, skipping word cloud generation.")
        return

    font_path = r"C:\Windows\Fonts\simhei.ttf"
    if not Path(font_path).exists():
        font_path = r"C:\Windows\Fonts\msyh.ttc"
    if not Path(font_path).exists():
        font_path = None

    resume_tokens = tokenize_text(resume_text)
    jd_tokens = tokenize_text(jd_text)

    # Filter short tokens
    resume_words = " ".join([w for w in resume_tokens if len(w) >= 2])
    jd_words = " ".join([w for w in jd_tokens if len(w) >= 2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    wc_resume = WordCloud(
        font_path=font_path, width=600, height=400,
        background_color="white", max_words=80, colormap="Blues"
    ).generate(resume_words)
    axes[0].imshow(wc_resume, interpolation="bilinear")
    axes[0].set_title("简历词云", fontsize=14)
    axes[0].axis("off")

    wc_jd = WordCloud(
        font_path=font_path, width=600, height=400,
        background_color="white", max_words=80, colormap="Oranges"
    ).generate(jd_words)
    axes[1].imshow(wc_jd, interpolation="bilinear")
    axes[1].set_title("岗位JD词云", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_wordcloud.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 01_wordcloud.png")


# ============================================================
# Chart 2: Similarity Heatmap (requirement x evidence)
# ============================================================
def generate_similarity_heatmap(resume_text: str, jd_text: str) -> None:
    """Generate a heatmap of cosine similarity between JD requirements and resume evidence."""
    resume_sections = split_resume_sections(resume_text)
    jd_requirements = split_jd_requirements(jd_text)

    requirement_texts = [r["text"] for r in jd_requirements if r.get("text")]
    evidence_items = []
    for section, items in resume_sections.items():
        for item in items:
            if item.strip():
                evidence_items.append({"section": section, "text": item.strip()})

    evidence_texts = [e["text"] for e in evidence_items]

    if not requirement_texts or not evidence_texts:
        print("[SKIP] Empty requirements or evidence, skipping heatmap.")
        return

    # Encode and compute similarity
    req_embeddings = encode_texts(requirement_texts)
    evi_embeddings = encode_texts(evidence_texts)
    sim_matrix = cosine_similarity_matrix(req_embeddings, evi_embeddings)

    # Truncate labels for display
    req_labels = [t[:20] + "..." if len(t) > 20 else t for t in requirement_texts]
    evi_labels = [f"{e['section'][:4]}:{e['text'][:12]}..." for e in evidence_items]

    fig, ax = plt.subplots(figsize=(max(10, len(evidence_texts) * 0.6), max(6, len(requirement_texts) * 0.5)))
    sns.heatmap(
        sim_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=evi_labels, yticklabels=req_labels,
        vmin=0, vmax=1, ax=ax, linewidths=0.5,
        annot_kws={"size": 7}
    )
    ax.set_xlabel("简历证据片段", fontsize=11)
    ax.set_ylabel("JD 岗位要求", fontsize=11)
    ax.set_title("JD要求 × 简历证据 语义相似度热力图 (BGE)", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_similarity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 02_similarity_heatmap.png")


# ============================================================
# Chart 3: t-SNE / PCA Vector Visualization
# ============================================================
def generate_vector_scatter(resume_text: str, jd_text: str) -> None:
    """Visualize BGE vectors of requirements and evidence in 2D space."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    resume_sections = split_resume_sections(resume_text)
    jd_requirements = split_jd_requirements(jd_text)

    requirement_texts = [r["text"] for r in jd_requirements if r.get("text")]
    evidence_texts = []
    for section, items in resume_sections.items():
        for item in items:
            if item.strip():
                evidence_texts.append(item.strip())

    all_texts = requirement_texts + evidence_texts
    if len(all_texts) < 4:
        print("[SKIP] Too few texts for scatter plot.")
        return

    embeddings = encode_texts(all_texts)

    # Use PCA if too few samples for t-SNE
    n_samples = len(all_texts)
    if n_samples >= 10:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, n_samples - 1))
        coords = reducer.fit_transform(embeddings)
        method_name = "t-SNE"
    else:
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
        method_name = "PCA"

    n_req = len(requirement_texts)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(coords[:n_req, 0], coords[:n_req, 1],
               c="red", marker="^", s=120, label="JD 要求", zorder=5, edgecolors="darkred")
    ax.scatter(coords[n_req:, 0], coords[n_req:, 1],
               c="steelblue", marker="o", s=80, label="简历证据", zorder=4, edgecolors="navy", alpha=0.7)

    # Annotate requirement points
    for i, text in enumerate(requirement_texts):
        ax.annotate(text[:15], (coords[i, 0], coords[i, 1]),
                    fontsize=7, ha="left", va="bottom", color="darkred")

    ax.set_title(f"BGE 语义向量 {method_name} 降维可视化\n（红色三角=JD要求，蓝色圆点=简历证据）", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_vector_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 03_vector_scatter.png")


# ============================================================
# Chart 4: TF-IDF vs BGE Comparison (6 eval cases)
# ============================================================
def generate_tfidf_vs_bge_chart(cases: list[dict]) -> None:
    """Bar chart comparing TF-IDF baseline and BGE semantic scores for all eval cases."""
    names = []
    tfidf_scores = []
    bge_scores = []

    for case in cases:
        result = calculate_match(case["resume"], case["jd"])
        baseline = result["tfidf_baseline"]
        tfidf_val = float(baseline.get("tfidf_similarity", baseline.get("baseline_score", 0.0)))
        bge_val = float(result["dimension_scores"]["语义证据匹配"])

        # Short name for display
        name = case["case_name"]
        if "：" in name:
            name = name.split("：")[0]
        names.append(name)
        tfidf_scores.append(tfidf_val)
        bge_scores.append(bge_val)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, tfidf_scores, width, label="TF-IDF Baseline", color="#f4a261")
    bars2 = ax.bar(x + width / 2, bge_scores, width, label="BGE 语义模型", color="#2a9d8f")

    ax.set_xlabel("实验样例", fontsize=11)
    ax.set_ylabel("分数", fontsize=11)
    ax.set_title("TF-IDF Baseline vs BGE 语义模型 对比", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_tfidf_vs_bge.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 04_tfidf_vs_bge.png")


# ============================================================
# Chart 5: Four-Dimension Radar Chart
# ============================================================
def generate_radar_chart(resume_text: str, jd_text: str) -> None:
    """Radar chart showing the four scoring dimensions."""
    result = calculate_match(resume_text, jd_text)
    dimensions = result["dimension_scores"]

    labels = list(dimensions.keys())
    values = [float(v) for v in dimensions.values()]

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, "o-", linewidth=2, color="#264653")
    ax.fill(angles_closed, values_closed, alpha=0.25, color="#2a9d8f")

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_title(f"四维匹配得分雷达图\n综合分: {result['total_score']:.0f}  等级: {result['match_level']}", fontsize=13, pad=20)

    # Add score values on each vertex
    for angle, value, label in zip(angles, values, labels):
        ax.annotate(f"{value:.1f}", xy=(angle, value), xytext=(5, 5),
                    textcoords="offset points", fontsize=10, color="#e76f51", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 05_radar_chart.png")


# ============================================================
# Chart 6: Overall Score Comparison (6 eval cases)
# ============================================================
def generate_score_comparison(cases: list[dict]) -> None:
    """Horizontal bar chart showing total scores and match levels for all eval cases."""
    names = []
    total_scores = []
    levels = []
    colors = []

    level_color_map = {
        "高度匹配": "#2a9d8f",
        "较高匹配": "#264653",
        "一般匹配": "#e9c46a",
        "匹配度较低": "#e76f51",
    }

    for case in cases:
        result = calculate_match(case["resume"], case["jd"])
        name = case["case_name"]
        if "：" in name:
            name = name.split("：")[1] if len(name.split("：")[1]) <= 16 else name.split("：")[0]
        names.append(name)
        total_scores.append(float(result["total_score"]))
        levels.append(result["match_level"])
        colors.append(level_color_map.get(result["match_level"], "#999999"))

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(names))
    bars = ax.barh(y, total_scores, color=colors, edgecolor="white", height=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("综合匹配分", fontsize=11)
    ax.set_title("6组实验样例 综合匹配度对比", fontsize=13)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)

    # Score and level labels
    for i, (bar, score, level) in enumerate(zip(bars, total_scores, levels)):
        ax.text(score + 1, i, f"{score:.0f} ({level})", va="center", fontsize=9)

    # Add threshold lines
    ax.axvline(x=85, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=70, color="blue", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=50, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(85, len(names) - 0.5, "高度匹配", fontsize=8, color="green", ha="center")
    ax.text(70, len(names) - 0.5, "较高匹配", fontsize=8, color="blue", ha="center")
    ax.text(50, len(names) - 0.5, "一般匹配", fontsize=8, color="orange", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_score_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 06_score_comparison.png")


# ============================================================
# Chart 7: Skill Coverage Pie Chart (by category)
# ============================================================
def generate_skill_coverage(resume_text: str, jd_text: str) -> None:
    """Pie/donut chart showing JD skill coverage by category."""
    skill_result = extract_skills(resume_text, jd_text)
    matched_by_cat = skill_result["matched_skills_by_category"]
    missing_by_cat = skill_result["missing_skills_by_category"]

    categories = []
    matched_counts = []
    missing_counts = []

    for cat in matched_by_cat:
        m = len(matched_by_cat[cat])
        mi = len(missing_by_cat.get(cat, []))
        if m + mi > 0:
            categories.append(cat)
            matched_counts.append(m)
            missing_counts.append(mi)

    if not categories:
        print("[SKIP] No skill data for coverage chart.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Left: overall coverage donut
    total_matched = sum(matched_counts)
    total_missing = sum(missing_counts)
    sizes = [total_matched, total_missing]
    labels = [f"命中 ({total_matched})", f"缺失 ({total_missing})"]
    colors_pie = ["#2a9d8f", "#e76f51"]
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels, colors=colors_pie, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75, textprops={"fontsize": 11}
    )
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    axes[0].add_artist(centre_circle)
    axes[0].set_title("JD 技能总体覆盖率", fontsize=13)

    # Right: stacked bar by category
    x = np.arange(len(categories))
    axes[1].bar(x, matched_counts, color="#2a9d8f", label="命中")
    axes[1].bar(x, missing_counts, bottom=matched_counts, color="#e76f51", label="缺失")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, fontsize=9, rotation=20, ha="right")
    axes[1].set_ylabel("技能数量", fontsize=10)
    axes[1].set_title("各类别技能命中/缺失分布", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_skill_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 07_skill_coverage.png")


# ============================================================
# Chart 8: Evidence Match Status Distribution
# ============================================================
def generate_match_status_distribution(resume_text: str, jd_text: str) -> None:
    """Bar chart showing how many JD requirements fall into each match status."""
    from core.preprocess import split_resume_sections, split_jd_requirements
    from core.evidence_matcher import match_evidence_with_status

    resume_sections = split_resume_sections(resume_text)
    jd_requirements = split_jd_requirements(jd_text)
    evidence_result = match_evidence_with_status(jd_requirements, resume_sections)
    matches = evidence_result["matches"]

    if not matches:
        print("[SKIP] No evidence matches for status distribution.")
        return

    status_counts = {"高度匹配": 0, "部分匹配": 0, "匹配不足": 0}
    for m in matches:
        s = m.get("match_status", "匹配不足")
        status_counts[s] = status_counts.get(s, 0) + 1

    statuses = list(status_counts.keys())
    counts = list(status_counts.values())
    colors_bar = ["#2a9d8f", "#e9c46a", "#e76f51"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(statuses, counts, color=colors_bar, edgecolor="white", width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(count), ha="center", fontsize=12, fontweight="bold")

    ax.set_xlabel("匹配状态", fontsize=11)
    ax.set_ylabel("JD 要求条数", fontsize=11)
    ax.set_title("JD 各项要求的匹配状态分布", fontsize=13)
    ax.set_ylim(0, max(counts) + 2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_match_status_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 08_match_status_distribution.png")


# ============================================================
# Chart 9: Similarity Score Distribution Histogram
# ============================================================
def generate_similarity_histogram(resume_text: str, jd_text: str) -> None:
    """Histogram of all pairwise similarity scores between requirements and evidence."""
    resume_sections = split_resume_sections(resume_text)
    jd_requirements = split_jd_requirements(jd_text)

    requirement_texts = [r["text"] for r in jd_requirements if r.get("text")]
    evidence_texts = []
    for section, items in resume_sections.items():
        for item in items:
            if item.strip():
                evidence_texts.append(item.strip())

    if not requirement_texts or not evidence_texts:
        print("[SKIP] Empty data for similarity histogram.")
        return

    req_embeddings = encode_texts(requirement_texts)
    evi_embeddings = encode_texts(evidence_texts)
    sim_matrix = cosine_similarity_matrix(req_embeddings, evi_embeddings)
    all_scores = sim_matrix.flatten()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(all_scores, bins=25, color="#264653", edgecolor="white", alpha=0.85)
    ax.axvline(x=0.75, color="green", linestyle="--", linewidth=1.5, label="高度匹配阈值 (0.75)")
    ax.axvline(x=0.55, color="orange", linestyle="--", linewidth=1.5, label="部分匹配阈值 (0.55)")
    ax.axvline(x=float(np.mean(all_scores)), color="red", linestyle="-", linewidth=1.5,
               label=f"均值 ({np.mean(all_scores):.3f})")

    ax.set_xlabel("余弦相似度", fontsize=11)
    ax.set_ylabel("频次", fontsize=11)
    ax.set_title("所有 JD要求×简历证据 相似度分数分布", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_similarity_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 09_similarity_histogram.png")


# ============================================================
# Chart 10: Stacked Dimension Scores (6 eval cases)
# ============================================================
def generate_dimension_stacked(cases: list[dict]) -> None:
    """Stacked bar chart decomposing total score into four dimensions for each case."""
    names = []
    dim_data: dict[str, list[float]] = {
        "技能实体匹配": [], "语义证据匹配": [], "项目经历匹配": [], "软技能匹配": []
    }
    weights = {"技能实体匹配": 0.30, "语义证据匹配": 0.40, "项目经历匹配": 0.20, "软技能匹配": 0.10}

    for case in cases:
        result = calculate_match(case["resume"], case["jd"])
        name = case["case_name"]
        if "：" in name:
            name = name.split("：")[0]
        names.append(name)
        for dim in dim_data:
            raw_score = float(result["dimension_scores"].get(dim, 0))
            dim_data[dim].append(raw_score * weights[dim])

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 6))

    colors_stack = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
    bottom = np.zeros(len(names))
    for i, (dim, values) in enumerate(dim_data.items()):
        vals = np.array(values)
        ax.bar(x, vals, bottom=bottom, color=colors_stack[i], label=f"{dim} (×{weights[dim]:.0%})", width=0.55)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("加权得分贡献", fontsize=11)
    ax.set_title("6组样例 四维度加权贡献堆叠图", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Total score label on top
    for i, total in enumerate(bottom):
        ax.text(i, total + 1, f"{total:.0f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_dimension_stacked.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] 10_dimension_stacked.png")


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=" * 50)
    print("Resume-JD Matcher Visualization Script")
    print("=" * 50)

    # Check model status
    status = get_embedding_status()
    print(f"BGE Model: {'OK' if status['available'] else 'UNAVAILABLE'} ({status['mode']})")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    if not status["available"]:
        print("[ERROR] BGE model not available. Cannot generate visualizations.")
        sys.exit(1)

    # Load data
    resume_text, jd_text = load_sample_data()
    cases = load_eval_cases()

    # Generate charts
    print("--- Generating charts ---")
    print()

    print("[1/10] Word cloud...")
    generate_wordcloud(resume_text, jd_text)

    print("[2/10] Similarity heatmap...")
    generate_similarity_heatmap(resume_text, jd_text)

    print("[3/10] Vector scatter (t-SNE/PCA)...")
    generate_vector_scatter(resume_text, jd_text)

    print("[4/10] TF-IDF vs BGE comparison...")
    generate_tfidf_vs_bge_chart(cases)

    print("[5/10] Radar chart...")
    generate_radar_chart(resume_text, jd_text)

    print("[6/10] Score comparison...")
    generate_score_comparison(cases)

    print("[7/10] Skill coverage...")
    generate_skill_coverage(resume_text, jd_text)

    print("[8/10] Match status distribution...")
    generate_match_status_distribution(resume_text, jd_text)

    print("[9/10] Similarity histogram...")
    generate_similarity_histogram(resume_text, jd_text)

    print("[10/10] Dimension stacked chart...")
    generate_dimension_stacked(cases)

    print()
    print("=" * 50)
    print(f"All charts saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
