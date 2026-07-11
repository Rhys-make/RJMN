"""Run all evaluation cases and print a compact experiment table."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = ROOT / "myenv310" / "Scripts" / "python.exe"
if PROJECT_PYTHON.exists() and Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
    os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.matcher import calculate_match  # noqa: E402


def main() -> None:
    cases = json.loads((ROOT / "data" / "eval_cases.json").read_text(encoding="utf-8"))

    print("| 样例 | 预期等级 | TF-IDF baseline | BGE 语义证据分 | 综合分 | 匹配等级 | 语义模型 |")
    print("| --- | --- | ---: | ---: | ---: | --- | --- |")
    for case in cases:
        result = calculate_match(case["resume"], case["jd"])
        semantic_model = result.get("semantic_model", {})
        baseline = result["tfidf_baseline"]
        print(
            "| {case_name} | {expected_level} | {tfidf:.2f} | {semantic:.2f} | {total:.2f} | {level} | {mode} |".format(
                case_name=case["case_name"],
                expected_level=case["expected_level"],
                tfidf=float(baseline.get("tfidf_similarity", baseline.get("baseline_score", 0.0))),
                semantic=float(result["dimension_scores"]["语义证据匹配"]),
                total=float(result["total_score"]),
                level=result["match_level"],
                mode=semantic_model.get("mode", ""),
            )
        )


if __name__ == "__main__":
    main()
