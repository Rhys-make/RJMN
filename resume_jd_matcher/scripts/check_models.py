"""Check whether local BGE and Qwen models are ready for the Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _check_files(path: Path, required_any: list[list[str]]) -> bool:
    print(f"目录: {path}")
    if not path.exists():
        print("状态: 缺失目录")
        return False

    ok = True
    for alternatives in required_any:
        found = [pattern for pattern in alternatives if list(path.glob(pattern))]
        if found:
            print(f"已找到: {' 或 '.join(found)}")
        else:
            print(f"缺失: {' 或 '.join(alternatives)}")
            ok = False
    return ok


def check_bge() -> bool:
    _print_header("BGE 语义模型检查")
    local_path = Path(config.EMBEDDING_MODEL_LOCAL_PATH)
    file_ok = _check_files(
        local_path,
        [
            ["config.json"],
            ["modules.json"],
            ["*.safetensors", "pytorch_model*.bin"],
            ["tokenizer.json", "vocab.txt", "sentencepiece.bpe.model"],
        ],
    )

    try:
        from sentence_transformers import SentenceTransformer

        source = str(local_path) if file_ok else config.EMBEDDING_MODEL_NAME
        load_from = "项目本地目录" if file_ok else "Hugging Face 本地缓存"
        print(f"尝试加载: {load_from} -> {source}")
        model = SentenceTransformer(source, local_files_only=True)
        embeddings = model.encode(
            ["熟悉 Python 和自然语言处理", "使用 Python 进行 NLP 文本处理"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        similarity = float(embeddings[0] @ embeddings[1])
        print(f"加载结果: 成功，向量维度 {embeddings.shape[1]}，测试相似度 {similarity:.4f}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"加载结果: 失败，{exc}")
        return False


def check_qwen() -> bool:
    _print_header("Qwen 生成模型检查")
    local_path = Path(config.GENERATION_MODEL_LOCAL_PATH)
    file_ok = _check_files(
        local_path,
        [
            ["config.json"],
            ["tokenizer.json", "tokenizer.model"],
            ["tokenizer_config.json"],
            ["model.safetensors", "*.safetensors", "pytorch_model*.bin"],
        ],
    )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = str(local_path) if file_ok else config.GENERATION_MODEL_NAME
        load_from = "项目本地目录" if file_ok else "Hugging Face 本地缓存"
        print(f"尝试加载: {load_from} -> {source}")
        kwargs = {"local_files_only": True}
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32, **kwargs)
        print(f"加载结果: 成功，词表大小 {len(tokenizer)}，参数设备 {next(model.parameters()).device}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"加载结果: 失败，{exc}")
        return False


def main() -> None:
    bge_ok = check_bge()
    qwen_ok = check_qwen()
    _print_header("结论")
    print(f"BGE 语义匹配: {'可用' if bge_ok else '不可用'}")
    print(f"Qwen 生成分析: {'可用' if qwen_ok else '不可用'}")
    if not bge_ok or not qwen_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
