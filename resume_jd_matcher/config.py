"""Project-level configuration for model backends and scoring."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = BASE_DIR / "models"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Put manually downloaded Hugging Face model files here.
# If these directories exist, the app loads them first and does not need network access.
EMBEDDING_MODEL_LOCAL_PATH = LOCAL_MODELS_DIR / "bge-small-zh-v1.5"
GENERATION_MODEL_LOCAL_PATH = LOCAL_MODELS_DIR / "Qwen2.5-1.5B-Instruct"

USE_GENERATION_MODEL = True
GENERATION_MAX_NEW_TOKENS = 768

# If local hardware is limited, change GENERATION_MODEL_NAME to:
# "Qwen/Qwen2.5-0.5B-Instruct"

HIGH_MATCH_THRESHOLD = 0.75
MEDIUM_MATCH_THRESHOLD = 0.55

EVIDENCE_TOP_K = 3
