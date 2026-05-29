from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_path(value: str) -> str:
    """Resolve a possibly-relative path against the project root."""
    p = Path(value)
    return str(p if p.is_absolute() else (BASE_DIR / p))


class Settings:
    # ---- Local LLM (Ollama, OpenAI-compatible endpoint) ----
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct")
    OLLAMA_MODEL_FALLBACK = os.getenv("OLLAMA_MODEL_FALLBACK", "qwen2.5:7b-instruct")

    # ---- Embeddings + Vector store (all local) ----
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    VECTOR_STORE_DIR = _resolve_path(os.getenv("VECTOR_STORE_DIR", "data/cache/vector_index"))
    # Chroma requires a 3–512 char collection name ([a-zA-Z0-9._-]).
    VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "knowledge_base")
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
    RAG_SCORE_FLOOR = float(os.getenv("RAG_SCORE_FLOOR", "0.75"))

    # ---- Agent loop ----
    MAX_TOOL_ITERS = int(os.getenv("MAX_TOOL_ITERS", "6"))

    # ---- Frontend ----
    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")


settings = Settings()
