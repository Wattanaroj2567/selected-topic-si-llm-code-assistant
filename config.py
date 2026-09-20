"""Shared local runtime settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / ".rag_data"
VECTOR_DIR = DATA_DIR / "vector"
GRAPH_DIR = DATA_DIR / "graph"

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5-coder:7b"
OLLAMA_TIMEOUT_SECONDS = 180

VECTOR_TOP_K = 5
VECTOR_MIN_SCORE = 0.25
