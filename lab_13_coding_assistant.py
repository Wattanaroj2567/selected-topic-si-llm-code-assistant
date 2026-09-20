"""Session 13-compatible entry point for the local coding assistant.

The real implementation lives in the focused project modules. Use
``load_assistant()`` to open the generated Vector and Graph indexes, then call
``assistant.query(question)`` exactly as the Streamlit UI does.
"""

from pathlib import Path

from hybrid_rag import HybridRAG, QuestionRouter


def load_assistant(data_dir: str | Path | None = None) -> HybridRAG:
    """Load the real Hybrid RAG assistant from a generated index directory."""
    if data_dir is None:
        return HybridRAG.from_data_dir()
    return HybridRAG.from_data_dir(data_dir)


__all__ = ["HybridRAG", "QuestionRouter", "load_assistant"]
