"""Session 12-compatible entry point for the Hybrid RAG engine.

The course lab keeps routing, Vector RAG, Graph RAG, and orchestration in one
file. This project uses the same flow but splits those responsibilities into
small modules that are easier to test and maintain. The public classes are
re-exported here so the connection to Session 12 remains explicit.
"""

from hybrid_rag import GraphRAG, HybridRAG, QuestionRouter

__all__ = ["GraphRAG", "HybridRAG", "QuestionRouter"]
