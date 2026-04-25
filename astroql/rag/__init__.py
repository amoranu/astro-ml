"""RAG subsystem (spec §6.5, §7).

Wraps astro-prod/rag_engine.py. If astro-prod is unavailable or corpora
not configured, retrieval returns []. This allows the full pipeline to
run offline in tests/CI.
"""
from .pipeline import RAGPipeline

__all__ = ["RAGPipeline"]
