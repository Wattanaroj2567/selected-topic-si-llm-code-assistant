"""FAISS vector index storage and vector RAG retrieval."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from ast_parser import CodeChunk
from config import VECTOR_MIN_SCORE, VECTOR_TOP_K
from ollama_client import OllamaClient


def is_thai_text(text: str) -> bool:
    return any("\u0e00" <= character <= "\u0e7f" for character in text)


def response_language_instruction(question: str) -> str:
    if is_thai_text(question):
        return (
            "The user asks in Thai. Respond entirely in natural Thai. "
            "Do not output Chinese, Japanese, or Korean characters. "
            "English is allowed only for function, class, module, and code "
            "identifiers. Do not discuss these language rules or invent "
            "implementations that are absent from the sources."
        )
    return (
        "Answer in the same language as the user's question and keep code "
        "identifiers unchanged."
    )


def contains_forbidden_script(text: str) -> bool:
    forbidden_ranges = (
        ("\u1100", "\u11ff"),
        ("\u3040", "\u30ff"),
        ("\u3130", "\u318f"),
        ("\u3400", "\u4dbf"),
        ("\u4e00", "\u9fff"),
        ("\uac00", "\ud7af"),
        ("\uf900", "\ufaff"),
    )
    return any(
        start <= character <= end
        for character in text
        for start, end in forbidden_ranges
    )


def clean_thai_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    clean = [
        paragraph.strip()
        for paragraph in paragraphs
        if is_thai_text(paragraph) and not contains_forbidden_script(paragraph)
    ]
    return "\n\n".join(paragraph for paragraph in clean if paragraph)


class ChatClient(Protocol):
    def chat(self, *, system: str, user: str) -> str: ...


def grounded_chat(client: ChatClient, *, system: str, question: str) -> str:
    answer = client.chat(system=system, user=question)
    if not is_thai_text(question) or not contains_forbidden_script(answer):
        return answer

    cleaned_answer = clean_thai_paragraphs(answer)
    if cleaned_answer:
        return cleaned_answer

    retry_system = (
        f"{system}\n\n"
        "The previous answer contained forbidden scripts. Rewrite the answer "
        "entirely in natural Thai without Chinese, Japanese, or Korean "
        "characters. Do not mention the correction or these language rules."
    )
    corrected = client.chat(system=retry_system, user=question)
    if not contains_forbidden_script(corrected):
        return corrected
    cleaned_correction = clean_thai_paragraphs(corrected)
    if not cleaned_correction:
        return "ระบบไม่สามารถสร้างคำตอบภาษาไทยที่ถูกต้องได้ กรุณาลองถามใหม่"
    return cleaned_correction


class EmbeddingClient(Protocol):
    model: str

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def chat(self, *, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class VectorStats:
    chunks: int
    dimensions: int
    model: str


class VectorRAG:
    def __init__(
        self,
        vector_dir: str | Path,
        *,
        client: EmbeddingClient | None = None,
        minimum_score: float = VECTOR_MIN_SCORE,
    ) -> None:
        self.vector_dir = Path(vector_dir).expanduser().resolve()
        index_path = self.vector_dir / "index.faiss"
        documents_path = self.vector_dir / "documents.json"
        metadata_path = self.vector_dir / "metadata.json"
        missing = [
            str(path)
            for path in (index_path, documents_path, metadata_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Vector index is incomplete ({', '.join(missing)}). Run setup.py first."
            )

        self.client = client or OllamaClient()
        self.minimum_score = minimum_score
        self.index = faiss.read_index(str(index_path))
        raw_documents = json.loads(documents_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(raw_documents, list):
            raise ValueError(  # noqa: TRY004 - keep index validation failures uniform
                "Vector documents.json must contain a list"
            )
        self.documents = [CodeChunk(**item) for item in raw_documents]
        if self.index.ntotal != len(self.documents):
            raise ValueError("FAISS index and document metadata counts do not match")
        if self.index.d != metadata.get("dimensions"):
            raise ValueError(
                "FAISS index and embedding metadata dimensions do not match"
            )
        if metadata.get("embedding_model") != self.client.model:
            raise ValueError(
                "Embedding model differs from the indexed model. Run setup.py again."
            )

    def query(self, question: str, *, k: int = VECTOR_TOP_K) -> dict:
        """Retrieve relevant chunks, augment a prompt, and ask the local LLM."""
        matches = self.search(question, k=k)
        if not matches:
            return {
                "answer": (
                    "ไม่พบโค้ดที่เกี่ยวข้องในโปรเจกต์ที่ index ไว้"
                    if is_thai_text(question)
                    else "I couldn't find relevant code in the indexed project."
                ),
                "sources": [],
            }

        context_sections = []
        sources = []
        for position, (chunk, score) in enumerate(matches, start=1):
            context_sections.append(
                f"[Source {position}] {chunk.file}:{chunk.line}-{chunk.end_line} "
                f"({chunk.symbol_id})\n{chunk.code}"
            )
            sources.append(
                {
                    "file": chunk.file,
                    "symbol": chunk.symbol_id,
                    "kind": chunk.kind,
                    "line": chunk.line,
                    "end_line": chunk.end_line,
                    "snippet": chunk.code,
                    "score": round(score, 4),
                }
            )

        system = (
            "You are a coding assistant. Answer only from the retrieved Python "
            "code below. If the code does not support a claim, say so. Be concise "
            "and mention function or class names when useful. "
            f"{response_language_instruction(question)}\n\n"
            + "\n\n".join(context_sections)
        )
        answer = grounded_chat(self.client, system=system, question=question)
        return {"answer": answer, "sources": sources}

    def search(
        self,
        question: str,
        *,
        k: int = VECTOR_TOP_K,
    ) -> list[tuple[CodeChunk, float]]:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        query_vector = np.asarray(self.client.embed([question]), dtype=np.float32)
        if query_vector.shape != (1, self.index.d):
            raise ValueError(
                f"Query embedding has shape {query_vector.shape}; expected (1, {self.index.d})"
            )
        normalized = np.ascontiguousarray(query_vector.copy(), dtype=np.float32)
        faiss.normalize_L2(normalized)
        scores, indexes = self.index.search(normalized, min(k, self.index.ntotal))
        return [
            (self.documents[int(index)], float(score))
            for score, index in zip(scores[0], indexes[0], strict=True)
            if index >= 0 and score >= self.minimum_score
        ]


def build_vector_index(
    chunks: list[CodeChunk],
    output_dir: str | Path,
    embedding_client: EmbeddingClient,
) -> VectorStats:
    """Embed code chunks and persist a cosine-similarity FAISS index."""
    if not chunks:
        raise ValueError("No Python code chunks found to index")
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)

    texts = [
        f"{chunk.kind} {chunk.symbol_id}\nFile: {chunk.file}\n{chunk.code}"
        for chunk in chunks
    ]
    embeddings = np.asarray(embedding_client.embed(texts), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding row count does not match code chunk count")
    if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be finite, non-empty vectors")

    normalized = np.ascontiguousarray(embeddings.copy(), dtype=np.float32)
    faiss.normalize_L2(normalized)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    faiss.write_index(index, str(path / "index.faiss"))
    (path / "documents.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "chunks": len(chunks),
                "dimensions": int(normalized.shape[1]),
                "embedding_model": embedding_client.model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return VectorStats(
        chunks=len(chunks),
        dimensions=int(normalized.shape[1]),
        model=embedding_client.model,
    )
