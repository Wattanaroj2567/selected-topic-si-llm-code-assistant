"""Graph RAG and hybrid query orchestration."""

import json
import re
import time
from pathlib import Path
from typing import Protocol

from code_graph import CodeGraph
from config import DATA_DIR
from ollama_client import OllamaClient, OllamaError
from rag_pipeline import (
    VectorRAG,
    grounded_chat,
    is_thai_text,
    response_language_instruction,
)

THAI_HELP_REQUESTS = {
    "สวัสดี",
    "สวัสดีครับ",
    "สวัสดีค่ะ",
    "หวัดดี",
    "ถามอะไรได้บ้าง",
    "ช่วยแนะนำคำถาม",
    "ช่วยแนะนำคำถามหน่อย",
    "แนะนำคำถาม",
}

THAI_HELP_ANSWER = """สวัสดีครับ ผมช่วยตอบคำถามเกี่ยวกับ Python codebase ที่ index ไว้

ลองถามได้ เช่น:
- ฟังก์ชัน mean ทำงานอย่างไร
- อธิบายการทำงานของ run_pipeline
- ใครเรียก mean
- pipeline นำเข้าอะไรบ้าง"""


class ChatClient(Protocol):
    def chat(self, *, system: str, user: str) -> str: ...


class QuestionRouter:
    def classify(self, question: str) -> str:
        lowered = " ".join(question.lower().split())
        graph_phrases = (
            "who calls",
            "called by",
            "which functions call",
            "what functions call",
            "what functions does",
            "what does",
            " import",
            "imports ",
            "depend on",
            "depends on",
            "dependencies",
            "relationship",
            "callers",
            "callees",
            "ใครเรียก",
            "ฟังก์ชันไหนเรียก",
            "ฟังก์ชันใดเรียก",
            "เรียกอะไร",
            "เรียกฟังก์ชัน",
            "นำเข้า",
            "อิมพอร์ต",
            "ขึ้นกับ",
            "ขึ้นอยู่กับ",
            "พึ่งพา",
            "ความสัมพันธ์",
        )
        if any(phrase in lowered for phrase in graph_phrases):
            if "what does" in lowered and not re.search(
                r"what does\s+\w+(?:\.py)?\s+(?:call|import|depend)", lowered
            ):
                return "vector"
            return "graph"
        return "vector"


class HybridRAG:
    def __init__(
        self, vector_rag, graph_rag, router: QuestionRouter | None = None
    ) -> None:
        self.vector_rag = vector_rag
        self.graph_rag = graph_rag
        self.router = router or QuestionRouter()

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path = DATA_DIR,
        *,
        client: OllamaClient | None = None,
    ) -> "HybridRAG":
        path = Path(data_dir).expanduser().resolve()
        metadata_path = path / "index_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Index metadata not found at {metadata_path}. Run setup.py first."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        project_path = metadata.get("project_path")
        if not isinstance(project_path, str) or not project_path:
            raise ValueError("Index metadata does not contain a valid project_path")
        shared_client = client or OllamaClient()
        return cls(
            VectorRAG(path / "vector", client=shared_client),
            GraphRAG(path / "graph", client=shared_client, project_path=project_path),
        )

    def query(self, question: str, *, force_mode: str | None = None) -> dict:
        started = time.perf_counter()
        if force_mode not in {None, "vector", "graph"}:
            raise ValueError("force_mode must be 'vector', 'graph', or None")
        cleaned_question = question.strip()
        mode = force_mode or self.router.classify(cleaned_question)
        if not cleaned_question:
            return {
                "answer": "Please enter a question about the indexed codebase.",
                "mode": mode,
                "sources": [],
                "elapsed": time.perf_counter() - started,
                "error": "Question cannot be empty",
            }

        normalized_question = cleaned_question.rstrip("!?。.").strip()
        if force_mode is None and normalized_question in THAI_HELP_REQUESTS:
            return {
                "answer": THAI_HELP_ANSWER,
                "mode": "help",
                "sources": [],
                "elapsed": time.perf_counter() - started,
                "error": None,
            }

        backend = self.graph_rag if mode == "graph" else self.vector_rag
        try:
            result = backend.query(cleaned_question)
        except OllamaError as error:
            return {
                "answer": str(error),
                "mode": mode,
                "sources": [],
                "elapsed": time.perf_counter() - started,
                "error": str(error),
            }
        return {
            "answer": result["answer"],
            "mode": mode,
            "sources": result.get("sources", []),
            "elapsed": time.perf_counter() - started,
            "error": None,
        }

    def close(self) -> None:
        for backend in (self.vector_rag, self.graph_rag):
            close = getattr(backend, "close", None)
            if close:
                close()


class GraphRAG:
    def __init__(
        self,
        graph_dir: str | Path,
        *,
        client: ChatClient | None = None,
        project_path: str | Path,
    ) -> None:
        self.graph = CodeGraph(graph_dir)
        self.client = client or OllamaClient()
        self.project_path = Path(project_path).expanduser().resolve()

    def query(self, question: str) -> dict:
        """Retrieve one relationship type from Kuzu and ground an LLM answer."""
        entity = extract_graph_entity(question)
        relationship = classify_graph_relationship(question)
        if not entity:
            return {
                "answer": (
                    "ไม่พบชื่อฟังก์ชันหรือโมดูลในคำถาม Graph"
                    if is_thai_text(question)
                    else "I couldn't identify a function or module name in the graph question."
                ),
                "sources": [],
                "relationship": relationship,
                "entity": None,
            }

        if relationship == "callers":
            rows = self.graph.query_callers(entity)
        elif relationship == "imports":
            rows = self.graph.query_imports(entity)
        else:
            rows = self.graph.query_callees(entity)

        if not rows:
            thai_relationships = {
                "callers": "ผู้เรียก",
                "callees": "ฟังก์ชันที่ถูกเรียก",
                "imports": "การนำเข้า",
            }
            return {
                "answer": (
                    f"ไม่พบความสัมพันธ์{thai_relationships[relationship]}สำหรับ '{entity}'"
                    if is_thai_text(question)
                    else f"No {relationship} relationship found for '{entity}'."
                ),
                "sources": [],
                "relationship": relationship,
                "entity": entity,
            }

        relation_word = "IMPORTS" if relationship == "imports" else "CALLS"
        relationships = dict.fromkeys(
            f"- {row['source_id']} {relation_word} {row['target_id']}" for row in rows
        )
        context = "\n".join(relationships)
        system = (
            "You are a code graph assistant. Answer only from these Kuzu graph "
            "relationships. State the exact symbol names and do not invent edges. "
            f"{response_language_instruction(question)}\n\n"
            f"Relationship type: {relationship}\nEntity: {entity}\n{context}"
        )
        answer = grounded_chat(self.client, system=system, question=question)
        return {
            "answer": answer,
            "sources": self._sources(rows, relationship),
            "relationship": relationship,
            "entity": entity,
        }

    def close(self) -> None:
        self.graph.close()

    def _sources(self, rows: list[dict], relationship: str) -> list[dict]:
        prefix = "source" if relationship in {"callers", "imports"} else "target"
        sources: list[dict] = []
        seen: set[str] = set()
        for row in rows:
            symbol = str(row[f"{prefix}_id"])
            file = str(row[f"{prefix}_file"])
            line = int(row[f"{prefix}_line"])
            source_key = f"{symbol}:{line}" if relationship == "imports" else symbol
            if source_key in seen:
                continue
            seen.add(source_key)
            end_line = int(row.get(f"{prefix}_end_line", line) or line)
            sources.append(
                {
                    "file": file,
                    "symbol": symbol,
                    "kind": str(row[f"{prefix}_kind"]),
                    "line": line,
                    "end_line": end_line,
                    "snippet": self._read_snippet(file, line, end_line),
                    "relationship": relationship,
                }
            )
        return sources

    def _read_snippet(self, relative_file: str, line: int, end_line: int) -> str:
        if not relative_file or line < 1:
            return ""
        source_path = (self.project_path / relative_file).resolve()
        if (
            source_path != self.project_path
            and self.project_path not in source_path.parents
        ):
            return ""
        try:
            lines = source_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return ""
        bounded_end = min(max(end_line, line), len(lines))
        return "\n".join(lines[line - 1 : bounded_end])


def classify_graph_relationship(question: str) -> str:
    lowered = question.lower()
    if any(
        phrase in lowered
        for phrase in (
            "import",
            "depend",
            "นำเข้า",
            "อิมพอร์ต",
            "ขึ้นกับ",
            "ขึ้นอยู่กับ",
            "พึ่งพา",
        )
    ):
        return "imports"
    if (
        "who calls" in lowered
        or "called by" in lowered
        or re.search(r"(?:which|what) functions? call\b", lowered)
        or "ใครเรียก" in lowered
        or "ฟังก์ชันไหนเรียก" in lowered
        or "ฟังก์ชันใดเรียก" in lowered
    ):
        return "callers"
    return "callees"


def extract_graph_entity(question: str) -> str | None:
    """Extract the function/module named by common graph question forms."""
    parenthesized = re.search(r"\b([A-Za-z_]\w*)\s*\(\)", question)
    if parenthesized:
        return parenthesized.group(1)

    identifier = r"([A-Za-z_]\w*(?:\.py)?)"
    patterns = [
        rf"(?:ใคร|ฟังก์ชันไหน|ฟังก์ชันใด)\s*เรียก(?:ใช้)?\s*(?:ฟังก์ชัน\s*)?{identifier}",
        rf"{identifier}\s*(?:เรียก(?:ใช้)?(?:ฟังก์ชัน)?(?:อะไร|ใด)|นำเข้า|อิมพอร์ต|ขึ้น(?:อยู่)?กับ|พึ่งพา)",
        r"(?:who calls|which functions? call|what functions? call)\s+(?:the\s+)?([A-Za-z_]\w*)",
        r"(?:what functions? does|what does)\s+([A-Za-z_]\w*(?:\.py)?)\s+(?:call|import|depend)",
        r"([A-Za-z_]\w*)\s+(?:is\s+)?called by",
        r"(?:imports?|dependencies)\s+(?:of|for)\s+([A-Za-z_]\w*(?:\.py)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1).removesuffix(".py")
    return None
