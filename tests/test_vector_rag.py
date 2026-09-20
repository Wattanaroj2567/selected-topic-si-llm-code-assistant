from pathlib import Path

import numpy as np
import pytest

from ast_parser import parse_codebase
from rag_pipeline import VectorRAG, build_vector_index

PROJECT_ROOT = Path(__file__).parents[1]
DATASET_ROOT = PROJECT_ROOT / "test_codebase"


class SemanticTestClient:
    model = "semantic-test"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "mean" in lowered:
                vector = [1.0, 0.0, 0.0]
            elif "pipeline" in lowered:
                vector = [0.0, 1.0, 0.0]
            elif "file:" in lowered:
                vector = [0.0, 0.0, 1.0]
            else:
                vector = [0.0, 0.0, 0.0]
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)

    def chat(self, *, system: str, user: str) -> str:
        if "def mean" not in system or "metrics.py" not in system:
            raise AssertionError("retrieved mean source was not included in the prompt")
        return "The mean function adds the values and divides by their count."


class ThaiSemanticTestClient(SemanticTestClient):
    def chat(self, *, system: str, user: str) -> str:
        if (
            "Respond entirely in natural Thai" not in system
            or "Do not output Chinese, Japanese, or Korean" not in system
        ):
            raise AssertionError("vector prompt did not enforce Thai-only output")
        if "def mean" not in system or "metrics.py" not in system:
            raise AssertionError("retrieved mean source was not included in the prompt")
        return "ฟังก์ชัน mean หาผลรวมแล้วหารด้วยจำนวนข้อมูล"


class RetryingThaiSemanticTestClient(SemanticTestClient):
    def __init__(self) -> None:
        self.chat_calls = 0

    def chat(self, *, system: str, user: str) -> str:
        self.chat_calls += 1
        if self.chat_calls == 1:
            return "mean 函数计算平均值"
        if "The previous answer contained forbidden scripts" not in system:
            raise AssertionError("retry prompt did not explain the language violation")
        return "ฟังก์ชัน mean คำนวณค่าเฉลี่ย"


class MixedScriptThaiSemanticTestClient(SemanticTestClient):
    def __init__(self) -> None:
        self.chat_calls = 0

    def chat(self, *, system: str, user: str) -> str:
        self.chat_calls += 1
        if self.chat_calls > 1:
            raise AssertionError("a clean Thai paragraph should not require a retry")
        return (
            "คำตอบก่อนหน้า含有中文จึงใช้ไม่ได้\n\n"
            "ฟังก์ชัน mean คำนวณค่าเฉลี่ยจากผลรวมหารด้วยจำนวนข้อมูล"
        )


def test_vector_rag_answers_from_retrieved_chunk_and_returns_real_source(
    tmp_path: Path,
) -> None:
    client = SemanticTestClient()
    vector_dir = tmp_path / "vector"
    parsed = parse_codebase(DATASET_ROOT)
    build_vector_index(parsed.chunks, vector_dir, client)
    rag = VectorRAG(vector_dir, client=client, minimum_score=0.5)

    result = rag.query("What does the mean function do?", k=3)

    assert result["answer"] == (
        "The mean function adds the values and divides by their count."
    )
    assert result["sources"]
    mean_source = next(
        source for source in result["sources"] if source["symbol"] == "metrics.mean"
    )
    assert mean_source["file"] == "metrics.py"
    assert mean_source["line"] > 0
    assert "def mean" in mean_source["snippet"]
    assert mean_source["score"] >= 0.5


def test_vector_rag_returns_no_result_without_calling_chat_model(
    tmp_path: Path,
) -> None:
    client = SemanticTestClient()
    vector_dir = tmp_path / "vector"
    build_vector_index(parse_codebase(DATASET_ROOT).chunks, vector_dir, client)
    rag = VectorRAG(vector_dir, client=client, minimum_score=0.5)

    result = rag.query("quantum zebras", k=3)

    assert "couldn't find relevant code" in result["answer"]
    assert result["sources"] == []


def test_vector_rag_reports_missing_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"Run setup\.py first"):
        VectorRAG(tmp_path / "missing", client=SemanticTestClient())


def test_vector_rag_answers_thai_question_from_faiss_in_thai(
    tmp_path: Path,
) -> None:
    client = ThaiSemanticTestClient()
    vector_dir = tmp_path / "vector"
    build_vector_index(parse_codebase(DATASET_ROOT).chunks, vector_dir, client)
    rag = VectorRAG(vector_dir, client=client, minimum_score=0.5)

    result = rag.query("ฟังก์ชัน mean ทำงานอย่างไร", k=3)

    assert result["answer"] == "ฟังก์ชัน mean หาผลรวมแล้วหารด้วยจำนวนข้อมูล"
    assert "metrics.mean" in {source["symbol"] for source in result["sources"]}


def test_vector_rag_reports_no_thai_result_in_thai(tmp_path: Path) -> None:
    client = ThaiSemanticTestClient()
    vector_dir = tmp_path / "vector"
    build_vector_index(parse_codebase(DATASET_ROOT).chunks, vector_dir, client)
    rag = VectorRAG(vector_dir, client=client, minimum_score=0.5)

    result = rag.query("ม้าลายควอนตัม", k=3)

    assert result["answer"] == "ไม่พบโค้ดที่เกี่ยวข้องในโปรเจกต์ที่ index ไว้"
    assert result["sources"] == []


def test_vector_rag_retries_when_thai_answer_contains_chinese(tmp_path: Path) -> None:
    client = RetryingThaiSemanticTestClient()
    vector_dir = tmp_path / "vector"
    build_vector_index(parse_codebase(DATASET_ROOT).chunks, vector_dir, client)
    rag = VectorRAG(vector_dir, client=client, minimum_score=0.5)

    result = rag.query("ฟังก์ชัน mean ทำงานอย่างไร", k=3)

    assert result["answer"] == "ฟังก์ชัน mean คำนวณค่าเฉลี่ย"
    assert client.chat_calls == 2


def test_vector_rag_keeps_clean_thai_paragraph_without_retry(tmp_path: Path) -> None:
    client = MixedScriptThaiSemanticTestClient()
    vector_dir = tmp_path / "vector"
    build_vector_index(parse_codebase(DATASET_ROOT).chunks, vector_dir, client)
    rag = VectorRAG(vector_dir, client=client, minimum_score=0.5)

    result = rag.query("ฟังก์ชัน mean ทำงานอย่างไร", k=3)

    assert result["answer"] == ("ฟังก์ชัน mean คำนวณค่าเฉลี่ยจากผลรวมหารด้วยจำนวนข้อมูล")
    assert client.chat_calls == 1
