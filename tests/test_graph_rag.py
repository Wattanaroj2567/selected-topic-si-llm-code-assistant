from pathlib import Path

import numpy as np

from ast_parser import parse_codebase
from code_graph import build_graph
from hybrid_rag import GraphRAG

PROJECT_ROOT = Path(__file__).parents[1]
DATASET_ROOT = PROJECT_ROOT / "test_codebase"


class GraphAnswerClient:
    model = "unused"

    def embed(self, texts: list[str]) -> np.ndarray:
        raise AssertionError("Graph RAG must not use vector embeddings")

    def chat(self, *, system: str, user: str) -> str:
        if "Who calls" in user:
            if (
                "metrics.variance" not in system
                or "reporting.build_report" not in system
            ):
                raise AssertionError("caller relationships were not in graph context")
            return "mean is called by variance and build_report."
        if "import" in user:
            if "pipeline IMPORTS data_loader" not in system:
                raise AssertionError("import relationships were not in graph context")
            if system.count("pipeline IMPORTS reporting") != 1:
                raise AssertionError(
                    "duplicate import relationships reached the prompt"
                )
            return "pipeline imports data_loader, pathlib, and reporting."
        if "run_pipeline" in user:
            if (
                "data_loader.load_numbers" not in system
                or "reporting.build_report" not in system
            ):
                raise AssertionError("callee relationships were not in graph context")
            return "run_pipeline calls load_numbers and build_report."
        raise AssertionError(f"Unexpected graph question: {user}")


class ThaiGraphAnswerClient:
    def chat(self, *, system: str, user: str) -> str:
        if (
            "Respond entirely in natural Thai" not in system
            or "Do not output Chinese, Japanese, or Korean" not in system
        ):
            raise AssertionError("graph prompt did not enforce Thai-only output")
        if "metrics.variance" not in system or "reporting.build_report" not in system:
            raise AssertionError("caller relationships were not in graph context")
        return "mean ถูกเรียกใช้โดย variance และ build_report"


class RetryingThaiGraphAnswerClient:
    def __init__(self) -> None:
        self.chat_calls = 0

    def chat(self, *, system: str, user: str) -> str:
        self.chat_calls += 1
        if self.chat_calls == 1:
            return "mean 被 variance 调用"
        if "The previous answer contained forbidden scripts" not in system:
            raise AssertionError("retry prompt did not explain the language violation")
        return "mean ถูกเรียกโดย variance และ build_report"


def test_graph_rag_answers_callers_and_callees_from_kuzu(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graph"
    build_graph(parse_codebase(DATASET_ROOT), graph_dir)
    rag = GraphRAG(
        graph_dir,
        client=GraphAnswerClient(),
        project_path=DATASET_ROOT,
    )

    callers = rag.query("Who calls the mean function?")
    callees = rag.query("What functions does run_pipeline call?")

    assert callers["answer"] == "mean is called by variance and build_report."
    assert callers["relationship"] == "callers"
    assert {source["symbol"] for source in callers["sources"]} == {
        "metrics.variance",
        "reporting.build_report",
    }
    assert all(source["snippet"] for source in callers["sources"])

    assert callees["answer"] == "run_pipeline calls load_numbers and build_report."
    assert callees["relationship"] == "callees"
    assert {source["symbol"] for source in callees["sources"]} == {
        "data_loader.load_numbers",
        "reporting.build_report",
    }

    rag.close()


def test_graph_rag_import_sources_keep_each_retrieved_import_line(
    tmp_path: Path,
) -> None:
    graph_dir = tmp_path / "graph"
    build_graph(parse_codebase(DATASET_ROOT), graph_dir)
    rag = GraphRAG(graph_dir, client=GraphAnswerClient(), project_path=DATASET_ROOT)

    result = rag.query("What does pipeline import?")

    assert result["relationship"] == "imports"
    assert result["answer"] == "pipeline imports data_loader, pathlib, and reporting."
    assert len(result["sources"]) == 3
    assert {source["file"] for source in result["sources"]} == {"pipeline.py"}
    assert len({source["line"] for source in result["sources"]}) == 3
    rag.close()


def test_graph_rag_handles_unknown_function_without_calling_llm(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graph"
    build_graph(parse_codebase(DATASET_ROOT), graph_dir)
    rag = GraphRAG(graph_dir, client=GraphAnswerClient(), project_path=DATASET_ROOT)

    result = rag.query("Who calls does_not_exist?")

    assert "No callers relationship found" in result["answer"]
    assert result["sources"] == []
    rag.close()


def test_graph_rag_answers_thai_question_from_kuzu_in_thai(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graph"
    build_graph(parse_codebase(DATASET_ROOT), graph_dir)
    rag = GraphRAG(
        graph_dir,
        client=ThaiGraphAnswerClient(),
        project_path=DATASET_ROOT,
    )

    result = rag.query("ใครเรียก mean")

    assert result["answer"] == "mean ถูกเรียกใช้โดย variance และ build_report"
    assert result["relationship"] == "callers"
    assert {source["symbol"] for source in result["sources"]} == {
        "metrics.variance",
        "reporting.build_report",
    }
    rag.close()


def test_graph_rag_reports_unknown_thai_symbol_in_thai(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graph"
    build_graph(parse_codebase(DATASET_ROOT), graph_dir)
    rag = GraphRAG(graph_dir, client=GraphAnswerClient(), project_path=DATASET_ROOT)

    result = rag.query("ใครเรียก does_not_exist")

    assert result["answer"] == "ไม่พบความสัมพันธ์ผู้เรียกสำหรับ 'does_not_exist'"
    assert result["sources"] == []
    rag.close()


def test_graph_rag_retries_when_thai_answer_contains_chinese(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graph"
    build_graph(parse_codebase(DATASET_ROOT), graph_dir)
    client = RetryingThaiGraphAnswerClient()
    rag = GraphRAG(graph_dir, client=client, project_path=DATASET_ROOT)

    result = rag.query("ใครเรียก mean")

    assert result["answer"] == "mean ถูกเรียกโดย variance และ build_report"
    assert client.chat_calls == 2
    rag.close()
