from pathlib import Path

import pytest

from ast_parser import parse_codebase
from code_graph import CodeGraph, build_graph

PROJECT_ROOT = Path(__file__).parents[1]
DATASET_ROOT = PROJECT_ROOT / "test_codebase"


def test_kuzu_graph_returns_real_call_and_import_relationships(tmp_path: Path) -> None:
    parsed = parse_codebase(DATASET_ROOT)
    graph_path = tmp_path / "graph"

    stats = build_graph(parsed, graph_path)

    assert stats.local_modules == parsed.files_parsed
    assert stats.symbols == len(parsed.symbols)
    assert stats.call_edges == len(parsed.calls)

    with CodeGraph(graph_path) as graph:
        callers = graph.query_callers("mean")
        callees = graph.query_callees("run_pipeline")
        imports = graph.query_imports("pipeline")

        assert {item["source_id"] for item in callers} == {
            "metrics.variance",
            "reporting.build_report",
        }
        assert {item["target_id"] for item in callees} == {
            "data_loader.load_numbers",
            "reporting.build_report",
        }
        assert {item["target_id"] for item in imports} == {
            "data_loader",
            "pathlib",
            "reporting",
        }
        assert all(item["source_file"] for item in callers + callees + imports)


def test_kuzu_graph_returns_empty_results_for_unknown_name(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph"
    build_graph(parse_codebase(DATASET_ROOT), graph_path)

    with CodeGraph(graph_path) as graph:
        assert graph.query_callers("does_not_exist") == []
        assert graph.query_callees("does_not_exist") == []
        assert graph.query_imports("does_not_exist") == []


def test_code_graph_reports_missing_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"Run setup\.py first"):
        CodeGraph(tmp_path / "missing")
