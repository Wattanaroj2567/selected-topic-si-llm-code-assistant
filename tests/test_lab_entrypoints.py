import importlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).parents[1]


def import_course_entrypoint(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        pytest.fail(f"Course entry point {module_name}.py is missing: {error}")


def test_lab_12_entrypoint_routes_thai_relationship_question() -> None:
    lab_12 = import_course_entrypoint("lab_12_hybrid_rag")

    assert lab_12.QuestionRouter().classify("ใครเรียก mean") == "graph"


def test_lab_13_assistant_respects_supplied_index_path(tmp_path: Path) -> None:
    lab_13 = import_course_entrypoint("lab_13_coding_assistant")
    missing_index = tmp_path / "missing-index"

    with pytest.raises(FileNotFoundError, match="missing-index"):
        lab_13.load_assistant(missing_index)


def test_lab_13_app_runs_the_real_streamlit_ui(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "missing-index"))

    try:
        app = AppTest.from_file(str(PROJECT_ROOT / "lab_13_app.py")).run(timeout=15)
    except FileNotFoundError as error:
        pytest.fail(f"Course Streamlit entry point is missing: {error}")

    assert not app.exception
    assert app.title[0].value == "Local Coding Assistant"
    assert any("Index unavailable" in item.value for item in app.error)
