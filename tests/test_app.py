from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from setup import index_project

APP_PATH = Path(__file__).parents[1] / "app.py"
DATASET_ROOT = APP_PATH.parent / "test_codebase"


class DeterministicEmbedder:
    model = "nomic-embed-text"

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [
                    float(index + 1),
                    float(len(text)),
                    float(text.count("def ") + text.count("class ")),
                ]
                for index, text in enumerate(texts)
            ],
            dtype=np.float32,
        )


def test_app_reports_missing_configured_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "missing-index"))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)

    assert not app.exception
    assert any("Index unavailable" in item.value for item in app.error)
    assert app.chat_input[0].disabled


def test_app_renders_chat_controls_and_index_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "rag-data"
    index_project(
        DATASET_ROOT,
        data_dir=data_dir,
        embedding_client=DeterministicEmbedder(),
    )
    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "Local Coding Assistant"
    assert app.chat_input[0].placeholder == "Ask about the indexed Python codebase..."
    assert any(button.label == "Clear Chat" for button in app.button)
    assert any("Index ready" in item.value for item in app.success)
    rendered_text = "\n".join(item.value for item in app.markdown)
    assert "ตัวอย่างคำถามภาษาไทย" in rendered_text
    assert "ฟังก์ชัน mean ทำงานอย่างไร" in rendered_text
    assert "ใครเรียก mean" in rendered_text


def test_clear_chat_removes_rendered_history(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "missing-index"))
    app = AppTest.from_file(str(APP_PATH))
    app.session_state["messages"] = [
        {"role": "user", "content": "Old question"},
        {
            "role": "assistant",
            "content": "Old answer",
            "mode": "vector",
            "sources": [],
            "elapsed": 0.1,
            "error": None,
        },
    ]
    app.run(timeout=15)
    assert any(item.value == "Old answer" for item in app.markdown)

    next(button for button in app.button if button.label == "Clear Chat").click().run(
        timeout=15
    )

    assert not any(item.value == "Old answer" for item in app.markdown)
