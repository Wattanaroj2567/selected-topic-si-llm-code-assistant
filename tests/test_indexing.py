import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from setup import index_project

PROJECT_ROOT = Path(__file__).parents[1]
DATASET_ROOT = PROJECT_ROOT / "test_codebase"


class DeterministicEmbedder:
    model = "test-embedding"

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for index, text in enumerate(texts):
            rows.append(
                [
                    float(index + 1),
                    float(len(text)),
                    float(text.count("def ") + text.count("class ")),
                ]
            )
        return np.asarray(rows, dtype=np.float32)


def test_index_project_builds_faiss_json_and_kuzu_from_supplied_path(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "rag-data"

    stats = index_project(
        DATASET_ROOT,
        data_dir=data_dir,
        embedding_client=DeterministicEmbedder(),
    )

    vector_index = faiss.read_index(str(data_dir / "vector" / "index.faiss"))
    documents = json.loads(
        (data_dir / "vector" / "documents.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (data_dir / "index_metadata.json").read_text(encoding="utf-8")
    )

    assert stats.project_path == str(DATASET_ROOT.resolve())
    assert stats.files_scanned == 6
    assert stats.files_indexed == 6
    assert stats.chunks == len(documents)
    assert vector_index.ntotal == stats.chunks
    assert vector_index.d == 3
    assert stats.graph_nodes > stats.chunks
    assert stats.graph_edges > 0
    assert (data_dir / "graph").is_dir()
    assert metadata["project_path"] == str(DATASET_ROOT.resolve())
    assert metadata["embedding_model"] == "test-embedding"


def test_index_project_rejects_directory_without_python_chunks(tmp_path: Path) -> None:
    empty_project = tmp_path / "empty-project"
    empty_project.mkdir()

    with pytest.raises(ValueError, match="No Python code chunks"):
        index_project(
            empty_project,
            data_dir=tmp_path / "rag-data",
            embedding_client=DeterministicEmbedder(),
        )
