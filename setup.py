"""CLI entry point for indexing a Python codebase."""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from ast_parser import parse_codebase
from code_graph import build_graph
from config import DATA_DIR
from ollama_client import OllamaClient, OllamaError
from rag_pipeline import build_vector_index


class EmbeddingClient(Protocol):
    model: str

    def embed(self, texts: list[str]): ...


@dataclass(frozen=True)
class IndexStats:
    project_path: str
    files_scanned: int
    files_indexed: int
    chunks: int
    graph_nodes: int
    graph_edges: int
    warnings: tuple[str, ...]
    elapsed_seconds: float


def index_project(
    project_path: str | Path,
    *,
    data_dir: str | Path,
    embedding_client: EmbeddingClient | None = None,
) -> IndexStats:
    """Index one supplied project into vector and graph stores."""
    started = time.perf_counter()
    project = Path(project_path).expanduser().resolve()
    output = Path(data_dir).expanduser().resolve()
    parsed = parse_codebase(project)
    if not parsed.chunks:
        raise ValueError(f"No Python code chunks found in {project}")

    client = embedding_client or OllamaClient()
    output.mkdir(parents=True, exist_ok=True)
    vector_stats = build_vector_index(parsed.chunks, output / "vector", client)
    graph_stats = build_graph(parsed, output / "graph")
    warnings = tuple(f"{item.file}: {item.message}" for item in parsed.warnings)
    stats = IndexStats(
        project_path=str(project),
        files_scanned=parsed.files_scanned,
        files_indexed=parsed.files_parsed,
        chunks=vector_stats.chunks,
        graph_nodes=graph_stats.nodes,
        graph_edges=graph_stats.edges,
        warnings=warnings,
        elapsed_seconds=time.perf_counter() - started,
    )
    (output / "index_metadata.json").write_text(
        json.dumps(
            {
                "project_path": stats.project_path,
                "indexed_at": datetime.now(UTC).isoformat(),
                "files_scanned": stats.files_scanned,
                "files_indexed": stats.files_indexed,
                "chunks": stats.chunks,
                "graph_nodes": stats.graph_nodes,
                "graph_edges": stats.graph_edges,
                "embedding_model": vector_stats.model,
                "embedding_dimensions": vector_stats.dimensions,
                "warnings": list(stats.warnings),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index a Python codebase for the Local Coding Assistant"
    )
    parser.add_argument("--project", required=True, help="Python project directory")
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Generated index directory (default: .rag_data)",
    )
    args = parser.parse_args(argv)

    print(f"Indexing: {Path(args.project).expanduser().resolve()}")
    try:
        stats = index_project(args.project, data_dir=args.data_dir)
    except (OSError, ValueError, OllamaError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    for warning in stats.warnings:
        print(f"WARNING: {warning}")
    print("Index complete")
    print(f"  Files: {stats.files_indexed}/{stats.files_scanned}")
    print(f"  Chunks: {stats.chunks}")
    print(f"  Graph nodes: {stats.graph_nodes}")
    print(f"  Graph edges: {stats.graph_edges}")
    print(f"  Time: {stats.elapsed_seconds:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
