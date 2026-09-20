"""Build and query the Kuzu code relationship graph."""

import gc
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import kuzu

from ast_parser import ParsedCodebase


@dataclass(frozen=True)
class GraphStats:
    local_modules: int
    external_modules: int
    symbols: int
    define_edges: int
    contains_edges: int
    import_edges: int
    call_edges: int

    @property
    def nodes(self) -> int:
        return self.local_modules + self.external_modules + self.symbols

    @property
    def edges(self) -> int:
        return (
            self.define_edges
            + self.contains_edges
            + self.import_edges
            + self.call_edges
        )


def build_graph(parsed: ParsedCodebase, graph_path: str | Path) -> GraphStats:
    """Create a fresh Kuzu database from a parsed codebase."""
    path = Path(graph_path).expanduser().resolve()
    _reset_graph_path(path)
    path.mkdir(parents=True, exist_ok=True)

    external_ids = sorted(
        {
            relationship.imported_id
            for relationship in parsed.imports
            if relationship.is_external
        }
    )
    module_by_id = {module.id: module for module in parsed.modules}

    database = kuzu.Database(str(_database_file(path)))
    connection = kuzu.Connection(database)
    try:
        connection.execute(
            "CREATE NODE TABLE Module("
            "id STRING, name STRING, search_id STRING, search_name STRING, "
            "file STRING, external BOOLEAN, PRIMARY KEY(id))"
        )
        connection.execute(
            "CREATE NODE TABLE Symbol("
            "id STRING, name STRING, search_id STRING, search_name STRING, "
            "kind STRING, file STRING, line INT64, end_line INT64, parent_id STRING, "
            "PRIMARY KEY(id))"
        )
        connection.execute("CREATE REL TABLE DEFINES(FROM Module TO Symbol)")
        connection.execute("CREATE REL TABLE CONTAINS(FROM Symbol TO Symbol)")
        connection.execute(
            "CREATE REL TABLE IMPORTS("
            "FROM Module TO Module, imported_name STRING, line INT64)"
        )
        connection.execute("CREATE REL TABLE CALLS(FROM Symbol TO Symbol, line INT64)")

        for module in parsed.modules:
            _create_module(
                connection,
                module.id,
                module.name,
                module.file,
                external=False,
            )
        for module_id in external_ids:
            _create_module(
                connection,
                module_id,
                module_id.rsplit(".", 1)[-1],
                "",
                external=True,
            )

        for symbol in parsed.symbols:
            connection.execute(
                "CREATE (:Symbol {"
                "id: $id, name: $name, search_id: $search_id, "
                "search_name: $search_name, kind: $kind, file: $file, "
                "line: $line, end_line: $end_line, parent_id: $parent_id})",
                {
                    "id": symbol.id,
                    "name": symbol.name,
                    "search_id": symbol.id.lower(),
                    "search_name": symbol.name.lower(),
                    "kind": symbol.kind,
                    "file": symbol.file,
                    "line": symbol.line,
                    "end_line": symbol.end_line,
                    "parent_id": symbol.parent_id or "",
                },
            )
            connection.execute(
                "MATCH (m:Module), (s:Symbol) "
                "WHERE m.id = $module_id AND s.id = $symbol_id "
                "CREATE (m)-[:DEFINES]->(s)",
                {
                    "module_id": _symbol_module(symbol.id, module_by_id),
                    "symbol_id": symbol.id,
                },
            )

        contains_edges = 0
        for symbol in parsed.symbols:
            if not symbol.parent_id:
                continue
            connection.execute(
                "MATCH (parent:Symbol), (child:Symbol) "
                "WHERE parent.id = $parent_id AND child.id = $child_id "
                "CREATE (parent)-[:CONTAINS]->(child)",
                {"parent_id": symbol.parent_id, "child_id": symbol.id},
            )
            contains_edges += 1

        for relationship in parsed.imports:
            connection.execute(
                "MATCH (source:Module), (target:Module) "
                "WHERE source.id = $source_id AND target.id = $target_id "
                "CREATE (source)-[:IMPORTS {imported_name: $name, line: $line}]->(target)",
                {
                    "source_id": relationship.importer_id,
                    "target_id": relationship.imported_id,
                    "name": relationship.imported_name,
                    "line": relationship.line,
                },
            )

        for relationship in parsed.calls:
            connection.execute(
                "MATCH (source:Symbol), (target:Symbol) "
                "WHERE source.id = $source_id AND target.id = $target_id "
                "CREATE (source)-[:CALLS {line: $line}]->(target)",
                {
                    "source_id": relationship.caller_id,
                    "target_id": relationship.callee_id,
                    "line": relationship.line,
                },
            )
    finally:
        connection.close()
        del connection
        del database
        gc.collect()

    return GraphStats(
        local_modules=len(parsed.modules),
        external_modules=len(external_ids),
        symbols=len(parsed.symbols),
        define_edges=len(parsed.symbols),
        contains_edges=contains_edges,
        import_edges=len(parsed.imports),
        call_edges=len(parsed.calls),
    )


class CodeGraph:
    def __init__(self, graph_path: str | Path) -> None:
        path = Path(graph_path).expanduser().resolve()
        database_file = _database_file(path)
        if not database_file.is_file():
            raise FileNotFoundError(
                f"Graph index not found at {path}. Run setup.py first."
            )
        self._database = kuzu.Database(str(database_file), read_only=True)
        self._connection = kuzu.Connection(self._database)

    def query_callers(self, name: str) -> list[dict]:
        return self._query(
            "MATCH (source:Symbol)-[relation:CALLS]->(target:Symbol) "
            "WHERE target.search_name = $name OR target.search_id = $name "
            "RETURN source.id AS source_id, source.name AS source_name, "
            "source.kind AS source_kind, source.file AS source_file, "
            "source.line AS source_line, source.end_line AS source_end_line, "
            "target.id AS target_id, "
            "target.name AS target_name, target.kind AS target_kind, "
            "target.file AS target_file, target.line AS target_line, "
            "target.end_line AS target_end_line, "
            "relation.line AS relationship_line "
            "ORDER BY source.id",
            name,
        )

    def query_callees(self, name: str) -> list[dict]:
        return self._query(
            "MATCH (source:Symbol)-[relation:CALLS]->(target:Symbol) "
            "WHERE source.search_name = $name OR source.search_id = $name "
            "RETURN source.id AS source_id, source.name AS source_name, "
            "source.kind AS source_kind, source.file AS source_file, "
            "source.line AS source_line, source.end_line AS source_end_line, "
            "target.id AS target_id, "
            "target.name AS target_name, target.kind AS target_kind, "
            "target.file AS target_file, target.line AS target_line, "
            "target.end_line AS target_end_line, "
            "relation.line AS relationship_line "
            "ORDER BY target.id",
            name,
        )

    def query_imports(self, name: str) -> list[dict]:
        return self._query(
            "MATCH (source:Module)-[relation:IMPORTS]->(target:Module) "
            "WHERE source.search_name = $name OR source.search_id = $name "
            "RETURN source.id AS source_id, source.name AS source_name, "
            "'module' AS source_kind, source.file AS source_file, "
            "relation.line AS source_line, relation.line AS source_end_line, "
            "target.id AS target_id, "
            "target.name AS target_name, 'module' AS target_kind, "
            "target.file AS target_file, 0 AS target_line, 0 AS target_end_line, "
            "relation.line AS relationship_line "
            "ORDER BY target.id",
            name,
        )

    def _query(self, statement: str, name: str) -> list[dict]:
        normalized = name.strip().removesuffix("()").lower()
        if not normalized:
            return []
        result = self._connection.execute(statement, {"name": normalized})
        return result.get_as_df().to_dict(orient="records")

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None
        self._database = None
        gc.collect()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _reset_graph_path(path: Path) -> None:
    if path == Path(path.anchor) or not path.name:
        raise ValueError(f"Refusing to replace unsafe graph path: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _database_file(path: Path) -> Path:
    return path / "code_graph.kuzu"


def _create_module(
    connection: kuzu.Connection,
    module_id: str,
    name: str,
    file: str,
    *,
    external: bool,
) -> None:
    connection.execute(
        "CREATE (:Module {"
        "id: $id, name: $name, search_id: $search_id, "
        "search_name: $search_name, file: $file, external: $external})",
        {
            "id": module_id,
            "name": name,
            "search_id": module_id.lower(),
            "search_name": name.lower(),
            "file": file,
            "external": external,
        },
    )


def _symbol_module(symbol_id: str, modules: dict[str, object]) -> str:
    matches = [
        module_id
        for module_id in modules
        if symbol_id == module_id or symbol_id.startswith(f"{module_id}.")
    ]
    if not matches:
        raise ValueError(f"No module owns symbol {symbol_id}")
    return max(matches, key=len)
