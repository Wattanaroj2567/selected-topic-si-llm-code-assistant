"""Parse Python source into vector chunks and code relationships."""

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CodeSymbol:
    id: str
    name: str
    qualified_name: str
    kind: str
    file: str
    line: int
    end_line: int
    parent_id: str | None = None


@dataclass(frozen=True)
class CodeModule:
    id: str
    name: str
    file: str


@dataclass(frozen=True)
class CodeChunk:
    symbol_id: str
    name: str
    kind: str
    file: str
    line: int
    end_line: int
    code: str


@dataclass(frozen=True)
class ImportRelationship:
    importer_id: str
    imported_id: str
    imported_name: str
    line: int
    is_external: bool


@dataclass(frozen=True)
class CallRelationship:
    caller_id: str
    callee_id: str
    line: int


@dataclass(frozen=True)
class ParseWarning:
    file: str
    message: str


@dataclass
class ParsedCodebase:
    root: str
    modules: list[CodeModule] = field(default_factory=list)
    symbols: list[CodeSymbol] = field(default_factory=list)
    chunks: list[CodeChunk] = field(default_factory=list)
    imports: list[ImportRelationship] = field(default_factory=list)
    calls: list[CallRelationship] = field(default_factory=list)
    warnings: list[ParseWarning] = field(default_factory=list)
    files_scanned: int = 0
    files_parsed: int = 0


def parse_codebase(project_path: str | Path) -> ParsedCodebase:
    """Parse every readable Python file below *project_path*.

    Syntax errors are reported as warnings so one broken file cannot prevent the
    rest of a codebase from being indexed. Only calls that resolve to symbols
    inside the indexed project become call-graph edges.
    """
    root = Path(project_path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project path is not a directory: {root}")

    python_files = sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file()
        and not any(
            part.startswith(".") or part == "__pycache__"
            for part in path.relative_to(root).parts[:-1]
        )
    )
    parsed = ParsedCodebase(root=str(root), files_scanned=len(python_files))
    sources: dict[str, str] = {}
    trees: dict[str, ast.Module] = {}

    for path in python_files:
        relative_file = path.relative_to(root).as_posix()
        module_id = _module_id(path, root)
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative_file)
        except (OSError, UnicodeError, SyntaxError) as error:
            parsed.warnings.append(
                ParseWarning(relative_file, f"{type(error).__name__}: {error}")
            )
            continue

        parsed.modules.append(
            CodeModule(
                id=module_id, name=module_id.rsplit(".", 1)[-1], file=relative_file
            )
        )
        sources[module_id] = source
        trees[module_id] = tree
        parsed.files_parsed += 1

    known_modules = set(trees)
    symbol_nodes: dict[str, ast.AST] = {}
    symbol_modules: dict[str, str] = {}
    symbol_qualnames: dict[str, str] = {}

    for module_id, tree in trees.items():
        _collect_symbols(
            module_id=module_id,
            body=tree.body,
            source=sources[module_id],
            relative_file=next(
                item.file for item in parsed.modules if item.id == module_id
            ),
            parsed=parsed,
            symbol_nodes=symbol_nodes,
            symbol_modules=symbol_modules,
            symbol_qualnames=symbol_qualnames,
        )

    import_aliases: dict[str, dict[str, str]] = {}
    module_aliases: dict[str, dict[str, str]] = {}
    seen_imports: set[tuple[str, str, str, int]] = set()

    for module_id, tree in trees.items():
        imported_symbols: dict[str, str] = {}
        imported_modules: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_id = alias.name
                    bound_name = alias.asname or alias.name.split(".", 1)[0]
                    imported_modules[bound_name] = imported_id
                    _append_import(
                        parsed,
                        seen_imports,
                        module_id,
                        imported_id,
                        alias.name,
                        node.lineno,
                        imported_id not in known_modules,
                    )
            elif isinstance(node, ast.ImportFrom):
                base_module = _resolve_import_from(module_id, node.module, node.level)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    possible_module = ".".join(
                        part for part in (base_module, alias.name) if part
                    )
                    bound_name = alias.asname or alias.name
                    if possible_module in known_modules:
                        imported_modules[bound_name] = possible_module
                        imported_id = possible_module
                    else:
                        imported_id = base_module
                        imported_symbols[bound_name] = possible_module
                    if imported_id:
                        _append_import(
                            parsed,
                            seen_imports,
                            module_id,
                            imported_id,
                            alias.name,
                            node.lineno,
                            imported_id not in known_modules,
                        )
        import_aliases[module_id] = imported_symbols
        module_aliases[module_id] = imported_modules

    symbol_ids = set(symbol_nodes)
    for caller_id, node in symbol_nodes.items():
        module_id = symbol_modules[caller_id]
        qualname = symbol_qualnames[caller_id]
        class_qualname = qualname.rsplit(".", 1)[0] if "." in qualname else None
        collector = _CallCollector()
        collector.visit_statements(getattr(node, "body", []))
        for call in collector.calls:
            callee_id = _resolve_call(
                call.func,
                module_id=module_id,
                class_qualname=class_qualname,
                symbol_ids=symbol_ids,
                imported_symbols=import_aliases[module_id],
                imported_modules=module_aliases[module_id],
            )
            if callee_id:
                parsed.calls.append(
                    CallRelationship(
                        caller_id=caller_id, callee_id=callee_id, line=call.lineno
                    )
                )

    parsed.calls.sort(key=lambda item: (item.caller_id, item.callee_id, item.line))
    parsed.imports.sort(
        key=lambda item: (
            item.importer_id,
            item.imported_id,
            item.imported_name,
            item.line,
        )
    )
    parsed.symbols.sort(key=lambda item: item.id)
    parsed.chunks.sort(key=lambda item: item.symbol_id)
    parsed.modules.sort(key=lambda item: item.id)
    parsed.warnings.sort(key=lambda item: item.file)
    return parsed


def _module_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__" and len(parts) > 1:
        parts.pop()
    return ".".join(parts)


def _collect_symbols(
    *,
    module_id: str,
    body: list[ast.stmt],
    source: str,
    relative_file: str,
    parsed: ParsedCodebase,
    symbol_nodes: dict[str, ast.AST],
    symbol_modules: dict[str, str],
    symbol_qualnames: dict[str, str],
    parent_qualname: str | None = None,
    parent_kind: str | None = None,
) -> None:
    lines = source.splitlines()
    for node in body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        qualname = f"{parent_qualname}.{node.name}" if parent_qualname else node.name
        symbol_id = f"{module_id}.{qualname}"
        parent_id = f"{module_id}.{parent_qualname}" if parent_qualname else None
        if isinstance(node, ast.ClassDef):
            kind = "class"
        elif parent_kind == "class":
            kind = "method"
        else:
            kind = "function"
        end_line = node.end_lineno or node.lineno
        parsed.symbols.append(
            CodeSymbol(
                id=symbol_id,
                name=node.name,
                qualified_name=symbol_id,
                kind=kind,
                file=relative_file,
                line=node.lineno,
                end_line=end_line,
                parent_id=parent_id,
            )
        )
        parsed.chunks.append(
            CodeChunk(
                symbol_id=symbol_id,
                name=node.name,
                kind=kind,
                file=relative_file,
                line=node.lineno,
                end_line=end_line,
                code="\n".join(lines[node.lineno - 1 : end_line]),
            )
        )
        symbol_nodes[symbol_id] = node
        symbol_modules[symbol_id] = module_id
        symbol_qualnames[symbol_id] = qualname
        _collect_symbols(
            module_id=module_id,
            body=node.body,
            source=source,
            relative_file=relative_file,
            parsed=parsed,
            symbol_nodes=symbol_nodes,
            symbol_modules=symbol_modules,
            symbol_qualnames=symbol_qualnames,
            parent_qualname=qualname,
            parent_kind=kind,
        )


def _resolve_import_from(
    current_module: str, imported_module: str | None, level: int
) -> str:
    if level == 0:
        return imported_module or ""
    current_package = current_module.split(".")[:-1]
    climb = max(level - 1, 0)
    if climb:
        current_package = (
            current_package[:-climb] if climb <= len(current_package) else []
        )
    if imported_module:
        current_package.extend(imported_module.split("."))
    return ".".join(current_package)


def _append_import(
    parsed: ParsedCodebase,
    seen: set[tuple[str, str, str, int]],
    importer_id: str,
    imported_id: str,
    imported_name: str,
    line: int,
    is_external: bool,
) -> None:
    key = (importer_id, imported_id, imported_name, line)
    if key in seen:
        return
    seen.add(key)
    parsed.imports.append(
        ImportRelationship(
            importer_id=importer_id,
            imported_id=imported_id,
            imported_name=imported_name,
            line=line,
            is_external=is_external,
        )
    )


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def visit_statements(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.visit(statement)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _resolve_call(
    function: ast.expr,
    *,
    module_id: str,
    class_qualname: str | None,
    symbol_ids: set[str],
    imported_symbols: dict[str, str],
    imported_modules: dict[str, str],
) -> str | None:
    if isinstance(function, ast.Name):
        name = function.id
        candidates = []
        if name in imported_symbols:
            candidates.append(imported_symbols[name])
        if class_qualname:
            candidates.append(f"{module_id}.{class_qualname}.{name}")
        candidates.append(f"{module_id}.{name}")
        return next(
            (candidate for candidate in candidates if candidate in symbol_ids), None
        )

    parts = _attribute_parts(function)
    if not parts:
        return None
    if parts[0] in {"self", "cls"} and class_qualname:
        candidate = f"{module_id}.{class_qualname}.{'.'.join(parts[1:])}"
        return candidate if candidate in symbol_ids else None
    if parts[0] in imported_modules:
        candidate = ".".join([imported_modules[parts[0]], *parts[1:]])
        return candidate if candidate in symbol_ids else None
    return None


def _attribute_parts(expression: ast.expr) -> list[str]:
    parts: list[str] = []
    while isinstance(expression, ast.Attribute):
        parts.append(expression.attr)
        expression = expression.value
    if isinstance(expression, ast.Name):
        parts.append(expression.id)
        return list(reversed(parts))
    return []
