from pathlib import Path

from ast_parser import parse_codebase

PROJECT_ROOT = Path(__file__).parents[1]
DATASET_ROOT = PROJECT_ROOT / "test_codebase"


def test_parse_codebase_resolves_expected_local_imports_and_calls() -> None:
    parsed = parse_codebase(DATASET_ROOT)

    imports = {(item.importer_id, item.imported_id) for item in parsed.imports}
    calls = {(item.caller_id, item.callee_id) for item in parsed.calls}

    assert ("reporting", "metrics") in imports
    assert ("pipeline", "data_loader") in imports
    assert ("pipeline", "reporting") in imports
    assert ("main", "pipeline") in imports

    assert ("metrics.variance", "metrics.mean") in calls
    assert ("metrics.standard_deviation", "metrics.variance") in calls
    assert ("reporting.build_report", "metrics.mean") in calls
    assert (
        "reporting.build_report",
        "metrics.standard_deviation",
    ) in calls
    assert ("pipeline.run_pipeline", "data_loader.load_numbers") in calls
    assert ("pipeline.run_pipeline", "reporting.build_report") in calls
    assert ("main.main", "pipeline.run_pipeline") in calls

    render = next(
        symbol
        for symbol in parsed.symbols
        if symbol.id == "reporting.StatisticsReport.render"
    )
    assert render.kind == "method"
    assert render.parent_id == "reporting.StatisticsReport"
    assert render.file == "reporting.py"
    assert render.line > 0
    assert any(chunk.symbol_id == render.id for chunk in parsed.chunks)


def test_parse_codebase_skips_syntax_error_and_reports_warning(tmp_path: Path) -> None:
    (tmp_path / "valid.py").write_text(
        "def working():\n    return 42\n",
        encoding="utf-8",
    )
    (tmp_path / "broken.py").write_text(
        "def broken(:\n    pass\n",
        encoding="utf-8",
    )

    parsed = parse_codebase(tmp_path)

    assert parsed.files_scanned == 2
    assert parsed.files_parsed == 1
    assert [symbol.id for symbol in parsed.symbols] == ["valid.working"]
    assert len(parsed.warnings) == 1
    assert parsed.warnings[0].file == "broken.py"
    assert "SyntaxError" in parsed.warnings[0].message
