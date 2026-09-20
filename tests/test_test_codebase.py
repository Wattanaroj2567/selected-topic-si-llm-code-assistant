from pathlib import Path

import pytest

from test_codebase.metrics import mean, standard_deviation
from test_codebase.pipeline import run_pipeline
from test_codebase.reporting import StatisticsReport


def test_run_pipeline_loads_numbers_and_builds_statistics_report(
    tmp_path: Path,
) -> None:
    numbers_file = tmp_path / "numbers.txt"
    numbers_file.write_text("2\n4\n6\n", encoding="utf-8")

    report = run_pipeline(numbers_file)

    assert isinstance(report, StatisticsReport)
    assert report.count == 3
    assert report.mean_value == 4.0
    assert report.standard_deviation == pytest.approx(1.6329931619)
    assert report.render() == "Count: 3 | Mean: 4.00 | Std dev: 1.63"


def test_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one number"):
        mean([])

    with pytest.raises(ValueError, match="at least one number"):
        standard_deviation([])
