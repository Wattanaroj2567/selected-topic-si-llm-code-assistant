"""Orchestrate loading and reporting."""

from pathlib import Path

from .data_loader import load_numbers
from .reporting import StatisticsReport, build_report


def run_pipeline(source: str | Path) -> StatisticsReport:
    """Load a number file and build its statistics report."""
    numbers = load_numbers(source)
    return build_report(numbers)
