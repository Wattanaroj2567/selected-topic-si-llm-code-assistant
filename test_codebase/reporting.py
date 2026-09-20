"""Create a display-friendly statistics report."""

from collections.abc import Sequence
from dataclasses import dataclass

from .metrics import mean, standard_deviation


@dataclass(frozen=True)
class StatisticsReport:
    """Calculated values returned by the pipeline."""

    count: int
    mean_value: float
    standard_deviation: float

    def render(self) -> str:
        """Format the report for a terminal."""
        return (
            f"Count: {self.count} | Mean: {self.mean_value:.2f} | "
            f"Std dev: {self.standard_deviation:.2f}"
        )


def build_report(numbers: Sequence[float]) -> StatisticsReport:
    """Calculate metrics and return a report object."""
    return StatisticsReport(
        count=len(numbers),
        mean_value=mean(numbers),
        standard_deviation=standard_deviation(numbers),
    )
