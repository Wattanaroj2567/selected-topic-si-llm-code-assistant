"""Load numeric input for the statistics pipeline."""

from pathlib import Path


def load_numbers(source: str | Path) -> list[float]:
    """Read one number per line, ignoring blank lines."""
    lines = Path(source).read_text(encoding="utf-8").splitlines()
    return [float(line.strip()) for line in lines if line.strip()]
