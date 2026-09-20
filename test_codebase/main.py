"""Command-line entry point for the sample statistics project."""

import argparse

from .pipeline import run_pipeline


def main() -> None:
    """Run the pipeline for a number file and print the report."""
    parser = argparse.ArgumentParser(description="Summarize a number file")
    parser.add_argument("source", help="UTF-8 text file containing one number per line")
    args = parser.parse_args()

    report = run_pipeline(args.source)
    print(report.render())


if __name__ == "__main__":
    main()
