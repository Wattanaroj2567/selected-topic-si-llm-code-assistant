"""Real Ollama + FAISS + Kuzu smoke test used before demo/submission."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_rag import HybridRAG
from rag_pipeline import contains_forbidden_script

CASES = [
    ("What does the mean function do?", "vector", "metrics.mean"),
    ("Explain how run_pipeline works.", "vector", "pipeline.run_pipeline"),
    (
        "How is standard deviation calculated?",
        "vector",
        "metrics.standard_deviation",
    ),
    ("Who calls the mean function?", "graph", "metrics.variance"),
    (
        "What functions does run_pipeline call?",
        "graph",
        "data_loader.load_numbers",
    ),
    ("ฟังก์ชัน mean ทำงานอย่างไร", "vector", "metrics.mean"),
    ("อธิบายการทำงานของ run_pipeline", "vector", "pipeline.run_pipeline"),
    (
        "ฟังก์ชัน standard_deviation คำนวณอย่างไร",
        "vector",
        "metrics.standard_deviation",
    ),
    ("ใครเรียก mean", "graph", "metrics.variance"),
    (
        "run_pipeline เรียกฟังก์ชันอะไรบ้าง",
        "graph",
        "data_loader.load_numbers",
    ),
    ("pipeline นำเข้าอะไรบ้าง", "graph", "pipeline"),
]


def contains_thai(text: str) -> bool:
    return any("\u0e00" <= character <= "\u0e7f" for character in text)


def main() -> int:
    assistant = HybridRAG.from_data_dir()
    failures = 0
    try:
        for question, expected_mode, expected_symbol in CASES:
            result = assistant.query(question)
            symbols = {source["symbol"] for source in result["sources"]}
            passed = (
                result["error"] is None
                and result["mode"] == expected_mode
                and expected_symbol in symbols
                and bool(result["answer"].strip())
                and (
                    not contains_thai(question)
                    or (
                        contains_thai(result["answer"])
                        and not contains_forbidden_script(result["answer"])
                    )
                )
            )
            failures += not passed
            print(
                f"[{'PASS' if passed else 'FAIL'}] {expected_mode:6} | "
                f"{question} | sources={sorted(symbols)}"
            )

        help_result = assistant.query("สวัสดีครับ")
        help_passed = (
            help_result["mode"] == "help"
            and help_result["sources"] == []
            and contains_thai(help_result["answer"])
            and not contains_forbidden_script(help_result["answer"])
        )
        failures += not help_passed
        print(f"[{'PASS' if help_passed else 'FAIL'}] help   | สวัสดีครับ | sources=[]")
    finally:
        assistant.close()
    total = len(CASES) + 1
    print(f"\nResult: {total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
