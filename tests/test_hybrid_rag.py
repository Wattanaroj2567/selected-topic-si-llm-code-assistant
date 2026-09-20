from hybrid_rag import (
    HybridRAG,
    QuestionRouter,
    classify_graph_relationship,
    extract_graph_entity,
)

ROUTING_CASES = [
    ("What does the mean function do?", "vector"),
    ("Explain how run_pipeline works.", "vector"),
    ("How is standard deviation calculated?", "vector"),
    ("Where is input validation handled?", "vector"),
    ("Who calls mean()?", "graph"),
    ("What functions does run_pipeline call?", "graph"),
    ("What does pipeline import?", "graph"),
    ("Which functions call variance?", "graph"),
    ("Show the relationships around build_report.", "graph"),
    ("Describe the pipeline architecture.", "vector"),
]


class StubRAG:
    def __init__(self, label: str) -> None:
        self.label = label

    def query(self, question: str) -> dict:
        return {
            "answer": f"{self.label} answered: {question}",
            "sources": [{"symbol": f"{self.label}.source"}],
        }

    def close(self) -> None:
        pass


class UnexpectedRAG:
    def query(self, question: str) -> dict:
        raise AssertionError("help requests must not call a retrieval backend")

    def close(self) -> None:
        pass


def test_question_router_reaches_target_accuracy() -> None:
    router = QuestionRouter()
    correct = sum(
        router.classify(question) == expected for question, expected in ROUTING_CASES
    )

    assert correct >= 8


def test_question_router_and_graph_parser_understand_thai_questions() -> None:
    router = QuestionRouter()
    cases = [
        ("ฟังก์ชัน mean ทำงานอย่างไร", "vector", None, None),
        ("ใครเรียก mean", "graph", "mean", "callers"),
        (
            "run_pipeline เรียกฟังก์ชันอะไรบ้าง",
            "graph",
            "run_pipeline",
            "callees",
        ),
        ("pipeline นำเข้าอะไรบ้าง", "graph", "pipeline", "imports"),
        ("pipeline ขึ้นอยู่กับอะไร", "graph", "pipeline", "imports"),
    ]

    for question, expected_mode, expected_entity, expected_relationship in cases:
        assert router.classify(question) == expected_mode
        if expected_mode == "graph":
            assert extract_graph_entity(question) == expected_entity
            assert classify_graph_relationship(question) == expected_relationship


def test_hybrid_rag_routes_to_real_backend_and_returns_common_schema() -> None:
    hybrid = HybridRAG(StubRAG("vector"), StubRAG("graph"))

    vector_result = hybrid.query("Explain how mean works")
    graph_result = hybrid.query("Who calls mean?", force_mode="graph")

    assert vector_result["mode"] == "vector"
    assert vector_result["sources"] == [{"symbol": "vector.source"}]
    assert vector_result["answer"].startswith("vector answered")
    assert vector_result["elapsed"] >= 0
    assert vector_result["error"] is None

    assert graph_result["mode"] == "graph"
    assert graph_result["sources"] == [{"symbol": "graph.source"}]
    assert graph_result["answer"].startswith("graph answered")
    assert graph_result["error"] is None


def test_thai_greeting_uses_help_without_retrieval() -> None:
    hybrid = HybridRAG(UnexpectedRAG(), UnexpectedRAG())

    result = hybrid.query("สวัสดีครับ")

    assert result["mode"] == "help"
    assert result["sources"] == []
    assert "ฟังก์ชัน mean ทำงานอย่างไร" in result["answer"]
    assert "ใครเรียก mean" in result["answer"]
    assert result["error"] is None
