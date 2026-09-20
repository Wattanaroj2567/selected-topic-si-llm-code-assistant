import pytest

from ollama_client import OllamaClient, OllamaError


def test_ollama_client_normalizes_host_without_url_scheme() -> None:
    client = OllamaClient(base_url="127.0.0.1:11434")

    assert client.base_url == "http://127.0.0.1:11434"


def test_ollama_client_reports_unreachable_service() -> None:
    client = OllamaClient(base_url="http://127.0.0.1:1", timeout=1)

    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        client.embed(["hello"])
