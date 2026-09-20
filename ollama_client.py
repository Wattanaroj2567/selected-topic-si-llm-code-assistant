"""Small, explicit client for the local Ollama HTTP API."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT_SECONDS,
)


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str = OLLAMA_BASE_URL,
        embedding_model: str = EMBEDDING_MODEL,
        chat_model: str = CHAT_MODEL,
        timeout: int = OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        normalized_url = base_url if "://" in base_url else f"http://{base_url}"
        self.base_url = normalized_url.rstrip("/")
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.timeout = timeout

    @property
    def model(self) -> str:
        return self.embedding_model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("At least one text is required for embedding")
        response = self._post(
            "/api/embed",
            {"model": self.embedding_model, "input": texts},
        )
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError("Ollama returned an invalid embedding response")
        try:
            matrix = np.asarray(embeddings, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise OllamaError("Ollama returned non-numeric embeddings") from error
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise OllamaError("Ollama returned empty or inconsistent embeddings")
        return matrix

    def chat(self, *, system: str, user: str) -> str:
        response = self._post(
            "/api/chat",
            {
                "model": self.chat_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0},
            },
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama returned an empty chat response")
        return content.strip()

    def _post(self, endpoint: str, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise OllamaError(
                f"Ollama HTTP {error.code}: {detail or error.reason}"
            ) from error
        except (URLError, TimeoutError, ConnectionError) as error:
            raise OllamaError(
                "Cannot reach Ollama. Start it and confirm the required models are installed."
            ) from error
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as error:
            raise OllamaError("Ollama returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise OllamaError("Ollama returned an unexpected response")
        if decoded.get("error"):
            raise OllamaError(f"Ollama error: {decoded['error']}")
        return decoded
