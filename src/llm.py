"""Client for a local Ollama server."""

import logging

import requests

from config import LLM_MODEL, LLM_REQUEST_TIMEOUT_SECONDS, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised for any failure talking to the LLM, with a message safe to
    show a user (no stack traces / raw connection errors leaking through)."""


class OllamaClient:
    def __init__(self, model=LLM_MODEL, base_url=OLLAMA_BASE_URL, timeout=LLM_REQUEST_TIMEOUT_SECONDS):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def chat(self, messages):
        try:
            response = requests.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": messages, "stream": False},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(
                f"Could not reach Ollama at {self._base_url}. "
                "Is `ollama serve` running?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMError(
                f"Ollama did not respond within {self._timeout}s."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            if response.status_code == 404:
                raise LLMError(
                    f"Model '{self._model}' not found on the Ollama server. "
                    f"Run `ollama pull {self._model}`."
                ) from exc
            raise LLMError(f"Ollama returned an error: {exc}") from exc

        try:
            return response.json()["message"]["content"]
        except (KeyError, ValueError) as exc:
            raise LLMError(f"Unexpected response format from Ollama: {response.text[:200]}") from exc
