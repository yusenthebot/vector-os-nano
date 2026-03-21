"""LocalProvider — Ollama wrapper using the OpenAI-compatible API.

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1,
so this is a thin subclass of OpenAIProvider with Ollama-specific defaults.

Supports any Ollama-hosted model: llama3, mistral, gemma2, phi3, etc.
Also works with LM Studio (same API, different port).

No ROS2 imports. No API key required for local inference.
"""
from __future__ import annotations

from vector_os_nano.llm.openai_compat import OpenAIProvider

_DEFAULT_MODEL = "llama3"
_DEFAULT_HOST = "http://localhost:11434"


class LocalProvider(OpenAIProvider):
    """LLM provider wrapping a locally-running Ollama instance.

    Uses Ollama's OpenAI-compatible /v1 endpoint. No API key needed
    for local inference — an empty string is used as a placeholder.

    Args:
        model: Ollama model name (default "llama3").
        host: base URL of the Ollama server (default "http://localhost:11434").
        **kwargs: forwarded to OpenAIProvider (max_history, temperature, etc.).

    Example::

        provider = LocalProvider(model="mistral")
        plan = provider.plan(goal="pick the cup", ...)
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        **kwargs,
    ) -> None:
        api_base = f"{host.rstrip('/')}/v1"
        super().__init__(
            api_key=kwargs.pop("api_key", "ollama"),
            model=model,
            api_base=api_base,
            **kwargs,
        )
