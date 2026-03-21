"""LLM provider implementations for Vector OS.

All providers implement the LLMProvider protocol defined in base.py.

Exports:
- LLMProvider: structural protocol (use for isinstance checks)
- ClaudeProvider: Anthropic Claude via OpenRouter or direct API
- OpenAIProvider: any OpenAI-compatible API (OpenAI, vLLM, LM Studio, etc.)
- LocalProvider: Ollama local inference (subclass of OpenAIProvider)

No ROS2 imports anywhere in this subpackage.
"""
from vector_os_nano.llm.base import LLMProvider
from vector_os_nano.llm.claude import ClaudeProvider
from vector_os_nano.llm.openai_compat import OpenAIProvider
from vector_os_nano.llm.local import LocalProvider

__all__ = [
    "LLMProvider",
    "ClaudeProvider",
    "OpenAIProvider",
    "LocalProvider",
]
