"""LLM provider implementations.

Supported providers:
  - anthropic: Claude (3.5 Sonnet, Opus, etc.)
  - openai: GPT-4o, GPT-4-turbo
  - gemini: Gemini 2.0 Flash/Pro
  - ollama: Local models (llama3.2-vision, llava, gemma3)
"""

from llm_regime.providers.base import BaseLLMProvider, LLMResponse
from typing import Optional, Dict, Any


def create_provider(
    provider: str = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> BaseLLMProvider:
    """Factory to create an LLM provider.

    Parameters
    ----------
    provider : "anthropic", "openai", "gemini", or "ollama"
    api_key : API key (not needed for ollama)
    model : model identifier (uses provider default if None)
    **kwargs : extra args passed to provider constructor

    Returns
    -------
    BaseLLMProvider instance
    """
    provider = provider.lower()

    if provider in ("anthropic", "claude"):
        from llm_regime.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key, model=model, **kwargs)

    elif provider in ("openai", "gpt", "chatgpt"):
        from llm_regime.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key, model=model, **kwargs)

    elif provider in ("gemini", "google"):
        from llm_regime.providers.gemini_provider import GeminiProvider
        return GeminiProvider(api_key=api_key, model=model, **kwargs)

    elif provider in ("ollama", "local"):
        from llm_regime.providers.ollama_provider import OllamaProvider
        return OllamaProvider(model=model, **kwargs)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: anthropic, openai, gemini, ollama"
        )


__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "create_provider",
]
