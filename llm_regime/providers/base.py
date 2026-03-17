"""Base provider interface for LLM regime analysis.

All providers implement this interface, making the analyzer
LLM-agnostic. Switch between Claude, GPT, Gemini, or local
models by changing the provider configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMResponse:
    """Raw response from an LLM provider."""
    text: str
    model: str
    provider: str
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    raw: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers.

    Subclasses implement `analyze_image()` which sends a chart image
    to the LLM and returns the raw text response.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'anthropic', 'openai', 'ollama')."""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model identifier."""
        ...

    @abstractmethod
    def analyze_image(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Send a chart image to the LLM for analysis.

        Parameters
        ----------
        image_base64 : base64-encoded PNG image
        system_prompt : system message defining the analyst role
        user_prompt : user message with analysis request
        model : model identifier (uses default if None)
        temperature : sampling temperature (low for consistency)
        max_tokens : maximum response tokens

        Returns
        -------
        LLMResponse with the raw text (expected to be JSON)
        """
        ...

    def is_available(self) -> bool:
        """Check if this provider is properly configured."""
        try:
            return self._check_availability()
        except Exception:
            return False

    def _check_availability(self) -> bool:
        """Override in subclass to check API keys, connections, etc."""
        return True
