"""Anthropic Claude provider for chart analysis.

Supports Claude 3.5 Sonnet, Claude 3 Opus, and Claude 4 models.
Claude is particularly strong at chart visual analysis.
"""

from __future__ import annotations

import time
import os
from typing import Optional

from llm_regime.providers.base import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Claude via Anthropic API."""

    # Approximate cost per 1K tokens (input/output)
    COST_MAP = {
        "claude-sonnet-4-20250514": (0.003, 0.015),
        "claude-haiku-4-5-20251001": (0.001, 0.005),
        "claude-opus-4-6": (0.015, 0.075),
    }

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model or "claude-sonnet-4-20250514"
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        return self._client

    def _check_availability(self) -> bool:
        return bool(self._api_key)

    def analyze_image(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        client = self._get_client()
        model = model or self._model

        t0 = time.monotonic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }],
        )
        latency = int((time.monotonic() - t0) * 1000)

        text = response.content[0].text
        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens

        # Estimate cost
        rates = self.COST_MAP.get(model, (0.003, 0.015))
        cost = (in_tok / 1000 * rates[0]) + (out_tok / 1000 * rates[1])

        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
        )
