"""OpenAI GPT provider for chart analysis.

Supports GPT-4o, GPT-4-turbo, and other vision-capable models.
"""

from __future__ import annotations

import time
import os
from typing import Optional

from llm_regime.providers.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """GPT via OpenAI API."""

    COST_MAP = {
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4-turbo": (0.01, 0.03),
    }

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model or "gpt-4o"
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
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
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        )
        latency = int((time.monotonic() - t0) * 1000)

        text = response.choices[0].message.content
        in_tok = response.usage.prompt_tokens if response.usage else 0
        out_tok = response.usage.completion_tokens if response.usage else 0

        rates = self.COST_MAP.get(model, (0.005, 0.015))
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
