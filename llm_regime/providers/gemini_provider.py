"""Google Gemini provider for chart analysis.

Uses the new google.genai library (replaces deprecated google.generativeai).
Supports Gemini 2.5 Flash (free), 2.5 Pro, 3 Flash Preview.
"""

from __future__ import annotations

import time
import os
import base64
from typing import Optional

from llm_regime.providers.base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Gemini via Google GenAI API."""

    COST_MAP = {
        "gemini-2.5-flash": (0.0003, 0.0025),
        "gemini-2.5-pro": (0.00125, 0.010),
        "gemini-2.5-flash-lite": (0.0001, 0.0004),
    }

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._model = model or "gemini-2.5-flash"
        self._client = None

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "Install google-genai: pip3 install google-genai"
                )
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
        max_tokens: int = 2048,
    ) -> LLMResponse:
        client = self._get_client()
        model_name = model or self._model

        from google.genai import types

        image_bytes = base64.b64decode(image_base64)

        t0 = time.monotonic()
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        types.Part.from_text(text=user_prompt),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        latency = int((time.monotonic() - t0) * 1000)

        text = response.text or ""

        # Token usage
        in_tok = 0
        out_tok = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            in_tok = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            out_tok = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        rates = self.COST_MAP.get(model_name, (0.0003, 0.0025))
        cost = (in_tok / 1000 * rates[0]) + (out_tok / 1000 * rates[1])

        return LLMResponse(
            text=text,
            model=model_name,
            provider=self.name,
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
        )
