"""Ollama provider for local LLM chart analysis.

Runs vision models locally — no API keys, no cloud, no cost.
Supports llama3.2-vision, llava, gemma3, and other vision models.

Requires Ollama installed: https://ollama.com
Pull a vision model: ollama pull llama3.2-vision
"""

from __future__ import annotations

import time
import os
import base64
from typing import Optional

from llm_regime.providers.base import BaseLLMProvider, LLMResponse


class OllamaProvider(BaseLLMProvider):
    """Local LLM via Ollama."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._model = model or "llama3.2-vision"
        self._base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._client = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self._base_url)
            except ImportError:
                raise ImportError("Install ollama: pip install ollama")
        return self._client

    def _check_availability(self) -> bool:
        try:
            client = self._get_client()
            client.list()
            return True
        except Exception:
            return False

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
        model_name = model or self._model

        # Ollama accepts base64 images directly
        t0 = time.monotonic()
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_base64],
                },
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        latency = int((time.monotonic() - t0) * 1000)

        text = response["message"]["content"]

        # Ollama provides token counts in some responses
        in_tok = response.get("prompt_eval_count", 0)
        out_tok = response.get("eval_count", 0)

        return LLMResponse(
            text=text,
            model=model_name,
            provider=self.name,
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=0.0,  # local = free
        )
