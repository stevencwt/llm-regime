"""RegimeAnalyzer — the primary API for LLM-based regime detection.

Usage:
    from llm_regime import RegimeAnalyzer

    # Create with any provider
    analyzer = RegimeAnalyzer(provider="anthropic", api_key="sk-...")
    # or: RegimeAnalyzer(provider="ollama", model="llama3.2-vision")

    # Analyze from OHLCV data
    result = analyzer.analyze(
        asset="BTC",
        timeframe="5m",
        opens=opens, highs=highs, lows=lows, closes=closes,
        volumes=volumes,
    )

    print(result.regime)          # "UPTREND"
    print(result.scalp_direction) # "LONG_ONLY"
    print(result.confidence)      # 4

    # Results are cached — subsequent calls return cached result
    # until TTL expires (default 30 min)
"""

from __future__ import annotations

import json
import re
import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np

from llm_regime.schema import RegimeResult, Regime
from llm_regime.providers import create_provider, BaseLLMProvider
from llm_regime.prompts import SYSTEM_PROMPT, build_regime_prompt
from llm_regime.charts import generate_chart_image, chart_to_base64
from llm_regime.cache import RegimeCache

logger = logging.getLogger(__name__)


class RegimeAnalyzer:
    """LLM-powered regime analyzer.

    Generates a chart image from OHLCV data, sends it to an LLM for
    visual analysis, and returns a structured RegimeResult.

    Parameters
    ----------
    provider : LLM provider name ("anthropic", "openai", "gemini", "ollama")
    api_key : API key (not needed for ollama)
    model : model identifier (uses provider default if None)
    cache_ttl : seconds to cache results (default 1800 = 30 min)
    cache_path : optional file path for persistent cache
    temperature : LLM sampling temperature (default 0.1 for consistency)
    sma_periods : which SMAs to draw on charts
    chart_dpi : chart image resolution
    provider_kwargs : extra kwargs for provider constructor
    """

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_ttl: int = 1800,
        cache_path: Optional[str] = None,
        temperature: float = 0.1,
        sma_periods: Tuple[int, ...] = (20, 50),
        chart_dpi: int = 100,
        **provider_kwargs,
    ):
        self._provider = create_provider(
            provider=provider, api_key=api_key, model=model, **provider_kwargs
        )
        self._cache = RegimeCache(default_ttl=cache_ttl, persist_path=cache_path)
        self._temperature = temperature
        self._sma_periods = sma_periods
        self._chart_dpi = chart_dpi

    @property
    def provider(self) -> BaseLLMProvider:
        return self._provider

    @property
    def cache(self) -> RegimeCache:
        return self._cache

    def analyze(
        self,
        asset: str,
        timeframe: str,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        force_refresh: bool = False,
        extra_context: str = "",
        cache_ttl: Optional[int] = None,
        analysis_window: Optional[int] = None,
    ) -> RegimeResult:
        """Analyze market regime from OHLCV data.

        Parameters
        ----------
        asset : asset symbol (e.g. "BTC", "SOL")
        timeframe : candle timeframe (e.g. "5m", "1h", "1d")
        opens, highs, lows, closes : price arrays
        volumes : optional volume array
        timestamps : optional timestamp array
        force_refresh : bypass cache
        extra_context : additional context for the LLM
        cache_ttl : override cache TTL for this call
        analysis_window : if set, the LLM focuses on the last N bars.
                          Earlier bars are shown grayed out as context.
                          Example: 500 total bars + analysis_window=100 means
                          400 gray context bars + 100 colored analysis bars.

        Returns
        -------
        RegimeResult with regime classification
        """
        # Build cache key that includes window
        cache_key_tf = f"{timeframe}:w{analysis_window}" if analysis_window else timeframe

        # Check cache first
        if not force_refresh:
            cached = self._cache.get(asset, cache_key_tf)
            if cached is not None:
                logger.debug("Cache hit for %s:%s", asset, cache_key_tf)
                return cached

        # Generate chart image
        closes = np.asarray(closes, dtype=float)
        opens = np.asarray(opens, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)

        if timestamps is None:
            timestamps = np.arange(len(closes))

        chart_png = generate_chart_image(
            timestamps=timestamps,
            opens=opens, highs=highs, lows=lows, closes=closes,
            volumes=volumes,
            asset=asset,
            timeframe=timeframe,
            sma_periods=self._sma_periods,
            dpi=self._chart_dpi,
            analysis_window=analysis_window,
        )
        chart_b64 = chart_to_base64(chart_png)

        # Compute quantitative context (from analysis window if set)
        if analysis_window and analysis_window < len(closes):
            price_data = self._compute_price_data(closes[-analysis_window:], self._sma_periods)
        else:
            price_data = self._compute_price_data(closes, self._sma_periods)

        # Build prompt
        user_prompt = build_regime_prompt(
            asset=asset,
            timeframe=timeframe,
            bars=len(closes),
            price_data=price_data,
            extra_context=extra_context,
            analysis_window=analysis_window,
        )

        # Call LLM with retry for truncated/failed responses
        logger.info("Calling %s/%s for %s:%s (%d bars)",
                     self._provider.name, self._provider.default_model,
                     asset, timeframe, len(closes))

        max_retries = 3
        total_latency = 0
        total_cost = 0.0
        result = None

        for attempt in range(max_retries):
            try:
                response = self._provider.analyze_image(
                    image_base64=chart_b64,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=self._temperature,
                )
                total_latency += response.latency_ms
                total_cost += response.cost_usd

                result = self._parse_response(response.text, asset, timeframe, len(closes))
                result.provider = response.provider
                result.model = response.model
                result.latency_ms = total_latency
                result.cost_usd = round(total_cost, 6)

                # If we got a valid regime (not UNKNOWN from parse failure), accept it
                if result.regime != Regime.UNKNOWN.value:
                    break

                # UNKNOWN from parse failure — retry
                if attempt < max_retries - 1:
                    import time
                    wait = 5 * (attempt + 1)
                    logger.info("Retry %d/%d in %ds (parse failed: %s)",
                                attempt + 1, max_retries, wait, result.reasoning[:80])
                    time.sleep(wait)

            except Exception as e:
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    import time
                    wait = 10 * (attempt + 1)
                    logger.info("Retry in %ds...", wait)
                    time.sleep(wait)
                else:
                    result = RegimeResult(
                        regime=Regime.UNKNOWN.value,
                        asset=asset, timeframe=timeframe, bars_analyzed=len(closes),
                        reasoning=f"LLM call failed after {max_retries} attempts: {e}",
                    )

        if result is None:
            result = RegimeResult(
                regime=Regime.UNKNOWN.value,
                asset=asset, timeframe=timeframe, bars_analyzed=len(closes),
                reasoning="No result after retries",
            )

        # Cache result (with window-aware key)
        result.timeframe = cache_key_tf
        self._cache.put(result, ttl=cache_ttl)
        result.timeframe = timeframe  # restore clean timeframe for consumer

        logger.info(
            "Regime: %s (conf=%d) bias=%s dir=%s | %dms $%.4f",
            result.regime, result.confidence, result.bias,
            result.scalp_direction, result.latency_ms, result.cost_usd,
        )

        return result

    def analyze_from_chart(
        self,
        chart_png: bytes,
        asset: str = "",
        timeframe: str = "",
        bars: int = 0,
        extra_context: str = "",
    ) -> RegimeResult:
        """Analyze from a pre-generated chart image (PNG bytes).

        Useful when the chart is generated externally (e.g., TradingView
        screenshot, custom charting library).
        """
        chart_b64 = chart_to_base64(chart_png)
        user_prompt = build_regime_prompt(
            asset=asset, timeframe=timeframe, bars=bars,
            extra_context=extra_context,
        )

        response = self._provider.analyze_image(
            image_base64=chart_b64,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self._temperature,
        )

        result = self._parse_response(response.text, asset, timeframe, bars)
        result.provider = response.provider
        result.model = response.model
        result.latency_ms = response.latency_ms
        result.cost_usd = response.cost_usd
        return result

    def analyze_multi_scale(
        self,
        asset: str,
        timeframe: str,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        macro_bars: int = 500,
        micro_bars: int = 100,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Analyze regime at two scales and combine.

        Returns both macro (trend direction) and micro (current structure)
        classifications, plus a combined recommendation.

        Parameters
        ----------
        macro_bars : total bars for context (default 500 = ~42h at 5m)
        micro_bars : analysis window for current state (default 100 = ~8h at 5m)

        Returns
        -------
        dict with keys:
            macro : RegimeResult for the broad trend
            micro : RegimeResult for the current window
            combined_direction : "LONG_ONLY" | "SHORT_ONLY" | "BOTH" | "NO_TRADE"
            combined_regime : synthesized regime string
            confidence : min of macro and micro confidence
        """
        n = len(closes)
        actual_macro = min(macro_bars, n)
        actual_micro = min(micro_bars, n)

        # Macro: analyze all bars (broad trend)
        macro_result = self.analyze(
            asset=asset, timeframe=timeframe,
            opens=opens[-actual_macro:], highs=highs[-actual_macro:],
            lows=lows[-actual_macro:], closes=closes[-actual_macro:],
            volumes=volumes[-actual_macro:] if volumes is not None else None,
            timestamps=timestamps[-actual_macro:] if timestamps is not None else None,
            force_refresh=force_refresh,
            extra_context=f"This is the MACRO view ({actual_macro} bars). Classify the broad trend direction.",
        )

        # Micro: analyze with context but focus on last N bars
        micro_result = self.analyze(
            asset=asset, timeframe=timeframe,
            opens=opens[-actual_macro:], highs=highs[-actual_macro:],
            lows=lows[-actual_macro:], closes=closes[-actual_macro:],
            volumes=volumes[-actual_macro:] if volumes is not None else None,
            timestamps=timestamps[-actual_macro:] if timestamps is not None else None,
            force_refresh=force_refresh,
            analysis_window=actual_micro,
            extra_context=f"This is the MICRO view (focus on last {actual_micro} bars). Classify the current short-term state within the broader context.",
        )

        # Combine: macro sets direction, micro refines it
        direction = self._combine_directions(macro_result, micro_result)

        # Combined regime: use micro regime but validate against macro
        if macro_result.is_bullish and micro_result.is_bearish:
            combined = "TRANSITION"  # conflicting signals
        elif macro_result.is_bearish and micro_result.is_bullish:
            combined = "TRANSITION"  # conflicting signals
        else:
            combined = micro_result.regime  # micro prevails when aligned

        return {
            "macro": macro_result,
            "micro": micro_result,
            "combined_direction": direction,
            "combined_regime": combined,
            "confidence": min(macro_result.confidence, micro_result.confidence),
        }

    def _combine_directions(self, macro: RegimeResult, micro: RegimeResult) -> str:
        """Combine macro and micro into a single direction.

        Rules:
          - Macro bullish → only allow LONG (regardless of micro)
          - Macro bearish → only allow SHORT
          - Macro ranging + micro bullish → LONG_ONLY
          - Macro ranging + micro bearish → SHORT_ONLY
          - Macro ranging + micro ranging → BOTH
          - Any TRANSITION → NO_TRADE
        """
        if macro.regime == "TRANSITION" or micro.regime == "TRANSITION":
            return "NO_TRADE"

        if macro.is_bullish:
            return "LONG_ONLY"
        elif macro.is_bearish:
            return "SHORT_ONLY"
        elif macro.is_ranging:
            if micro.is_bullish:
                return "LONG_ONLY"
            elif micro.is_bearish:
                return "SHORT_ONLY"
            else:
                return "BOTH"

        return micro.scalp_direction

    def _compute_price_data(self, closes: np.ndarray, sma_periods: Tuple[int, ...]) -> Dict[str, Any]:
        """Compute quantitative metrics to supplement visual analysis."""
        data: Dict[str, Any] = {}
        n = len(closes)

        data["current_price"] = float(closes[-1])
        data["price_change_pct"] = float((closes[-1] / closes[0] - 1) * 100)

        for period in sma_periods:
            if n >= period:
                sma = float(np.mean(closes[-period:]))
                data[f"sma_{period}"] = sma

        # ATR approximation (using close-to-close)
        if n >= 20:
            returns = np.abs(np.diff(closes[-21:])) / closes[-21:-1]
            data["atr_pct"] = float(np.mean(returns) * 100)

        return data

    def _parse_response(self, text: str, asset: str, timeframe: str, bars: int) -> RegimeResult:
        """Parse LLM JSON response into RegimeResult.

        Handles various LLM quirks:
          - Markdown code fences around JSON
          - Extra text before/after JSON
          - Trailing commas in JSON
          - TRUNCATED responses (Gemini cuts off mid-JSON)
          - Missing or extra fields
        """
        logger.debug("Raw LLM response (%d chars): %s", len(text), text[:500])

        # Strip markdown fences
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # Try to find complete JSON object
        json_match = re.search(r"\{.*\}", text, re.DOTALL)

        if json_match:
            json_str = json_match.group()
        elif "{" in text:
            # Truncated JSON — has opening { but no closing }
            json_str = text[text.index("{"):]
            json_str = self._repair_truncated_json(json_str)
            logger.debug("Repaired truncated JSON: %s", json_str[:200])
        else:
            logger.warning("No JSON found in LLM response: %s", text[:200])
            return RegimeResult(
                regime=Regime.UNKNOWN.value,
                asset=asset, timeframe=timeframe, bars_analyzed=bars,
                reasoning=f"Failed to parse LLM response: {text[:200]}",
            )

        # Fix trailing commas
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Last resort: extract fields with regex from the raw text
            data = self._extract_fields_fallback(text)
            if not data:
                logger.warning("Cannot parse LLM response: %s", text[:300])
                return RegimeResult(
                    regime=Regime.UNKNOWN.value,
                    asset=asset, timeframe=timeframe, bars_analyzed=bars,
                    reasoning=f"JSON parse failed, raw: {text[:200]}",
                )

        # Validate regime value
        regime = data.get("regime", "UNKNOWN")
        valid_regimes = {r.value for r in Regime}
        if regime not in valid_regimes:
            logger.warning("Invalid regime '%s', defaulting to UNKNOWN", regime)
            regime = "UNKNOWN"

        # Build result
        result = RegimeResult(
            regime=regime,
            confidence=min(5, max(1, int(data.get("confidence", 1)))),
            volatility=data.get("volatility", "MODERATE"),
            trend_strength=data.get("trend_strength", "NONE"),
            bias=data.get("bias", "NEUTRAL"),
            scalp_direction=data.get("scalp_direction", "BOTH"),
            reasoning=data.get("reasoning", ""),
            pattern=data.get("pattern", ""),
            asset=asset,
            timeframe=timeframe,
            bars_analyzed=bars,
        )

        # Parse key levels
        key_levels = data.get("key_levels", [])
        if isinstance(key_levels, list):
            result.key_levels = [
                {"price": kl.get("price", 0), "type": kl.get("type", ""), "strength": kl.get("strength", "moderate")}
                for kl in key_levels
                if isinstance(kl, dict) and "price" in kl
            ]

        return result

    def _repair_truncated_json(self, json_str: str) -> str:
        """Attempt to repair truncated JSON by closing open structures.

        Handles cases like:
          {"regime": "DOWNTREND", "confidence": 4, "volatility": "MODERATE", "trend_
        """
        # Remove any trailing incomplete key-value pair
        # Find last complete value (ends with , or quoted string)
        last_complete = max(
            json_str.rfind('",'),
            json_str.rfind('],'),
            json_str.rfind('},'),
            json_str.rfind(': "'),
        )

        if last_complete > 0:
            # Find the end of the last complete value
            after = json_str[last_complete:]
            if after.startswith('",'):
                json_str = json_str[:last_complete + 1]  # keep the closing quote
            elif after.startswith('],') or after.startswith('},'):
                json_str = json_str[:last_complete + 1]
            elif after.startswith(': "'):
                # Find if there's a closing quote
                close_q = after.find('"', 3)
                if close_q > 0:
                    json_str = json_str[:last_complete + close_q + 1]
                else:
                    json_str = json_str[:last_complete]

        # Remove trailing comma
        json_str = json_str.rstrip().rstrip(",")

        # Close any open brackets/braces
        open_braces = json_str.count("{") - json_str.count("}")
        open_brackets = json_str.count("[") - json_str.count("]")
        json_str += "]" * max(0, open_brackets)
        json_str += "}" * max(0, open_braces)

        return json_str

    def _extract_fields_fallback(self, text: str) -> dict:
        """Last resort: extract regime fields from text using regex.

        Even if JSON is broken, we can often extract the key fields.
        """
        data = {}

        # Extract regime
        regime_match = re.search(r'"regime"\s*:\s*"([A-Z_]+)"', text)
        if regime_match:
            data["regime"] = regime_match.group(1)

        # Extract confidence
        conf_match = re.search(r'"confidence"\s*:\s*(\d+)', text)
        if conf_match:
            data["confidence"] = int(conf_match.group(1))

        # Extract other fields
        for field in ["volatility", "trend_strength", "bias", "scalp_direction", "reasoning", "pattern"]:
            match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text)
            if match:
                data[field] = match.group(1)

        return data if "regime" in data else {}
