"""Prompt engineering for LLM regime classification.

The prompts are designed to:
  1. Be model-agnostic (work with Claude, GPT, Gemini, Llama)
  2. Return structured JSON (parseable, not conversational)
  3. Focus on regime classification (not price prediction)
  4. Include chain-of-thought reasoning

The system prompt establishes the role; the user prompt includes the chart
image plus any quantitative context. The LLM responds with JSON only.
"""

from __future__ import annotations

from typing import Optional, Dict, Any


SYSTEM_PROMPT = """You are an expert technical analyst specializing in market regime classification. Your role is to analyze price charts and classify the current market regime.

IMPORTANT RULES:
1. Respond ONLY with valid JSON. No prose, no markdown, no explanation outside JSON.
2. Base your analysis on visual chart patterns — price structure, trend, support/resistance, volatility.
3. Be honest about confidence — if the chart is ambiguous, say so with low confidence.
4. Focus on the CURRENT regime (right edge of chart), not historical patterns.

Your JSON response must follow this exact schema:
{
  "regime": "STRONG_UPTREND|UPTREND|WEAK_UPTREND|RANGING|WEAK_DOWNTREND|DOWNTREND|STRONG_DOWNTREND|TRANSITION",
  "confidence": 1-5,
  "volatility": "LOW|MODERATE|HIGH|EXTREME",
  "trend_strength": "STRONG|MODERATE|WEAK|NONE",
  "bias": "BULLISH|BEARISH|NEUTRAL",
  "scalp_direction": "LONG_ONLY|SHORT_ONLY|BOTH|NO_TRADE",
  "key_levels": [{"price": 73500, "type": "resistance", "strength": "strong"}],
  "reasoning": "brief 1-2 sentence explanation",
  "pattern": "pattern name if any, empty string if none"
}

REGIME DEFINITIONS:
- STRONG_UPTREND: Clear higher highs + higher lows, price well above rising SMA, strong momentum
- UPTREND: Generally rising with pullbacks, price above SMA most of the time
- WEAK_UPTREND: Slight upward bias but choppy, price near SMA
- RANGING: Price oscillating between clear support/resistance, SMA flat
- WEAK_DOWNTREND: Slight downward bias but choppy
- DOWNTREND: Generally falling with bounces, price below SMA
- STRONG_DOWNTREND: Clear lower highs + lower lows, price well below falling SMA
- TRANSITION: Market structure is changing (breakout/breakdown in progress)

SCALP_DIRECTION RULES:
- STRONG_UPTREND/UPTREND → LONG_ONLY (don't short an uptrend)
- STRONG_DOWNTREND/DOWNTREND → SHORT_ONLY (don't buy a downtrend)
- WEAK_UPTREND → LONG_ONLY (cautious)
- WEAK_DOWNTREND → SHORT_ONLY (cautious)
- RANGING → BOTH (mean reversion both ways)
- TRANSITION → NO_TRADE (wait for clarity)

CONFIDENCE SCALE:
1 = Very uncertain, chart is ambiguous
2 = Somewhat uncertain, multiple interpretations possible
3 = Moderate confidence, regime is visible but not strong
4 = High confidence, clear regime with supporting evidence
5 = Very high confidence, textbook regime with strong signals"""


def build_regime_prompt(
    asset: str = "",
    timeframe: str = "",
    bars: int = 0,
    price_data: Optional[Dict[str, Any]] = None,
    extra_context: str = "",
    analysis_window: Optional[int] = None,
) -> str:
    """Build the user prompt that accompanies the chart image.

    Parameters
    ----------
    asset : asset symbol (e.g. "BTC", "SOL")
    timeframe : candle timeframe (e.g. "5m", "1h")
    bars : number of bars in the chart
    price_data : optional dict with computed metrics
    extra_context : any additional context
    analysis_window : if set, tells LLM to focus on the last N bars
                      (which are highlighted in full color on the chart)

    Returns
    -------
    str : the user prompt text
    """
    parts = []

    parts.append("Analyze this price chart and classify the current market regime.")
    parts.append("Respond ONLY with valid JSON matching the schema in your instructions.")

    if asset:
        parts.append(f"Asset: {asset}")
    if timeframe:
        parts.append(f"Timeframe: {timeframe}")
    if bars:
        parts.append(f"Total bars on chart: {bars}")

    # Analysis window instruction
    if analysis_window and analysis_window < (bars or 9999):
        parts.append(
            f"IMPORTANT: The chart shows {bars} bars for context, but "
            f"the last {analysis_window} bars are highlighted in full color. "
            f"The grayed-out bars on the left are context only. "
            f"Classify the regime based on the HIGHLIGHTED (colored) bars on the right side. "
            f"Use the grayed context to understand the broader trend, but your regime "
            f"classification should describe the CURRENT state in the highlighted window."
        )

    # Add quantitative context if available
    if price_data:
        ctx_parts = []
        if "current_price" in price_data:
            ctx_parts.append(f"Current price: {price_data['current_price']:.2f}")
        if "price_change_pct" in price_data:
            ctx_parts.append(f"Period change: {price_data['price_change_pct']:+.1f}%")
        if "sma_20" in price_data and "sma_50" in price_data:
            sma20 = price_data["sma_20"]
            sma50 = price_data["sma_50"]
            if sma20 > sma50:
                ctx_parts.append(f"SMA(20)={sma20:.2f} > SMA(50)={sma50:.2f} (bullish cross)")
            else:
                ctx_parts.append(f"SMA(20)={sma20:.2f} < SMA(50)={sma50:.2f} (bearish cross)")
        if "atr_pct" in price_data:
            ctx_parts.append(f"ATR: {price_data['atr_pct']:.2f}% of price")

        if ctx_parts:
            parts.append("Quantitative context: " + ", ".join(ctx_parts))

    if extra_context:
        parts.append(f"Additional context: {extra_context}")

    parts.append("Focus on the RIGHT EDGE of the chart (current state). What regime is the market in RIGHT NOW?")

    return "\n".join(parts)


def build_multi_tf_prompt(
    asset: str,
    timeframes: list,
    extra_context: str = "",
) -> str:
    """Build prompt for multi-timeframe analysis."""
    parts = []
    parts.append(f"This chart shows {asset} at multiple timeframes: {', '.join(timeframes)}.")
    parts.append("Analyze ALL timeframes together to determine the current regime.")
    parts.append("Higher timeframes carry more weight for regime classification.")
    parts.append("Respond ONLY with valid JSON matching the schema in your instructions.")

    if extra_context:
        parts.append(f"Additional context: {extra_context}")

    return "\n".join(parts)
