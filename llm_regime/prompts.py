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


SYSTEM_PROMPT = """You are an expert technical analyst for a live crypto trading bot. Analyze the candlestick chart and return ONLY a single valid JSON object — no prose, no markdown, no code fences.

MANDATORY: Your response must match this exact structure. Every field is required. Never return an empty array for key_levels — always identify at least 2–3 price levels from the chart.

{
  "regime": "STRONG_UPTREND|UPTREND|WEAK_UPTREND|RANGING|WEAK_DOWNTREND|DOWNTREND|STRONG_DOWNTREND|TRANSITION",
  "confidence": 1-5,
  "volatility": "LOW|MODERATE|HIGH|EXTREME",
  "trend_strength": "STRONG|MODERATE|WEAK|NONE",
  "bias": "BULLISH|BEARISH|NEUTRAL",
  "scalp_direction": "LONG_ONLY|SHORT_ONLY|BOTH|NO_TRADE",
  "key_levels": [
    {"price": 83200, "type": "support", "strength": "strong|moderate|weak", "reason": "one phrase explaining the level"},
    {"price": 85500, "type": "resistance", "strength": "strong|moderate|weak", "reason": "one phrase explaining the level"}
  ],
  "reasoning": "1-3 sentences citing specific visual features observed",
  "pattern": "pattern name or empty string",
  "structure": {
    "hh_hl": true,
    "lh_ll": false,
    "broken": false,
    "summary": "one sentence describing current swing structure"
  },
  "nearest_zone": {
    "price": 83200,
    "type": "support|resistance",
    "quality": "high|moderate|low",
    "distance_pct": 1.43,
    "rationale": "one sentence explaining why this zone matters",
    "tradeable": true
  }
}

REGIME DEFINITIONS:
- STRONG_UPTREND: Clear HH+HL sequence, price well above rising SMA, strong momentum
- UPTREND: Generally rising with pullbacks, price above SMA most of the time
- WEAK_UPTREND: Slight upward tilt, choppy, price near SMA
- RANGING: Price bouncing between visible ceiling and floor, SMA flat
- WEAK_DOWNTREND: Slight downward tilt, choppy, frequent bounces
- DOWNTREND: Generally falling with bounces, price below SMA
- STRONG_DOWNTREND: Clear LH+LL sequence, price well below falling SMA
- TRANSITION: Structure actively breaking — prior range breaking out/down

SCALP_DIRECTION RULES:
- STRONG_UPTREND/UPTREND/WEAK_UPTREND → LONG_ONLY
- STRONG_DOWNTREND/DOWNTREND/WEAK_DOWNTREND → SHORT_ONLY
- RANGING → BOTH
- TRANSITION → NO_TRADE

CONFIDENCE: 1=ambiguous, 2=uncertain, 3=moderate, 4=clear with evidence, 5=textbook. Be conservative — do not assign 4 or 5 unless the regime is obvious from multiple visual cues.

KEY_LEVELS RULES (CRITICAL):
- ALWAYS return 2–4 levels. Returning an empty array [] is not acceptable.
- Look for: prior swing highs/lows, consolidation bases/tops, round numbers with price reaction, SMA confluences, broken S/R levels that have flipped.
- For a DOWNTREND: the prior consolidation base that broke is resistance; recent swing lows are support; prior lower highs are resistance.
- For an UPTREND: prior swing highs that broke are support; recent swing highs are resistance.
- Each level must have a "reason" — a short phrase stating what makes it significant visually.

STRUCTURE RULES:
- hh_hl: true only if you can see higher highs AND higher lows in the highlighted window
- lh_ll: true only if you can see lower highs AND lower lows in the highlighted window
- hh_hl and lh_ll cannot both be true
- broken: true if a significant prior swing point was just taken out
- summary: one sentence describing the swing sequence you actually see

NEAREST_ZONE RULES:
- Identify the single most significant price level closest to current price
- In a downtrend: the nearest overhead resistance (prior consolidation base, recent lower high)
- In an uptrend: the nearest support below current price
- In RANGING: the nearest range boundary
- distance_pct: approximate % from current price to the zone (always positive)
- tradeable: true if the zone is within ~3% of current price AND quality is high or moderate
- If truly no zone is visible, set quality to "low", tradeable to false, and explain in rationale

Focus on the RIGHT EDGE of the chart — classify what is happening NOW."""


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
