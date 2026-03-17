"""llm-regime: LLM-powered market regime detection.

Generates price charts and sends them to vision-capable LLMs for
regime classification. Supports Claude, GPT, Gemini, and local
models via Ollama.

Quick start:
    from llm_regime import RegimeAnalyzer

    analyzer = RegimeAnalyzer(provider="anthropic")
    result = analyzer.analyze(
        asset="BTC", timeframe="5m",
        opens=opens, highs=highs, lows=lows, closes=closes,
    )
    print(result.regime, result.scalp_direction)
"""

__version__ = "0.1.0"

from llm_regime.schema import RegimeResult, Regime, Volatility, TrendStrength
from llm_regime.analyzer import RegimeAnalyzer
from llm_regime.providers import create_provider

__all__ = [
    "RegimeAnalyzer",
    "RegimeResult",
    "Regime",
    "Volatility",
    "TrendStrength",
    "create_provider",
]
