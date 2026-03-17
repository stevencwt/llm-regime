#!/usr/bin/env python3
"""Example: Analyze BTC regime using Claude.

Usage:
    # With Anthropic (requires ANTHROPIC_API_KEY env var)
    python3 example_basic.py --provider anthropic --asset BTC

    # With OpenAI
    python3 example_basic.py --provider openai --asset BTC

    # With local Ollama (requires ollama running with llama3.2-vision)
    python3 example_basic.py --provider ollama --asset BTC

    # With custom data (from CSV)
    python3 example_basic.py --provider anthropic --csv data.csv
"""

import argparse
import numpy as np
from llm_regime import RegimeAnalyzer


def generate_sample_data(regime="uptrend", n=200):
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(42)

    if regime == "uptrend":
        drift = 0.002
    elif regime == "downtrend":
        drift = -0.002
    else:
        drift = 0.0

    returns = drift + rng.randn(n) * 0.005
    closes = 100 * np.exp(np.cumsum(returns))
    opens = closes * (1 + rng.randn(n) * 0.001)
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.randn(n) * 0.002))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.randn(n) * 0.002))
    volumes = 1000 + rng.randint(0, 500, n).astype(float)

    return opens, highs, lows, closes, volumes


def main():
    parser = argparse.ArgumentParser(description="LLM Regime Detection Example")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "gemini", "ollama"])
    parser.add_argument("--model", default=None, help="Override default model")
    parser.add_argument("--api-key", default=None, help="API key (or use env vars)")
    parser.add_argument("--asset", default="BTC", help="Asset symbol")
    parser.add_argument("--regime", default="uptrend", choices=["uptrend", "downtrend", "ranging"])
    parser.add_argument("--bars", type=int, default=200, help="Number of bars")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  LLM Regime Analysis — {args.provider} / {args.asset}")
    print(f"{'='*60}\n")

    # Create analyzer
    analyzer = RegimeAnalyzer(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        cache_ttl=300,  # 5 min cache for testing
    )

    print(f"Provider: {analyzer.provider.name}")
    print(f"Model: {analyzer.provider.default_model}")
    print(f"Available: {analyzer.provider.is_available()}")

    # Generate or load data
    opens, highs, lows, closes, volumes = generate_sample_data(args.regime, args.bars)
    print(f"\nData: {len(closes)} bars, synthetic {args.regime}")
    print(f"Price: {closes[0]:.2f} → {closes[-1]:.2f} ({(closes[-1]/closes[0]-1)*100:+.1f}%)")

    # Analyze
    print(f"\nCalling LLM...")
    result = analyzer.analyze(
        asset=args.asset,
        timeframe="5m",
        opens=opens, highs=highs, lows=lows, closes=closes,
        volumes=volumes,
    )

    # Display result
    print(f"\n{'─'*60}")
    print(f"  RESULT")
    print(f"{'─'*60}")
    print(f"  Regime:          {result.regime}")
    print(f"  Confidence:      {result.confidence}/5")
    print(f"  Volatility:      {result.volatility}")
    print(f"  Trend strength:  {result.trend_strength}")
    print(f"  Bias:            {result.bias}")
    print(f"  Scalp direction: {result.scalp_direction}")
    print(f"  Pattern:         {result.pattern or 'none'}")
    print(f"  Reasoning:       {result.reasoning}")

    if result.key_levels:
        print(f"\n  Key levels:")
        for kl in result.key_levels:
            print(f"    {kl['type']:12s} {kl['price']:.2f} ({kl['strength']})")

    print(f"\n  Latency:   {result.latency_ms}ms")
    print(f"  Cost:      ${result.cost_usd:.4f}")
    print(f"  Provider:  {result.provider}/{result.model}")

    # Test caching
    print(f"\n  Calling again (should be cached)...")
    result2 = analyzer.analyze(
        asset=args.asset, timeframe="5m",
        opens=opens, highs=highs, lows=lows, closes=closes,
    )
    print(f"  Cached: {result2.regime} (latency: {result2.latency_ms}ms)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
