#!/usr/bin/env python3
"""Test LLM regime detection with real market data from Hyperliquid.

Run from zpair directory (needs hyperliquid_utils_adapter):
    cd /Users/user/zpair
    python3 /Users/user/llm-regime/test_real_data.py

Tests multiple assets and windows, saves charts for visual verification.
"""

import sys
import os
import json
import time
import numpy as np

# Add zpair to path for Hyperliquid adapter
sys.path.insert(0, "/Users/user/zpair")
sys.path.insert(0, "/Users/user/llm-regime")

from llm_regime import RegimeAnalyzer
from llm_regime.charts import generate_chart_image
from hyperliquid_utils_adapter import HyperliquidAdapter


def load_credentials():
    for p in ["/Users/user/zpair/config_BTC_5m.json",
              "/Users/user/zpair/config_default.json",
              "/Users/user/zpair/config.json"]:
        if os.path.exists(p):
            with open(p) as f:
                cfg = json.load(f)
            hl = cfg.get("trading", {}).get("hyperliquid", {})
            pk, wa = hl.get("private_key", ""), hl.get("wallet_address", "")
            if pk and len(pk) > 60 and wa and len(wa) > 40:
                return pk, wa
    print("ERROR: No credentials found")
    sys.exit(1)


def fetch_candles(adapter, coin, tf, bars):
    raw = adapter.get_candles(coin, tf, bars)
    if not raw:
        return None
    opens = np.array([float(c["o"]) for c in raw])
    highs = np.array([float(c["h"]) for c in raw])
    lows = np.array([float(c["l"]) for c in raw])
    closes = np.array([float(c["c"]) for c in raw])
    volumes = np.array([float(c["v"]) for c in raw])
    return opens, highs, lows, closes, volumes


def main():
    pk, wa = load_credentials()
    adapter = HyperliquidAdapter(pk, wa)
    analyzer = RegimeAnalyzer(provider="gemini", cache_ttl=60)

    # Test configurations
    tests = [
        # (asset, timeframe, total_bars, analysis_window, description)
        ("BTC", "5m", 500, None, "BTC 5m full 500 bars"),
        ("BTC", "5m", 500, 100, "BTC 5m last 100 bars (500 context)"),
        ("SOL", "5m", 500, None, "SOL 5m full 500 bars"),
        ("SOL", "5m", 500, 100, "SOL 5m last 100 bars (500 context)"),
        ("ETH", "5m", 500, None, "ETH 5m full 500 bars"),
        ("BTC", "1h", 150, None, "BTC 1h full 150 bars"),
    ]

    print("=" * 70)
    print("  LLM REGIME - REAL MARKET DATA TEST")
    print("=" * 70)

    for i, (asset, tf, bars, window, desc) in enumerate(tests):
        if i > 0:
            print("  ... waiting 12s for rate limit ...")
            time.sleep(12)

        print(f"\n  Test {i+1}: {desc}")
        print(f"  {'-'*60}")

        data = fetch_candles(adapter, asset, tf, bars)
        if data is None:
            print(f"  ERROR: No data for {asset} {tf}")
            continue

        opens, highs, lows, closes, volumes = data
        pchg = (closes[-1] / closes[0] - 1) * 100
        print(f"  Data: {len(closes)} bars | {closes[0]:.2f} -> {closes[-1]:.2f} ({pchg:+.1f}%)")

        # Save chart for visual inspection
        chart_name = f"real_{asset}_{tf}_{bars}"
        if window:
            chart_name += f"_w{window}"
        png = generate_chart_image(
            np.arange(len(closes)), opens, highs, lows, closes, volumes,
            asset=asset, timeframe=tf, analysis_window=window,
        )
        chart_path = f"/Users/user/llm-regime/{chart_name}.png"
        with open(chart_path, "wb") as f:
            f.write(png)

        try:
            result = analyzer.analyze(
                asset=asset, timeframe=tf,
                opens=opens, highs=highs, lows=lows, closes=closes,
                volumes=volumes,
                analysis_window=window,
                force_refresh=True,
            )

            print(f"  Regime:     {result.regime}")
            print(f"  Confidence: {result.confidence}/5")
            print(f"  Direction:  {result.scalp_direction}")
            print(f"  Bias:       {result.bias}")
            print(f"  Volatility: {result.volatility}")
            print(f"  Reasoning:  {result.reasoning}")
            if result.key_levels:
                for kl in result.key_levels:
                    print(f"  Level:      {kl['type']} {kl['price']:.2f} ({kl['strength']})")
            print(f"  Latency:    {result.latency_ms}ms | Cost: ${result.cost_usd:.4f}")
            print(f"  Chart:      {chart_path}")

        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")

    print(f"\n{'='*70}")
    print("  DONE - Open the saved chart PNGs to visually verify each result")
    print("=" * 70)


if __name__ == "__main__":
    main()
