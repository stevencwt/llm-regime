#!/usr/bin/env python3
"""Test different bar counts and timeframes to find optimal LLM config.

Sends the SAME market moment with different chart configurations
to see which gives the most accurate/useful regime classification.

Run from zpair directory:
    cd /Users/user/zpair
    python3 /Users/user/llm-regime/test_bars_comparison.py
"""

import sys
import os
import json
import time
import numpy as np

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
    sys.exit("No credentials found")


def fetch_candles(adapter, coin, tf, bars):
    raw = adapter.get_candles(coin, tf, bars)
    if not raw:
        return None
    return {
        "opens": np.array([float(c["o"]) for c in raw]),
        "highs": np.array([float(c["h"]) for c in raw]),
        "lows": np.array([float(c["l"]) for c in raw]),
        "closes": np.array([float(c["c"]) for c in raw]),
        "volumes": np.array([float(c["v"]) for c in raw]),
    }


def run_test(analyzer, label, asset, tf, data, total_bars, window, chart_prefix):
    """Run a single regime analysis and save chart."""
    n = min(total_bars, len(data["closes"]))
    o = data["opens"][-n:]
    h = data["highs"][-n:]
    l = data["lows"][-n:]
    c = data["closes"][-n:]
    v = data["volumes"][-n:]

    pchg = (c[-1] / c[0] - 1) * 100
    if window:
        w_pchg = (c[-1] / c[-min(window, n)] - 1) * 100
    else:
        w_pchg = pchg

    png = generate_chart_image(
        np.arange(n), o, h, l, c, v,
        asset=asset, timeframe=tf, analysis_window=window,
    )
    chart_path = f"/Users/user/llm-regime/{chart_prefix}.png"
    with open(chart_path, "wb") as f:
        f.write(png)

    result = analyzer.analyze(
        asset=asset, timeframe=tf,
        opens=o, highs=h, lows=l, closes=c, volumes=v,
        analysis_window=window,
        force_refresh=True,
    )

    return {
        "label": label,
        "bars": n,
        "window": window,
        "tf": tf,
        "price_chg": pchg,
        "window_chg": w_pchg,
        "regime": result.regime,
        "confidence": result.confidence,
        "direction": result.scalp_direction,
        "reasoning": result.reasoning,
        "latency": result.latency_ms,
        "chart": chart_path,
    }


def main():
    pk, wa = load_credentials()
    adapter = HyperliquidAdapter(pk, wa)
    analyzer = RegimeAnalyzer(provider="gemini", cache_ttl=10)

    # Fetch data for both timeframes
    print("Fetching BTC data...")
    btc_5m = fetch_candles(adapter, "BTC", "5m", 500)
    time.sleep(1)
    btc_1m = fetch_candles(adapter, "BTC", "1m", 500)

    if btc_5m is None or btc_1m is None:
        sys.exit("Failed to fetch data")

    print(f"  BTC 5m: {len(btc_5m['closes'])} bars, "
          f"{btc_5m['closes'][0]:.0f} -> {btc_5m['closes'][-1]:.0f}")
    print(f"  BTC 1m: {len(btc_1m['closes'])} bars, "
          f"{btc_1m['closes'][0]:.0f} -> {btc_1m['closes'][-1]:.0f}")

    # ============================================================
    # TEST MATRIX
    # ============================================================
    tests = [
        # --- 5m timeframe, varying bar counts ---
        ("5m x 100 bars (8h, no window)", "BTC", "5m", btc_5m, 100, None, "cmp_5m_100"),
        ("5m x 200 bars (17h, no window)", "BTC", "5m", btc_5m, 200, None, "cmp_5m_200"),
        ("5m x 300 bars (25h, no window)", "BTC", "5m", btc_5m, 300, None, "cmp_5m_300"),
        ("5m x 500 bars (42h, no window)", "BTC", "5m", btc_5m, 500, None, "cmp_5m_500"),

        # --- 5m with analysis window ---
        ("5m x 200 bars, window 50 (4h focus)", "BTC", "5m", btc_5m, 200, 50, "cmp_5m_200_w50"),
        ("5m x 300 bars, window 100 (8h focus)", "BTC", "5m", btc_5m, 300, 100, "cmp_5m_300_w100"),

        # --- 1m timeframe comparison ---
        ("1m x 100 bars (1.7h, no window)", "BTC", "1m", btc_1m, 100, None, "cmp_1m_100"),
        ("1m x 300 bars (5h, no window)", "BTC", "1m", btc_1m, 300, None, "cmp_1m_300"),
        ("1m x 500 bars (8h, no window)", "BTC", "1m", btc_1m, 500, None, "cmp_1m_500"),
    ]

    print(f"\n{'='*78}")
    print(f"  BAR COUNT & TIMEFRAME COMPARISON — BTC (same market moment)")
    print(f"{'='*78}")

    results = []
    for i, (label, asset, tf, data, bars, window, prefix) in enumerate(tests):
        if i > 0:
            print("  ... waiting 12s ...")
            time.sleep(12)

        print(f"\n  [{i+1}/{len(tests)}] {label}")
        try:
            r = run_test(analyzer, label, asset, tf, data, bars, window, prefix)
            results.append(r)
            print(f"    Regime: {r['regime']} (conf={r['confidence']}) dir={r['direction']}")
            print(f"    Price: {r['price_chg']:+.1f}% total, {r['window_chg']:+.1f}% in window")
            print(f"    Reason: {r['reasoning'][:100]}")
            print(f"    Latency: {r['latency']}ms | Chart: {r['chart']}")
        except Exception as e:
            print(f"    ERROR: {str(e)[:100]}")

    # Summary table
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"  {'Config':<40} {'Regime':<20} {'Conf':>4}  {'Direction':<12} {'Window chg':>10}")
    print(f"  {'-'*40} {'-'*20} {'-'*4}  {'-'*12} {'-'*10}")
    for r in results:
        w = f"w{r['window']}" if r['window'] else "full"
        label = f"{r['tf']} x {r['bars']} ({w})"
        print(f"  {label:<40} {r['regime']:<20} {r['confidence']:>4}  {r['direction']:<12} {r['window_chg']:>+9.1f}%")

    print(f"\n  Charts saved as cmp_*.png — open them to compare visual clarity")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
