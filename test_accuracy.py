#!/usr/bin/env python3
"""Test LLM regime accuracy against 6 known synthetic scenarios.

Uses realistic OHLCV generation where:
  - Open = previous close (no gaps)
  - High/Low derived from intra-bar range
  - Body and wick proportions match real markets
"""

from llm_regime import RegimeAnalyzer
from llm_regime.charts import generate_chart_image
import numpy as np
import time

analyzer = RegimeAnalyzer(provider='gemini', cache_ttl=10)


def make_realistic_ohlcv(n=200, drift=0.0, volatility=0.005, seed=42):
    """Generate realistic OHLCV data."""
    rng = np.random.RandomState(seed)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)
    volumes = np.zeros(n)
    opens[0] = 100.0
    for i in range(n):
        if i > 0:
            opens[i] = closes[i - 1]
        ret = drift + rng.randn() * volatility
        closes[i] = opens[i] * (1 + ret)
        body_top = max(opens[i], closes[i])
        body_bot = min(opens[i], closes[i])
        upper_wick = abs(rng.randn()) * volatility * opens[i] * 0.5
        lower_wick = abs(rng.randn()) * volatility * opens[i] * 0.5
        highs[i] = body_top + upper_wick
        lows[i] = body_bot - lower_wick
        base_vol = 1000
        vol_spike = abs(ret) / max(volatility, 1e-10)
        volumes[i] = base_vol * (1 + vol_spike) + rng.randint(0, 200)
    return opens, highs, lows, closes, volumes


def make_composite(parts, seed_start=50):
    """Join multiple OHLCV segments seamlessly."""
    all_o, all_h, all_l, all_c, all_v = [], [], [], [], []
    last_close = None
    for i, (n, drift, vol) in enumerate(parts):
        o, h, l, c, v = make_realistic_ohlcv(n, drift, vol, seed=seed_start + i)
        if last_close is not None:
            scale = last_close / o[0]
            o, h, l, c = o * scale, h * scale, l * scale, c * scale
        last_close = c[-1]
        all_o.append(o); all_h.append(h); all_l.append(l); all_c.append(c); all_v.append(v)
    return (np.concatenate(all_o), np.concatenate(all_h),
            np.concatenate(all_l), np.concatenate(all_c), np.concatenate(all_v))


tests = []

# 1. STRONG UPTREND
o, h, l, c, v = make_realistic_ohlcv(200, drift=0.002, volatility=0.004, seed=42)
tests.append(('STRONG_UPTREND', 'Steady climb', o, h, l, c, v))

# 2. STRONG DOWNTREND
o, h, l, c, v = make_realistic_ohlcv(200, drift=-0.002, volatility=0.004, seed=43)
tests.append(('STRONG_DOWNTREND,DOWNTREND', 'Steady decline', o, h, l, c, v))

# 3. RANGING
o, h, l, c, v = make_realistic_ohlcv(200, drift=0.0, volatility=0.003, seed=44)
tests.append(('RANGING,WEAK_UPTREND,WEAK_DOWNTREND', 'Flat oscillation', o, h, l, c, v))

# 4. WEAK UPTREND
o, h, l, c, v = make_realistic_ohlcv(200, drift=0.0004, volatility=0.005, seed=45)
tests.append(('WEAK_UPTREND,UPTREND,RANGING', 'Slight drift up noisy', o, h, l, c, v))

# 5. TRANSITION: flat then breakout
o, h, l, c, v = make_composite([(150, 0.0, 0.002), (50, 0.004, 0.005)], seed_start=60)
tests.append(('TRANSITION,UPTREND,STRONG_UPTREND', 'Flat then breakout', o, h, l, c, v))

# 6. V-RECOVERY
o, h, l, c, v = make_composite([(100, -0.003, 0.004), (100, 0.003, 0.004)], seed_start=70)
tests.append(('UPTREND,STRONG_UPTREND,TRANSITION', 'V-recovery now rising', o, h, l, c, v))

print('=' * 70)
print('  LLM REGIME ACCURACY TEST - 6 Scenarios (realistic candles)')
print('=' * 70)

correct = 0
total_run = 0
for i, (expected, desc, opens, highs, lows, closes, volumes) in enumerate(tests):
    if i > 0:
        print('  ... waiting 10s for rate limit ...')
        time.sleep(10)
    pchg = (closes[-1] / closes[0] - 1) * 100
    png = generate_chart_image(
        np.arange(len(closes)), opens, highs, lows, closes, volumes,
        asset=f'Test{i+1}', timeframe='5m',
    )
    with open(f'test_chart_{i+1}.png', 'wb') as f:
        f.write(png)
    try:
        result = analyzer.analyze(
            asset=f'Test{i+1}', timeframe='5m',
            opens=opens, highs=highs, lows=lows, closes=closes, volumes=volumes,
            force_refresh=True,
        )
        acceptable = [x.strip() for x in expected.split(',')]
        match = result.regime in acceptable
        if match: correct += 1
        total_run += 1
        icon = 'Y' if match else 'X'
        print(f'  [{icon}] Test {i+1}: {desc} ({pchg:+.1f}%)')
        print(f'      Expected: {expected}')
        print(f'      Got:      {result.regime} (conf={result.confidence}) dir={result.scalp_direction}')
        print(f'      Reason:   {result.reasoning}')
    except Exception as e:
        print(f'  [E] Test {i+1}: {desc} - ERROR: {str(e)[:100]}')
    print()

print(f'  SCORE: {correct}/{total_run} ({correct/max(total_run,1)*100:.0f}%)')
print('  Charts saved as test_chart_1.png through test_chart_6.png')
print('=' * 70)
