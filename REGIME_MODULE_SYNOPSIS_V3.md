# Regime Detection Module — Integration Guide for AI Systems (v3)

> **Purpose**: This document provides everything an AI assistant needs to install,
> integrate, and validate both the quantitative `regime-detection` package and the
> LLM-based `llm-regime` package in any trading bot.
>
> **What changed in v3**: Added `llm-regime` as a direction oracle that sits on top
> of the existing quant regime system. The quant system is NOT replaced — it continues
> to provide real-time volatility, liquidity, range hints, and exit mandates. The LLM
> adds superior direction detection via chart vision analysis.

---

## 1. What This System Does (Two-Tier Architecture)

### Tier 1: Quantitative Regime Detection (`regime-detection`)

A **pure computation** Python package that classifies market regime in real-time.
Runs on every tick. Provides:
- HMM (2D) → volatility regime classifier (NOT direction — see Section 7.4)
- DFA Hurst → trending vs mean-reverting
- CPD → structural break detection
- Drift detection → direction fallback (SMA slope + swing structure, ~85% accuracy)
- Volatility regime → EXPANDING / CONTRACTING / LOW_STABLE / MODERATE
- Liquidity → CONSOLIDATION / TRAP / PASSED
- Range hints → Donchian/Keltner boundaries
- Exit mandate → force close on regime shift
- **199 passing tests**

### Tier 2: LLM Vision Regime Detection (`llm-regime`)

A **chart-image-based** Python package that sends candlestick charts to vision-capable
LLMs (Gemini, Claude, GPT, Ollama) for regime classification. Runs periodically
(every 30 min for LTF, every 2h for HTF). Provides:
- Direction classification → STRONG_UPTREND through STRONG_DOWNTREND
- Scalp direction → LONG_ONLY / SHORT_ONLY / BOTH / NO_TRADE
- Key levels → support/resistance from visual chart analysis
- Pattern recognition → head & shoulders, triangles, breakouts
- Confidence → 1–5 scale
- **24 passing tests, ~90% accuracy validated on real BTC/SOL/ETH data**

### The Combined Pipeline

```
                    ┌─────────────────────────────────────────────┐
                    │  YOUR TRADING BOT (main loop, every tick)   │
                    └───────────────────┬─────────────────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              │                         │                          │
              ▼                         ▼                          ▼
  ┌───────────────────┐   ┌──────────────────────┐   ┌──────────────────┐
  │ RegimeManager     │   │ LLM Regime Bridge    │   │ Combined Layer   │
  │ (every tick)      │   │ (every 30 min)       │   │ (merges both)    │
  │                   │   │                      │   │                  │
  │ Hurst       ──┐   │   │ HTF: 1h × 150 bars  │   │ direction = LLM  │
  │ HMM 2D     ──┤   │   │   → UPTREND/etc     │   │   (or drift if   │
  │ CPD        ──┤   │   │                      │   │    LLM stale)    │
  │ Drift      ──┤   │   │ LTF: 5m × 200 w60   │   │                  │
  │ Vol regime ──┤   │   │   → DOWNTREND/etc    │   │ volatility =     │
  │ Liquidity  ──┤   │   │                      │   │   quant (always)  │
  │ Range hints──┤   │   │ Cached, async,       │   │                  │
  │ Exit mandate─┘   │   │ never blocks loop    │   │ exit_mandate =   │
  │                   │   │                      │   │   quant (always)  │
  └───────────────────┘   └──────────────────────┘   └──────────────────┘
```

---

## 2. Package Locations & Installation

### 2.1 Quantitative Tier

```
GitHub: https://github.com/stevencwt/regime-detection
Local:  /Users/user/regime-detection
```

```bash
pip install -e /Users/user/regime-detection
python -c "from regime_detection import RegimeManager; print('OK')"
```

Dependencies: numpy, pandas, pyyaml, hmmlearn, ruptures, scipy. Optional: fathon.

### 2.2 LLM Vision Tier

```
Local: /Users/user/llm-regime
```

```bash
pip install -e "/Users/user/llm-regime[google]"   # Gemini (recommended — cheapest)
# or: pip install -e "/Users/user/llm-regime[anthropic]"  # Claude
# or: pip install -e "/Users/user/llm-regime[ollama]"     # Local (free)
# or: pip install -e "/Users/user/llm-regime[all]"        # All providers

python -c "from llm_regime import RegimeAnalyzer; print('OK')"
```

Dependencies: numpy, matplotlib, Pillow, pyyaml. Provider-specific: google-genai / anthropic / openai / ollama.

### 2.3 Environment Variables

```bash
# For Gemini (recommended)
export GOOGLE_API_KEY="your_key_here"

# For Claude (alternative)
export ANTHROPIC_API_KEY="sk-ant-..."

# For OpenAI (alternative)
export OPENAI_API_KEY="sk-..."

# For Ollama (no key needed, just run ollama serve)
export OLLAMA_BASE_URL="http://localhost:11434"  # default
```

---

## 3. Quantitative Tier — Public API Reference

*(Unchanged from v2 — see RegimeManager constructor, .update(), .get_current_regime(), .get_json(), .reload_config(), .bar_count)*

---

## 4. LLM Vision Tier — Public API Reference

### 4.1 Constructor

```python
from llm_regime import RegimeAnalyzer

analyzer = RegimeAnalyzer(
    provider="gemini",           # "gemini" | "anthropic" | "openai" | "ollama"
    model=None,                  # None = provider default (gemini-2.5-flash)
    api_key=None,                # None = read from env var
    cache_ttl=1800,              # 30 min cache (default)
    cache_path=None,             # Optional file path for persistent cache
    temperature=0.1,             # Low for consistency
    sma_periods=(20, 50),        # SMAs drawn on chart
    chart_dpi=100,               # Chart resolution
)
```

### 4.2 .analyze() — Single Chart Analysis

```python
result = analyzer.analyze(
    asset="BTC",
    timeframe="5m",
    opens=np.array([...]),
    highs=np.array([...]),
    lows=np.array([...]),
    closes=np.array([...]),
    volumes=np.array([...]),     # optional
    analysis_window=60,          # optional — focus on last N bars
    force_refresh=False,         # bypass cache
    extra_context="",            # additional prompt context
)
```

Returns `RegimeResult` with: regime, confidence, bias, scalp_direction, key_levels, reasoning, pattern, latency_ms, cost_usd.

### 4.3 .analyze_multi_scale() — Two-Chart Macro+Micro Analysis

```python
result = analyzer.analyze_multi_scale(
    asset="BTC", timeframe="5m",
    opens=opens, highs=highs, lows=lows, closes=closes,
    macro_bars=500, micro_bars=100,
)

print(result["macro"].regime)          # "UPTREND"
print(result["micro"].regime)          # "DOWNTREND"
print(result["combined_direction"])    # "WAIT" (pullback in uptrend)
```

### 4.4 .analyze_from_chart() — External Chart Image

```python
with open("tradingview_screenshot.png", "rb") as f:
    result = analyzer.analyze_from_chart(
        chart_png=f.read(),
        asset="BTC", timeframe="1h",
    )
```

### 4.5 RegimeResult Properties

```python
result.is_bullish    # True if STRONG_UPTREND / UPTREND / WEAK_UPTREND
result.is_bearish    # True if STRONG_DOWNTREND / DOWNTREND / WEAK_DOWNTREND
result.is_ranging    # True if RANGING
result.should_trade  # True if scalp_direction != NO_TRADE and confidence >= 2
```

---

## 5. Output Schema — Quantitative Tier (v3.1 JSON)

*(Unchanged from v2 — see consensus_state, volatility_regime, signals, recommended_logic, exit_mandate)*

---

## 6. Output Schema — LLM Vision Tier (RegimeResult)

```json
{
  "regime": "UPTREND",
  "confidence": 4,
  "volatility": "MODERATE",
  "trend_strength": "MODERATE",
  "bias": "BULLISH",
  "scalp_direction": "LONG_ONLY",
  "key_levels": [
    {"price": 75500, "type": "resistance", "strength": "strong"},
    {"price": 73500, "type": "support", "strength": "strong"}
  ],
  "reasoning": "The market exhibits clear higher highs and higher lows...",
  "pattern": "",
  "asset": "BTC",
  "timeframe": "1h",
  "bars_analyzed": 150,
  "provider": "gemini",
  "model": "gemini-2.5-flash",
  "latency_ms": 9789,
  "cost_usd": 0.0009
}
```

### 6.1 Regime Values

| Value | Meaning | Scalp Direction |
|---|---|---|
| STRONG_UPTREND | Clear HH+HL, price well above rising SMAs | LONG_ONLY |
| UPTREND | Generally rising with pullbacks | LONG_ONLY |
| WEAK_UPTREND | Slight upward bias, choppy | LONG_ONLY (cautious) |
| RANGING | Oscillating between S/R, flat SMAs | BOTH |
| WEAK_DOWNTREND | Slight downward bias | SHORT_ONLY (cautious) |
| DOWNTREND | Generally falling with bounces | SHORT_ONLY |
| STRONG_DOWNTREND | Clear LH+LL, price well below falling SMAs | SHORT_ONLY |
| TRANSITION | Market structure changing | NO_TRADE |

---

## 7. Key Technical Findings (v3 — Important for Integrators)

### 7.1 HMM Is a Volatility Classifier, NOT Direction

The 2D HMM (returns + volatility) clusters by volatility magnitude. High-vol events (both UP spikes and DOWN crashes) group together. **Do not use HMM BULL/BEAR labels for directional trading decisions.** Use drift detection or LLM vision instead.

### 7.2 5m Charts Superior to 1m for LLM

Validated empirically: 1m charts introduce noise that misleads LLMs into seeing micro-trends. 1m × 300 called UPTREND on -1.4% data; 5m × 200 correctly called TRANSITION on the same market moment.

### 7.3 Analysis Window Improves Accuracy

Sending 200 bars with a 60-bar analysis window (grayed context + colored focus) produces more accurate and consistent results than sending all bars equally weighted. The LLM correctly focuses on the highlighted section while using the context for broader trend understanding.

### 7.4 Hurst Thresholds Lowered for 5m Crypto

At 5m crypto, Hurst rarely exceeds 0.55 even during strong trends. Thresholds adjusted:
- `trending_threshold`: 0.60 → 0.50
- `range_min_hurst`: 0.48 → 0.44
- `range_max_hurst`: 0.54 → 0.49

---

## 8. Integration Pattern — Combined System

### Step 1: Keep Existing Quant Integration

If the bot already has `regime_bridge.py` with `RegimeBridge` using `RegimeManager`, **keep it unchanged**. It continues to provide real-time volatility, liquidity, range hints, and exit mandates.

### Step 2: Add LLM Bridge

Create `llm_regime_bridge.py` alongside the existing `regime_bridge.py`:

```python
"""
llm_regime_bridge.py — LLM vision regime detection.
Runs periodically (not on every tick). Results are cached.
"""
import time
import threading
import logging
from typing import Optional, Dict, Any

import numpy as np

try:
    from llm_regime import RegimeAnalyzer, RegimeResult
    LLM_REGIME_AVAILABLE = True
except ImportError:
    LLM_REGIME_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMRegimeBridge:
    """Periodic LLM-based regime detection.

    Runs two analyses:
      - HTF (1h × 150 bars): directional filter, updated every 2 hours
      - LTF (5m × 200 bars, window 60): current state, updated every 30 min

    Results are cached and read by the combined layer on each tick.
    """

    def __init__(
        self,
        provider: str = "gemini",
        htf_interval: int = 7200,    # 2 hours
        ltf_interval: int = 1800,    # 30 min
        min_confidence: int = 3,     # below this, fall back to quant
    ):
        self.enabled = LLM_REGIME_AVAILABLE
        if not self.enabled:
            logger.warning("llm-regime not installed. LLM direction disabled.")
            return

        self.analyzer = RegimeAnalyzer(
            provider=provider,
            cache_ttl=max(htf_interval, ltf_interval) + 300,
        )
        self.htf_interval = htf_interval
        self.ltf_interval = ltf_interval
        self.min_confidence = min_confidence

        self._htf_result: Optional[RegimeResult] = None
        self._ltf_result: Optional[RegimeResult] = None
        self._htf_last_update: float = 0
        self._ltf_last_update: float = 0

    def maybe_update(
        self,
        asset: str,
        opens_5m: np.ndarray,
        highs_5m: np.ndarray,
        lows_5m: np.ndarray,
        closes_5m: np.ndarray,
        volumes_5m: Optional[np.ndarray] = None,
        opens_1h: Optional[np.ndarray] = None,
        highs_1h: Optional[np.ndarray] = None,
        lows_1h: Optional[np.ndarray] = None,
        closes_1h: Optional[np.ndarray] = None,
        volumes_1h: Optional[np.ndarray] = None,
    ):
        """Check if LTF or HTF need updating. Call this from the bot loop."""
        if not self.enabled:
            return

        now = time.time()

        # LTF update (5m × 200, window 60)
        if now - self._ltf_last_update >= self.ltf_interval:
            if len(closes_5m) >= 200:
                try:
                    self._ltf_result = self.analyzer.analyze(
                        asset=asset, timeframe="5m",
                        opens=opens_5m[-200:], highs=highs_5m[-200:],
                        lows=lows_5m[-200:], closes=closes_5m[-200:],
                        volumes=volumes_5m[-200:] if volumes_5m is not None else None,
                        analysis_window=60,
                        force_refresh=True,
                    )
                    self._ltf_last_update = now
                    logger.info(
                        "LLM LTF: %s conf=%d dir=%s (%dms $%.4f)",
                        self._ltf_result.regime, self._ltf_result.confidence,
                        self._ltf_result.scalp_direction,
                        self._ltf_result.latency_ms, self._ltf_result.cost_usd,
                    )
                except Exception as e:
                    logger.warning("LLM LTF failed: %s", e)

        # HTF update (1h × 150)
        if now - self._htf_last_update >= self.htf_interval:
            if closes_1h is not None and len(closes_1h) >= 150:
                try:
                    self._htf_result = self.analyzer.analyze(
                        asset=asset, timeframe="1h",
                        opens=opens_1h[-150:], highs=highs_1h[-150:],
                        lows=lows_1h[-150:], closes=closes_1h[-150:],
                        volumes=volumes_1h[-150:] if volumes_1h is not None else None,
                        force_refresh=True,
                    )
                    self._htf_last_update = now
                    logger.info(
                        "LLM HTF: %s conf=%d dir=%s (%dms $%.4f)",
                        self._htf_result.regime, self._htf_result.confidence,
                        self._htf_result.scalp_direction,
                        self._htf_result.latency_ms, self._htf_result.cost_usd,
                    )
                except Exception as e:
                    logger.warning("LLM HTF failed: %s", e)

    @property
    def htf(self) -> Optional[RegimeResult]:
        return self._htf_result

    @property
    def ltf(self) -> Optional[RegimeResult]:
        return self._ltf_result

    @property
    def direction(self) -> str:
        """Combined direction from HTF × LTF matrix.

        Returns: LONG_ONLY | SHORT_ONLY | BOTH | NO_TRADE | WAIT
        """
        if not self.enabled:
            return "BOTH"  # no opinion

        htf = self._htf_result
        ltf = self._ltf_result

        # If either is missing or low confidence, return no opinion
        if htf is None or htf.confidence < self.min_confidence:
            if ltf is not None and ltf.confidence >= self.min_confidence:
                return ltf.scalp_direction
            return "BOTH"

        if ltf is None or ltf.confidence < self.min_confidence:
            return htf.scalp_direction

        # Both available — apply decision matrix
        if htf.regime == "TRANSITION":
            return "NO_TRADE"

        if htf.is_bullish:
            if ltf.is_bullish:
                return "LONG_ONLY"
            elif ltf.is_ranging:
                return "LONG_ONLY"  # buy dips in uptrend
            else:
                return "WAIT"  # pullback — don't fight HTF but don't enter
        elif htf.is_bearish:
            if ltf.is_bearish:
                return "SHORT_ONLY"
            elif ltf.is_ranging:
                return "SHORT_ONLY"  # sell rips in downtrend
            else:
                return "WAIT"
        elif htf.is_ranging:
            if ltf.is_bullish:
                return "LONG_ONLY"
            elif ltf.is_bearish:
                return "SHORT_ONLY"
            else:
                return "BOTH"
        else:
            return ltf.scalp_direction

    @property
    def is_stale(self) -> bool:
        """True if LTF hasn't been updated in > 1.5× the interval."""
        if self._ltf_last_update == 0:
            return True
        return time.time() - self._ltf_last_update > self.ltf_interval * 1.5
```

### Step 3: Create Combined Layer

```python
"""
regime_combined.py — Merges quant + LLM regime signals.
"""

class CombinedRegime:
    """Reads from both quant RegimeBridge and LLM LLMRegimeBridge."""

    def __init__(self, quant_bridge, llm_bridge):
        self.quant = quant_bridge
        self.llm = llm_bridge

    def get_state(self) -> dict:
        """Get combined regime state for entry/exit strategies."""
        q = self.quant.get_regime() if self.quant else {}
        llm_dir = self.llm.direction if self.llm and not self.llm.is_stale else None
        drift_dir = q.get("signals", {}).get("drift_direction", "NONE")

        # Direction: prefer LLM, fall back to drift
        if llm_dir and llm_dir not in ("BOTH",):
            direction = llm_dir
            direction_source = "llm"
        elif drift_dir in ("UP", "DOWN"):
            direction = "LONG_ONLY" if drift_dir == "UP" else "SHORT_ONLY"
            direction_source = "drift"
        else:
            direction = "BOTH"
            direction_source = "none"

        return {
            # Direction (from LLM or drift fallback)
            "direction": direction,
            "direction_source": direction_source,

            # From quant (always real-time)
            "consensus_state": q.get("consensus_state", "UNKNOWN"),
            "volatility_regime": q.get("volatility_regime", "UNKNOWN"),
            "exit_mandate": q.get("exit_mandate", False),
            "confidence_score": q.get("confidence_score", 0),
            "recommended_logic": q.get("recommended_logic", "NO_TRADE"),
            "range_hints": q.get("signals", {}).get("range_hints"),

            # LLM details (for logging)
            "llm_htf_regime": self.llm.htf.regime if self.llm and self.llm.htf else None,
            "llm_ltf_regime": self.llm.ltf.regime if self.llm and self.llm.ltf else None,
            "llm_confidence": self.llm.ltf.confidence if self.llm and self.llm.ltf else None,
        }
```

### Step 4: Wire into bot main loop (minimal changes)

```python
# --- In your bot's main file ---

# BEFORE (existing):
from regime_bridge import RegimeBridge
regime_bridge = RegimeBridge(...)

# ADD (new):
from llm_regime_bridge import LLMRegimeBridge
from regime_combined import CombinedRegime

llm_bridge = LLMRegimeBridge(provider="gemini")
combined = CombinedRegime(regime_bridge, llm_bridge)

# IN THE LOOP — add one call:
llm_bridge.maybe_update(
    asset="BTC",
    opens_5m=candle_buffer_5m.opens,
    highs_5m=candle_buffer_5m.highs,
    lows_5m=candle_buffer_5m.lows,
    closes_5m=candle_buffer_5m.closes,
    volumes_5m=candle_buffer_5m.volumes,
    opens_1h=candle_buffer_1h.opens,   # if available
    highs_1h=candle_buffer_1h.highs,
    lows_1h=candle_buffer_1h.lows,
    closes_1h=candle_buffer_1h.closes,
    volumes_1h=candle_buffer_1h.volumes,
)

# CHANGE entry strategy to read combined state:
state = combined.get_state()
direction = state["direction"]        # LONG_ONLY / SHORT_ONLY / BOTH / WAIT / NO_TRADE
exit_mandate = state["exit_mandate"]  # from quant (real-time)
```

---

## 9. Decision Matrix: HTF × LTF

| HTF (1h) | LTF (5m w60) | Combined | Action |
|---|---|---|---|
| UPTREND+ | UPTREND+ | **STRONG LONG** | Aggressive longs, tight stops |
| UPTREND+ | RANGING | **LONG BIAS** | Buy dips only |
| UPTREND+ | DOWNTREND | **WAIT** | Pullback — don't catch falling knife |
| UPTREND+ | TRANSITION | **WAIT** | Pause until micro resolves |
| DOWNTREND+ | DOWNTREND+ | **STRONG SHORT** | Aggressive shorts |
| DOWNTREND+ | RANGING | **SHORT BIAS** | Sell rips only |
| DOWNTREND+ | UPTREND | **WAIT** | Counter-trend bounce |
| RANGING | UPTREND | **CAUTIOUS LONG** | Quick targets |
| RANGING | DOWNTREND | **CAUTIOUS SHORT** | Quick targets |
| RANGING | RANGING | **BOTH** | Mean reversion |
| TRANSITION | Any | **NO TRADE** | Market changing |

---

## 10. Existing Integration: zpair (Hyperliquid)

*(Updated from v2 to note LLM integration is designed but not yet wired into production)*

---

## 11. Troubleshooting

### Quantitative Tier
*(Unchanged from v2)*

### LLM Vision Tier

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: google.genai` | Wrong package installed | `pip install google-genai` (not google-generativeai) |
| 429 RESOURCE_EXHAUSTED | Free tier limit (20/day) | Enable billing at aistudio.google.com → API keys |
| Truncated JSON (UNKNOWN result) | Gemini cut off response | max_tokens increased to 2048; fallback parser extracts fields from partial JSON |
| UPTREND on 1m but TRANSITION on 5m | 1m noise misleads LLM | Use 5m charts (validated superior) |
| LLM says DOWNTREND but quant says BULL | Different signals — LLM sees right-edge pullback | Combined layer handles this: HTF UPTREND + LTF DOWNTREND = WAIT |
| High latency (>15s) | Gemini 2.5 Flash uses thinking tokens | Normal for complex charts; results are cached |
| LLM stale (>45 min old) | API failure or rate limit | Falls back to drift detection automatically |

---

## 12. Cost Analysis

| Provider | Model | Cost/call | Daily (3 assets) | Monthly |
|---|---|---|---|---|
| **Gemini** | 2.5 Flash | $0.0008 | $0.14 | **$4.50** |
| Anthropic | Sonnet 4 | $0.015 | $2.70 | $81 |
| Ollama | llama3.2-vision | Free | Free | **Free** |

Gemini recommended for production. Ollama for development/testing.

---

## 13. Key Design Decisions

1. **LLM does NOT replace quant** — it adds direction detection on top. Quant provides real-time signals the LLM cannot.
2. **Graceful degradation** — if LLM is unavailable, drift detection provides ~85% accurate direction fallback. Bot never stops.
3. **Cached, never blocking** — LLM calls take 7-15s but results are cached. The trading loop reads cached results in <1ms.
4. **HTF overrides LTF direction** — macro trend always wins. Don't short a 1h uptrend because of a 5m pullback.
5. **5m charts, not 1m** — validated empirically. 1m noise misleads the LLM.
6. **200 bars with window 60** — optimal balance of context and focus for 5m analysis.
7. **Image beats text for charts** — chart images cost ~560 tokens (fixed), raw OHLCV text costs 15,000-20,000 tokens. Image is cheaper AND more accurate.
