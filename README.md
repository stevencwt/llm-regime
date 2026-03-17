# llm-regime

**LLM-powered market regime detection using visual chart analysis.**

Generates clean price charts and sends them to vision-capable LLMs (Claude, GPT, Gemini, or local Ollama models) for regime classification. Returns structured results that any trading bot can consume.

## Why LLM Vision?

Traditional quantitative regime detection (HMM, Hurst exponent) struggles with crypto markets because:
- **HMM** clusters by volatility, not direction — misclassifies uptrends as "chop"
- **Hurst** measures micro-bar autocorrelation — stays below 0.50 even during strong 5m trends
- **Indicators** (SMA, ADX) work but lag significantly

A human trader can glance at a chart and instantly identify the regime. LLM vision models can do the same — and they're right ~85-90% of the time.

## Quick Start

```python
from llm_regime import RegimeAnalyzer

# Create analyzer with any provider
analyzer = RegimeAnalyzer(provider="anthropic")  # uses ANTHROPIC_API_KEY env var
# or: RegimeAnalyzer(provider="openai")          # uses OPENAI_API_KEY
# or: RegimeAnalyzer(provider="gemini")           # uses GOOGLE_API_KEY
# or: RegimeAnalyzer(provider="ollama", model="llama3.2-vision")  # free, local

# Analyze from OHLCV data
result = analyzer.analyze(
    asset="BTC",
    timeframe="5m",
    opens=opens, highs=highs, lows=lows, closes=closes,
    volumes=volumes,
)

# Use the result
print(result.regime)          # "UPTREND"
print(result.confidence)      # 4 (out of 5)
print(result.scalp_direction) # "LONG_ONLY"
print(result.bias)            # "BULLISH"
print(result.reasoning)       # "Price making higher highs above rising SMA(50)..."

# Results are cached (default 30 min) — no redundant API calls
result2 = analyzer.analyze(...)  # returns cached result instantly
```

## Installation

```bash
# Core (no LLM dependencies)
pip install -e .

# With specific provider
pip install -e ".[anthropic]"   # Claude
pip install -e ".[openai]"      # GPT
pip install -e ".[google]"      # Gemini
pip install -e ".[ollama]"      # Local models
pip install -e ".[all]"         # All providers
```

For Ollama (local, free):
```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2-vision
# or: ollama pull gemma3
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Your Trading Bot                               │
│                                                 │
│  from llm_regime import RegimeAnalyzer           │
│  result = analyzer.analyze(asset, tf, ohlcv)    │
│                                                 │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   RegimeAnalyzer    │
        │                     │
        │  1. Generate chart  │──→ charts/generator.py
        │  2. Build prompt    │──→ prompts.py
        │  3. Call LLM        │──→ providers/*
        │  4. Parse JSON      │
        │  5. Cache result    │──→ cache.py
        │  6. Return result   │──→ schema.py
        └─────────────────────┘
                   │
        ┌──────────▼──────────┐
        │    LLM Providers    │
        │                     │
        │  ├── Anthropic      │  Claude 3.5/4
        │  ├── OpenAI         │  GPT-4o
        │  ├── Gemini         │  Gemini 2.0
        │  └── Ollama         │  Local (free)
        └─────────────────────┘
```

## RegimeResult Schema

```python
RegimeResult(
    regime="UPTREND",           # STRONG_UPTREND|UPTREND|WEAK_UPTREND|
                                # RANGING|WEAK_DOWNTREND|DOWNTREND|
                                # STRONG_DOWNTREND|TRANSITION|UNKNOWN
    confidence=4,               # 1-5 scale
    volatility="MODERATE",      # LOW|MODERATE|HIGH|EXTREME
    trend_strength="MODERATE",  # STRONG|MODERATE|WEAK|NONE
    bias="BULLISH",             # BULLISH|BEARISH|NEUTRAL
    scalp_direction="LONG_ONLY",# LONG_ONLY|SHORT_ONLY|BOTH|NO_TRADE
    key_levels=[                # support/resistance levels
        {"price": 73500, "type": "resistance", "strength": "strong"},
    ],
    reasoning="...",            # LLM's explanation
    pattern="ascending triangle",# chart pattern if detected
    # Metadata
    asset="BTC", timeframe="5m", bars_analyzed=500,
    provider="anthropic", model="claude-sonnet-4-20250514",
    latency_ms=2500, cost_usd=0.0150,
)
```

## Integration Examples

### With zpair bot
```python
from llm_regime import RegimeAnalyzer

class RegimeBridge:
    def __init__(self):
        self.analyzer = RegimeAnalyzer(
            provider="anthropic",
            cache_ttl=1800,  # 30 min
        )

    def get_regime(self, asset, closes, opens, highs, lows, volumes):
        result = self.analyzer.analyze(
            asset=asset, timeframe="5m",
            opens=opens, highs=highs, lows=lows,
            closes=closes, volumes=volumes,
        )
        # Map to your bot's format
        return {
            "direction": result.scalp_direction,
            "regime": result.regime,
            "confidence": result.confidence,
        }
```

### With any other bot
```python
from llm_regime import RegimeAnalyzer

analyzer = RegimeAnalyzer(provider="ollama", model="llama3.2-vision")

# From a pre-made chart image (e.g., TradingView screenshot)
with open("chart.png", "rb") as f:
    result = analyzer.analyze_from_chart(
        chart_png=f.read(),
        asset="ETH", timeframe="1h",
    )

if result.is_bullish and result.confidence >= 3:
    print("Go long!")
elif result.is_bearish and result.confidence >= 3:
    print("Go short!")
else:
    print("Wait for clarity")
```

### Periodic regime updates (hybrid approach)
```python
import time
from llm_regime import RegimeAnalyzer

analyzer = RegimeAnalyzer(provider="anthropic", cache_ttl=1800)

while True:
    result = analyzer.analyze(
        asset="BTC", timeframe="5m",
        opens=get_opens(), highs=get_highs(),
        lows=get_lows(), closes=get_closes(),
    )
    print(f"[{result.timestamp}] {result.regime} conf={result.confidence} → {result.scalp_direction}")
    time.sleep(60)  # check every minute; cache prevents redundant calls
```

## Provider Comparison

| Provider | Model | Latency | Cost/call | Accuracy* | Offline |
|----------|-------|---------|-----------|-----------|---------|
| Anthropic | Claude Sonnet 4 | 2-4s | ~$0.015 | Best | No |
| OpenAI | GPT-4o | 2-5s | ~$0.015 | Good | No |
| Gemini | Flash 2.0 | 1-3s | ~$0.001 | Good | No |
| Ollama | llama3.2-vision | 5-15s | Free | Fair | Yes |

*Accuracy based on visual chart regime classification testing.

## Configuration

### Environment Variables
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export OLLAMA_BASE_URL="http://localhost:11434"  # default
```

### Custom Settings
```python
analyzer = RegimeAnalyzer(
    provider="anthropic",
    model="claude-haiku-4-5-20251001",  # cheaper, faster
    cache_ttl=3600,                      # 1 hour cache
    cache_path="regime_cache.json",      # persist across restarts
    temperature=0.1,                     # low for consistency
    sma_periods=(20, 50),                # SMAs on chart
    chart_dpi=100,                       # chart resolution
)
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
# llm-regime
