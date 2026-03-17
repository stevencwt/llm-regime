"""Tests for llm-regime core components."""

import json
import numpy as np
import pytest

from llm_regime.schema import RegimeResult, Regime, Volatility
from llm_regime.charts import generate_chart_image, chart_to_base64
from llm_regime.prompts import SYSTEM_PROMPT, build_regime_prompt
from llm_regime.cache import RegimeCache
from llm_regime.providers import create_provider


class TestRegimeResult:
    def test_default_values(self):
        r = RegimeResult()
        assert r.regime == "UNKNOWN"
        assert r.confidence == 0
        assert r.bias == "NEUTRAL"

    def test_to_dict(self):
        r = RegimeResult(regime="UPTREND", confidence=4, bias="BULLISH")
        d = r.to_dict()
        assert d["regime"] == "UPTREND"
        assert d["confidence"] == 4

    def test_to_json(self):
        r = RegimeResult(regime="RANGING", confidence=3)
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["regime"] == "RANGING"

    def test_from_dict(self):
        d = {"regime": "DOWNTREND", "confidence": 2, "bias": "BEARISH"}
        r = RegimeResult.from_dict(d)
        assert r.regime == "DOWNTREND"
        assert r.is_bearish

    def test_is_bullish(self):
        assert RegimeResult(regime="UPTREND").is_bullish
        assert RegimeResult(regime="STRONG_UPTREND").is_bullish
        assert RegimeResult(regime="WEAK_UPTREND").is_bullish
        assert not RegimeResult(regime="RANGING").is_bullish

    def test_is_bearish(self):
        assert RegimeResult(regime="DOWNTREND").is_bearish
        assert not RegimeResult(regime="UPTREND").is_bearish

    def test_should_trade(self):
        assert RegimeResult(scalp_direction="LONG_ONLY", confidence=3).should_trade
        assert not RegimeResult(scalp_direction="NO_TRADE", confidence=3).should_trade
        assert not RegimeResult(scalp_direction="LONG_ONLY", confidence=1).should_trade


class TestChartGenerator:
    def _sample_data(self, n=200):
        rng = np.random.RandomState(42)
        closes = 100 + np.cumsum(rng.randn(n) * 0.5)
        opens = closes + rng.randn(n) * 0.2
        highs = np.maximum(opens, closes) + np.abs(rng.randn(n) * 0.3)
        lows = np.minimum(opens, closes) - np.abs(rng.randn(n) * 0.3)
        volumes = 1000 + rng.randint(0, 500, n)
        timestamps = np.arange(n)
        return timestamps, opens, highs, lows, closes, volumes

    def test_generates_png_bytes(self):
        ts, o, h, l, c, v = self._sample_data()
        png = generate_chart_image(ts, o, h, l, c, v, asset="TEST", timeframe="5m")
        assert isinstance(png, bytes)
        assert len(png) > 1000
        assert png[:8] == b'\x89PNG\r\n\x1a\n'  # PNG magic bytes

    def test_base64_conversion(self):
        ts, o, h, l, c, v = self._sample_data(50)
        png = generate_chart_image(ts, o, h, l, c, v)
        b64 = chart_to_base64(png)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_no_volume(self):
        ts, o, h, l, c, _ = self._sample_data(50)
        png = generate_chart_image(ts, o, h, l, c, asset="BTC")
        assert isinstance(png, bytes)

    def test_minimum_bars(self):
        with pytest.raises(ValueError, match="at least 10"):
            generate_chart_image(
                np.arange(5), np.ones(5), np.ones(5), np.ones(5), np.ones(5)
            )


class TestPrompts:
    def test_system_prompt_has_schema(self):
        assert "regime" in SYSTEM_PROMPT
        assert "STRONG_UPTREND" in SYSTEM_PROMPT
        assert "JSON" in SYSTEM_PROMPT

    def test_build_prompt_basic(self):
        p = build_regime_prompt(asset="BTC", timeframe="5m", bars=500)
        assert "BTC" in p
        assert "5m" in p
        assert "500" in p

    def test_build_prompt_with_price_data(self):
        p = build_regime_prompt(
            asset="SOL",
            price_data={"current_price": 95.0, "price_change_pct": 5.2},
        )
        assert "95.00" in p
        assert "5.2" in p


class TestCache:
    def test_put_and_get(self):
        cache = RegimeCache(default_ttl=60)
        r = RegimeResult(regime="UPTREND", asset="BTC", timeframe="5m")
        cache.put(r)
        got = cache.get("BTC", "5m")
        assert got is not None
        assert got.regime == "UPTREND"

    def test_miss(self):
        cache = RegimeCache()
        assert cache.get("XXX", "1m") is None

    def test_invalidate(self):
        cache = RegimeCache(default_ttl=60)
        r = RegimeResult(regime="RANGING", asset="ETH", timeframe="1h")
        cache.put(r)
        cache.invalidate("ETH", "1h")
        assert cache.get("ETH", "1h") is None

    def test_ttl_expiry(self):
        cache = RegimeCache(default_ttl=0)  # instant expiry
        r = RegimeResult(regime="UPTREND", asset="BTC", timeframe="5m")
        cache.put(r)
        # Should be expired immediately
        import time
        time.sleep(0.01)
        assert cache.get("BTC", "5m") is None


class TestProviderFactory:
    def test_create_anthropic(self):
        p = create_provider("anthropic", api_key="test-key")
        assert p.name == "anthropic"

    def test_create_openai(self):
        p = create_provider("openai", api_key="test-key")
        assert p.name == "openai"

    def test_create_gemini(self):
        p = create_provider("gemini", api_key="test-key")
        assert p.name == "gemini"

    def test_create_ollama(self):
        p = create_provider("ollama")
        assert p.name == "ollama"

    def test_create_aliases(self):
        assert create_provider("claude", api_key="x").name == "anthropic"
        assert create_provider("gpt", api_key="x").name == "openai"
        assert create_provider("local").name == "ollama"

    def test_invalid_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("invalid_provider")
