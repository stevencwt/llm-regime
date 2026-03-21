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
        assert r.structure == {}
        assert r.nearest_zone == {}

    def test_to_dict(self):
        r = RegimeResult(regime="UPTREND", confidence=4, bias="BULLISH")
        d = r.to_dict()
        assert d["regime"] == "UPTREND"
        assert d["confidence"] == 4
        assert "structure" in d
        assert "nearest_zone" in d

    def test_to_json(self):
        r = RegimeResult(regime="RANGING", confidence=3)
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["regime"] == "RANGING"
        assert "structure" in parsed
        assert "nearest_zone" in parsed

    def test_from_dict(self):
        d = {"regime": "DOWNTREND", "confidence": 2, "bias": "BEARISH"}
        r = RegimeResult.from_dict(d)
        assert r.regime == "DOWNTREND"
        assert r.is_bearish

    def test_from_dict_with_new_fields(self):
        d = {
            "regime": "UPTREND",
            "confidence": 4,
            "structure": {"hh_hl": True, "lh_ll": False, "broken": False, "summary": "Three HH/HL intact"},
            "nearest_zone": {"price": 83200.0, "type": "support", "quality": "high",
                             "distance_pct": 1.43, "rationale": "Prior breakout level", "tradeable": True},
        }
        r = RegimeResult.from_dict(d)
        assert r.structure["hh_hl"] is True
        assert r.nearest_zone["tradeable"] is True
        assert r.nearest_zone["price"] == 83200.0

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

    def test_system_prompt_has_new_fields(self):
        assert "structure" in SYSTEM_PROMPT
        assert "nearest_zone" in SYSTEM_PROMPT
        assert "hh_hl" in SYSTEM_PROMPT
        assert "tradeable" in SYSTEM_PROMPT
        assert "distance_pct" in SYSTEM_PROMPT

    def test_system_prompt_has_key_level_reason(self):
        assert '"reason"' in SYSTEM_PROMPT

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


class TestNewFields:
    """Test parsing of structure and nearest_zone from LLM JSON responses."""

    def _make_analyzer(self):
        from llm_regime.analyzer import RegimeAnalyzer
        from unittest.mock import MagicMock
        analyzer = RegimeAnalyzer.__new__(RegimeAnalyzer)
        analyzer._provider = MagicMock()
        analyzer._cache = MagicMock()
        analyzer._temperature = 0.1
        analyzer._sma_periods = (20, 50)
        analyzer._chart_dpi = 100
        return analyzer

    def _full_json(self, **overrides):
        base = {
            "regime": "UPTREND",
            "confidence": 4,
            "volatility": "MODERATE",
            "trend_strength": "MODERATE",
            "bias": "BULLISH",
            "scalp_direction": "LONG_ONLY",
            "key_levels": [
                {"price": 83200, "type": "support", "strength": "strong", "reason": "prior breakout level"},
                {"price": 85500, "type": "resistance", "strength": "moderate", "reason": "recent swing high"},
            ],
            "reasoning": "Three HH/HL swings visible above the 20 SMA.",
            "pattern": "bull flag",
            "structure": {
                "hh_hl": True,
                "lh_ll": False,
                "broken": False,
                "summary": "Three HH/HL swings intact",
            },
            "nearest_zone": {
                "price": 83200.0,
                "type": "support",
                "quality": "high",
                "distance_pct": 1.43,
                "rationale": "Prior breakout level tested twice",
                "tradeable": True,
            },
        }
        base.update(overrides)
        return json.dumps(base)

    def test_parse_structure(self):
        analyzer = self._make_analyzer()
        result = analyzer._parse_response(self._full_json(), "BTC", "5m", 200)
        assert result.structure["hh_hl"] is True
        assert result.structure["lh_ll"] is False
        assert result.structure["broken"] is False
        assert "intact" in result.structure["summary"]

    def test_parse_nearest_zone(self):
        analyzer = self._make_analyzer()
        result = analyzer._parse_response(self._full_json(), "BTC", "5m", 200)
        assert result.nearest_zone["price"] == 83200.0
        assert result.nearest_zone["type"] == "support"
        assert result.nearest_zone["quality"] == "high"
        assert result.nearest_zone["distance_pct"] == 1.43
        assert result.nearest_zone["tradeable"] is True

    def test_parse_key_level_reason(self):
        analyzer = self._make_analyzer()
        result = analyzer._parse_response(self._full_json(), "BTC", "5m", 200)
        assert result.key_levels[0]["reason"] == "prior breakout level"
        assert result.key_levels[1]["reason"] == "recent swing high"

    def test_key_level_missing_reason_defaults_to_empty(self):
        """Old-format key_levels without 'reason' should still parse cleanly."""
        data = json.loads(self._full_json())
        for kl in data["key_levels"]:
            del kl["reason"]
        analyzer = self._make_analyzer()
        result = analyzer._parse_response(json.dumps(data), "BTC", "5m", 200)
        assert result.key_levels[0]["reason"] == ""

    def test_missing_structure_defaults_to_safe_values(self):
        """When LLM omits 'structure', parser should produce a safe all-false default."""
        data = json.loads(self._full_json())
        del data["structure"]
        analyzer = self._make_analyzer()
        result = analyzer._parse_response(json.dumps(data), "BTC", "5m", 200)
        assert result.structure["hh_hl"] is False
        assert result.structure["lh_ll"] is False
        assert result.structure["broken"] is False
        assert result.structure["summary"] == ""

    def test_missing_nearest_zone_defaults_to_safe_values(self):
        """When LLM omits 'nearest_zone', parser should produce a safe low-quality default."""
        data = json.loads(self._full_json())
        del data["nearest_zone"]
        analyzer = self._make_analyzer()
        result = analyzer._parse_response(json.dumps(data), "BTC", "5m", 200)
        assert result.nearest_zone["price"] == 0.0
        assert result.nearest_zone["quality"] == "low"
        assert result.nearest_zone["tradeable"] is False
        assert result.nearest_zone["rationale"] == ""

    def test_distance_pct_rounded_to_2dp(self):
        data = json.loads(self._full_json())
        data["nearest_zone"]["distance_pct"] = 1.4285714
        analyzer = self._make_analyzer()
        result = analyzer._parse_response(json.dumps(data), "BTC", "5m", 200)
        assert result.nearest_zone["distance_pct"] == 1.43

    def test_backward_compat_existing_fields_unchanged(self):
        """Ensure original fields are still present and correct after parsing."""
        analyzer = self._make_analyzer()
        result = analyzer._parse_response(self._full_json(), "SOL", "1h", 150)
        assert result.regime == "UPTREND"
        assert result.confidence == 4
        assert result.volatility == "MODERATE"
        assert result.trend_strength == "MODERATE"
        assert result.bias == "BULLISH"
        assert result.scalp_direction == "LONG_ONLY"
        assert result.reasoning != ""
        assert result.pattern == "bull flag"
        assert len(result.key_levels) == 2


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
