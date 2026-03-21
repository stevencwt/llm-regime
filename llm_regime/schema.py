"""Schema for LLM regime detection results.

Defines the data structures returned by regime analysis, independent of
which LLM provider is used. Any bot or strategy can consume these.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
import json


class Regime(str, Enum):
    """Market regime classification."""
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    RANGING = "RANGING"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class Volatility(str, Enum):
    """Volatility classification."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TrendStrength(str, Enum):
    """Trend strength classification."""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"


@dataclass
class KeyLevel:
    """A significant price level (support/resistance)."""
    price: float
    type: str  # "support" or "resistance"
    strength: str = "moderate"  # "strong", "moderate", "weak"


@dataclass
class RegimeResult:
    """Complete regime analysis result from LLM.

    This is the primary output that any consuming bot receives.
    It contains everything needed to make trading decisions.
    """
    # Core classification
    regime: str = Regime.UNKNOWN.value
    confidence: int = 0  # 1-5 scale
    volatility: str = Volatility.MODERATE.value
    trend_strength: str = TrendStrength.NONE.value

    # Direction bias for trading
    bias: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    scalp_direction: str = "BOTH"  # "LONG_ONLY", "SHORT_ONLY", "BOTH", "NO_TRADE"

    # Key levels
    key_levels: list = field(default_factory=list)

    # LLM reasoning (for debugging/logging)
    reasoning: str = ""
    pattern: str = ""  # e.g. "ascending triangle", "head and shoulders"

    # Market structure (new — swing sequence analysis)
    # {"hh_hl": bool, "lh_ll": bool, "broken": bool, "summary": str}
    structure: dict = field(default_factory=dict)

    # Nearest high-probability reaction zone (new — pullback entry support)
    # {"price": float, "type": str, "quality": str, "distance_pct": float,
    #  "rationale": str, "tradeable": bool}
    nearest_zone: dict = field(default_factory=dict)

    # Metadata
    asset: str = ""
    timeframe: str = ""
    bars_analyzed: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for serialization."""
        d = asdict(self)
        return d

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegimeResult":
        """Create from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def is_bullish(self) -> bool:
        return self.regime in (
            Regime.STRONG_UPTREND.value,
            Regime.UPTREND.value,
            Regime.WEAK_UPTREND.value,
        )

    @property
    def is_bearish(self) -> bool:
        return self.regime in (
            Regime.STRONG_DOWNTREND.value,
            Regime.DOWNTREND.value,
            Regime.WEAK_DOWNTREND.value,
        )

    @property
    def is_ranging(self) -> bool:
        return self.regime == Regime.RANGING.value

    @property
    def should_trade(self) -> bool:
        return self.scalp_direction != "NO_TRADE" and self.confidence >= 2
