"""Chart generator for LLM regime analysis.

Generates clean, standardized chart images optimized for LLM vision models.
The charts are intentionally simple — no clutter, clear candles, SMA, volume.
LLMs perform better with clean visuals than with indicator-heavy charts.
"""

from __future__ import annotations

import io
import base64
from typing import Optional, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def generate_chart_image(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    asset: str = "",
    timeframe: str = "",
    sma_periods: Tuple[int, ...] = (20, 50),
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 100,
    analysis_window: Optional[int] = None,
) -> bytes:
    """Generate a clean chart image as PNG bytes.

    Produces a 2-panel chart (price + volume) optimized for LLM vision:
      - Clean candlesticks with SMA overlays
      - Volume bars
      - No indicators, no clutter — just price structure

    Parameters
    ----------
    timestamps : array of epoch timestamps (seconds or ms)
    opens, highs, lows, closes : price arrays
    volumes : optional volume array
    asset : asset name for title
    timeframe : timeframe string for title
    sma_periods : which SMAs to draw
    figsize : figure size in inches
    dpi : resolution
    analysis_window : if set, only the last N bars are drawn in full color;
                      earlier bars are grayed out as "context". This helps
                      the LLM focus on the relevant period while still
                      seeing the broader trend.

    Returns
    -------
    bytes : PNG image data
    """
    n = len(closes)
    if n < 10:
        raise ValueError(f"Need at least 10 bars, got {n}")

    # Determine which bars are in the analysis window vs context
    if analysis_window and analysis_window < n:
        context_end = n - analysis_window  # bars before this are context
    else:
        context_end = 0  # all bars are in the analysis window

    # Create x-axis indices
    x = np.arange(n)

    # Figure setup
    has_volume = volumes is not None and len(volumes) == n
    height_ratios = [4, 1] if has_volume else [1]
    n_panels = 2 if has_volume else 1

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3, color="#d0d0d0", lw=0.5)
        ax.tick_params(colors="#333333", labelsize=9)

    ax_price = axes[0]

    # --- Candlesticks ---
    bar_width = 0.6
    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        is_context = i < context_end
        if is_context:
            color = "#c0c0c0"  # gray for context bars
            alpha = 0.5
        else:
            color = "#26a69a" if c >= o else "#ef5350"
            alpha = 1.0
        # Wick
        ax_price.plot([x[i], x[i]], [l, h], color=color, lw=0.8, alpha=alpha)
        # Body
        body_h = abs(c - o)
        body_bottom = min(o, c)
        if body_h > 0:
            ax_price.bar(x[i], body_h, bottom=body_bottom, width=bar_width,
                         color=color, edgecolor=color, linewidth=0.5, alpha=alpha)

    # --- Analysis window divider line ---
    if context_end > 0:
        ax_price.axvline(x=context_end - 0.5, color="#1565c0", lw=1.5,
                         ls="--", alpha=0.7, zorder=5)
        # Label
        ylim = ax_price.get_ylim()
        ax_price.text(context_end + 1, ylim[1] - (ylim[1] - ylim[0]) * 0.03,
                      f"← analyze last {analysis_window} bars",
                      fontsize=8, color="#1565c0", fontweight="bold", va="top")

    # --- SMAs ---
    sma_colors = ["#ff6f00", "#1565c0", "#7b1fa2"]
    for idx, period in enumerate(sma_periods):
        if n >= period:
            sma = np.convolve(closes, np.ones(period) / period, mode="valid")
            sma_x = x[period - 1:]
            color = sma_colors[idx % len(sma_colors)]
            ax_price.plot(sma_x, sma, color=color, lw=1.5, alpha=0.8,
                          label=f"SMA({period})")

    ax_price.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax_price.set_ylabel("Price", fontsize=10)

    # --- Price annotations ---
    price_start = closes[0]
    price_end = closes[-1]
    price_change = (price_end / price_start - 1) * 100
    window_label = f" | analyze last {analysis_window}" if analysis_window and analysis_window < n else ""
    ax_price.set_title(
        f"{asset} {timeframe} | {n} bars | "
        f"{price_start:.2f} → {price_end:.2f} ({price_change:+.1f}%){window_label}",
        fontsize=12, fontweight="bold", pad=10,
    )

    # --- Volume ---
    if has_volume:
        ax_vol = axes[1]
        vol_colors = []
        vol_alphas = []
        for i in range(n):
            is_context = i < context_end
            if is_context:
                vol_colors.append("#c0c0c0")
                vol_alphas.append(0.3)
            else:
                vol_colors.append("#26a69a" if closes[i] >= opens[i] else "#ef5350")
                vol_alphas.append(0.6)
        ax_vol.bar(x, volumes, width=bar_width, color=vol_colors, alpha=0.6)
        ax_vol.set_ylabel("Volume", fontsize=10)
        if context_end > 0:
            ax_vol.axvline(x=context_end - 0.5, color="#1565c0", lw=1.5,
                           ls="--", alpha=0.7, zorder=5)

    # X-axis: show every ~50 bars
    tick_step = max(1, n // 10)
    tick_positions = x[::tick_step]
    axes[-1].set_xticks(tick_positions)

    # bbox_inches="tight" in savefig handles layout — suppress the redundant tight_layout warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    # Render to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_to_base64(png_bytes: bytes) -> str:
    """Convert PNG bytes to base64 string for LLM API calls."""
    return base64.b64encode(png_bytes).decode("utf-8")


def generate_multi_timeframe_chart(
    data_by_tf: dict,
    asset: str = "",
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 100,
) -> bytes:
    """Generate a multi-timeframe chart (e.g., 5m + 1h side by side).

    Parameters
    ----------
    data_by_tf : dict mapping timeframe → dict with keys:
                 timestamps, opens, highs, lows, closes, volumes
    asset : asset name
    """
    tfs = list(data_by_tf.keys())
    n_tf = len(tfs)

    fig, axes = plt.subplots(n_tf, 1, figsize=figsize)
    if n_tf == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")

    for idx, tf in enumerate(tfs):
        d = data_by_tf[tf]
        ax = axes[idx]
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3, color="#d0d0d0", lw=0.5)

        closes = np.array(d["closes"])
        opens = np.array(d["opens"])
        highs = np.array(d["highs"])
        lows = np.array(d["lows"])
        n = len(closes)
        x = np.arange(n)

        for i in range(n):
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            color = "#26a69a" if c >= o else "#ef5350"
            ax.plot([x[i], x[i]], [l, h], color=color, lw=0.8)
            body_h = abs(c - o)
            if body_h > 0:
                ax.bar(x[i], body_h, bottom=min(o, c), width=0.6,
                       color=color, edgecolor=color, linewidth=0.5)

        # SMA
        if n >= 20:
            sma20 = np.convolve(closes, np.ones(20) / 20, mode="valid")
            ax.plot(x[19:], sma20, color="#ff6f00", lw=1.5, alpha=0.8, label="SMA(20)")
        if n >= 50:
            sma50 = np.convolve(closes, np.ones(50) / 50, mode="valid")
            ax.plot(x[49:], sma50, color="#1565c0", lw=1.5, alpha=0.8, label="SMA(50)")

        p_chg = (closes[-1] / closes[0] - 1) * 100
        ax.set_title(f"{asset} {tf} | {n} bars ({p_chg:+.1f}%)", fontsize=11)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
