"""
Plotting utilities for seeing regimes on price and return series
"""
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .model import SegmentationResult

def plot_price_with_regimes(
    df: pd.DataFrame,
    segmentation: SegmentationResult,
    ticker: str, 
    path: str | Path, 
    title_suffix: Optional[str] = None
) -> None:
    """
    Plot price and log returns with detected regimes highlighted, then save to a PNG
    
    Params:
        df: DataFrame w/ DatetimeIndex and columns ["price", "log_return"]
        segmentation: SegmentationResult produced by the model
        ticker: Ticker symbol
        path: Output path for the image
        title_suffix: (OPTIONAL) Text for title
    """
    if df.empty:
        raise ValueError("Can't plot an empty DataFrame")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dates = df.index
    prices = df["price"]
    returns = df["log_return"]

    fig, (ax_price, ax_ret) = plt.subplots(
        2,
        1,
        figsize=(11,7),
        sharex=True,
        gridspec_kw={"height_ratios": [3,1]}
    )

    ax_price.plot(dates,prices, linewidth=1.2, label="Price")

    colors = ["#e0f3ff", "#ffe0f0", "#e0ffe7", "#f3e0ff"]

    for idx, regime in enumerate(segmentation.regimes, start=1):
        seg_start = dates[regime.start]
        seg_end = dates[regime.end]
        color = colors[(idx - 1) % len(colors)]

        ax_price.axvspan(seg_start, seg_end, color=color, alpha=0.45)

        mid_idx = (regime.start+regime.end) // 2
        mid_date = dates[mid_idx]
        ax_price.text(
            mid_date,
            prices.max(),
            f"R{idx}",
            ha="center",
            va="top",
            fontsize=8,
            alpha=0.7
        )
    
    ax_price.set_ylabel("Price")
    base_title = f"{ticker} - Price & Regimes"
    if title_suffix:
        ax_price.set_title(f"{base_title} ({title_suffix})")
    else:
        ax_price.set_title(base_title)
    ax_price.legend(loc="upper left")

    ax_ret.plot(dates, returns, linewidth=.8)
    ax_ret.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.6)

    for regime in segmentation.regimes[:-1]:
        boundary_date = dates[regime.end]
        ax_ret.axvline(boundary_date, linestyle="--", linewidth=0.8, alpha=0.5)
    
    ax_ret.set_ylabel("Log return")
    ax_ret.set_xlabel("Date")
    ax_ret.grid(True, linestyle="--", alpha=0.3)

    ax_ret.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_ret.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_ret.xaxis.get_major_locator()))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)