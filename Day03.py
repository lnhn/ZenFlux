"""
Day03 -Quant Training
Theme:
- Market as Object 
- Time Series as SDynamic Process 
"""

#=== 
#1. Imports
#===

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence
import logging
import warnings

import akshare as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
#===
#2. Configuration
#====

@dataclass(frozen=True)
class MarketConfig:
    symbol: str = "600519"
    start_date: str = "20220101"
    end_date: str = "20251231"
    windows: List[int] = field(default_factory=lambda: [10, 50])
    fast: int = 10
    slow: int = 50


def _normalize_windows(windows: Sequence[int], fast: int, slow: int) -> List[int]:
    """Return sorted, unique, positive MA windows that include fast and slow."""
    merged = list(windows) + [fast, slow]
    normalized = sorted({w for w in merged if isinstance(w, int) and w > 0})
    if not normalized:
        raise ValueError("At least one positive integer window is required.")
    return normalized


#===
#3. Data Loading 
#====

def load_data(symbol: str = "600519", start_date: str = "20220101", end_date: str = "20251231") -> pd.DataFrame:
    """Load daily close prices from AkShare; fall back to synthetic data offline."""
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
    except Exception as exc:
        # In restricted or offline environments, AkShare may fail to reach its data source.
        logger.warning("AkShare download failed; using synthetic data instead: %s", exc)
        warnings.warn(
            f"AkShare data download failed ({exc}). Falling back to synthetic data.",
            RuntimeWarning,
        )
        dates = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq="B")
        # Simple random walk around an arbitrary starting price for demo purposes.
        close = 100 * (1 + np.random.normal(0, 0.01, len(dates))).cumprod()
        df = pd.DataFrame({"日期": dates, "收盘": close})

    df = df[['日期', '收盘']]
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# =====================
# 4. MarketSeries Class
# =====================


class MarketSeries:
    def __init__(self, df: pd.DataFrame, windows: Iterable[int] = (10,)) -> None:
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column.")
        self.df = df.copy()
        self.windows = list(windows)

    def returns(self) -> MarketSeries:
        self.df['ret'] = self.df['close'].pct_change()
        return self

    def rolling_mean(self) -> MarketSeries:
        for window in self.windows:
            ma_col = f'ma_{window}'
            if ma_col not in self.df.columns:
                self.df[ma_col] = self.df['close'].rolling(window).mean()
        return self

    def dropna(self) -> MarketSeries:
        self.df.dropna(inplace=True)
        return self

    def cross_signal(self, fast: int = 10, slow: int = 50) -> MarketSeries:
        fast_col = f'ma_{fast}'
        slow_col = f'ma_{slow}'
        # Ensure the moving averages used for the signal exist.
        if fast_col not in self.df.columns:
            self.df[fast_col] = self.df['close'].rolling(fast).mean()
        if slow_col not in self.df.columns:
            self.df[slow_col] = self.df['close'].rolling(slow).mean()

        diff = self.df[fast_col] - self.df[slow_col]

        self.df['golden_cross'] = (diff > 0) & (diff.shift(1) <= 0)
        self.df['death_cross'] = (diff < 0) & (diff.shift(1) >= 0)

        self.df['golden_cross'] = self.df['golden_cross'].fillna(False)
        self.df['death_cross'] = self.df['death_cross'].fillna(False)

        return self

    def plot(self) -> MarketSeries:
        # Ensure signals exist so plotting does not crash when cross_signal was not called.
        if 'golden_cross' not in self.df.columns or 'death_cross' not in self.df.columns:
            fast = self.windows[0] if self.windows else 10
            slow = self.windows[1] if len(self.windows) > 1 else max(50, fast * 2)
            self.cross_signal(fast=fast, slow=slow)
        fig, ax = plt.subplots(figsize=(12, 6))
        self.df['close'].plot(ax=ax, label='close', alpha=0.78)

        for window in self.windows:
            ma_col = f'ma_{window}'
            if ma_col in self.df.columns:
                self.df[ma_col].plot(ax=ax, label=ma_col)

        # Plot cross markers on the fast MA (closer to the actual intersection than close price).
        fast = self.windows[0] if self.windows else 10
        fast_col = f'ma_{fast}'

        gc = self.df.loc[self.df['golden_cross']]
        ax.scatter(gc.index, gc[fast_col], marker='^', color='g', s=100, label='Golden Cross')

        dc = self.df.loc[self.df['death_cross']]
        ax.scatter(dc.index, dc[fast_col], marker='v', color='red', s=80, label='Death Cross')
        ax.legend()
        ax.set_title("Price & Rolling Mean")
        plt.show()
        return self



#=====================
#5. Entry Point
#=====================

def main(config: Optional[MarketConfig] = None) -> MarketSeries:
    cfg = config or MarketConfig()
    windows = _normalize_windows(cfg.windows, fast=cfg.fast, slow=cfg.slow)
    data = load_data(symbol=cfg.symbol, start_date=cfg.start_date, end_date=cfg.end_date)
    series = MarketSeries(data, windows=windows)
    series.returns().rolling_mean().cross_signal(fast=cfg.fast, slow=cfg.slow).dropna()
    series.plot()
    return series


if __name__ == "__main__":
    main()
