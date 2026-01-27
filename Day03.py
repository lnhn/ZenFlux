"""
Day03 -Quant Training
Theme:
- Market as Object 
- Time Series as SDynamic Process 
"""

#=== 
#1. Imports
#===

import akshare as ak 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
#===
#2. Data Loading 
#====

def load_data(symbol="600519",start_date="20220101",end_date="20251231"):
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
        warnings.warn(
            f"AkShare data download failed ({exc}). Falling back to synthetic data.",
            RuntimeWarning,
        )
        dates = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq="B")
        # Simple random walk around an arbitrary starting price for demo purposes.
        close = 100 * (1 + np.random.normal(0, 0.01, len(dates))).cumprod()
        df = pd.DataFrame({"日期": dates, "收盘": close})

    df = df[['日期', '收盘']]
    df.columns = ['date','close']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date',inplace=True)
    return df 

# =====================
# 3. MarketSeries Class
# =====================


class MarketSeries:
    def __init__(self,df,windows=[10]):
        self.df = df.copy()
        self.windows = windows
    def returns(self):
        self.df['ret']=self.df['close'].pct_change()
        return self
    def rolling_mean(self):
        for window in self.windows:
            self.df[f'ma_{window}'] = self.df['close'].rolling(window).mean()
        return self 
    def dropna(self):
        self.df.dropna(inplace=True)
        return self
    def cross_signal(self,fast=10,slow=50):
        fast_col = f'ma_{fast}'
        low_col = f'ma_{slow}'
        # Always (re)compute the moving averages used for the signal.
        self.df[fast_col] = self.df['close'].rolling(fast).mean()
        self.df[low_col] = self.df['close'].rolling(slow).mean()

        diff = self.df[fast_col]-self.df[low_col]

        self.df['golden_cross'] = (diff >0) & (diff.shift(1)<=0)
        self.df['death_cross'] = (diff <0) & (diff.shift(1)>=0)

        self.df['golden_cross'] = self.df['golden_cross'].fillna(False)
        self.df['death_cross'] = self.df['death_cross'].fillna(False)

        return self 
   
    
    def plot(self):
        # Ensure signals exist so plotting does not crash when cross_signal was not called.
        if 'golden_cross' not in self.df.columns or 'death_cross' not in self.df.columns:
            fast = self.windows[0] if self.windows else 10
            slow = self.windows[1] if len(self.windows) > 1 else max(50, fast * 2)
            self.cross_signal(fast=fast, low=slow)
        fig,ax = plt.subplots(figsize=(12,6))
        self.df['close'].plot(ax=ax,label='close',alpha=0.78)

        for window in self.windows:
            ma_col = f'ma_{window}'
            if ma_col in self.df.columns:
                self.df[ma_col].plot(ax=ax,label=ma_col)

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
#5. Experimental Code 
#=====================

if __name__ == "__main__":
    data = load_data()
    marketSeries = MarketSeries(data,windows=[10,50])
    marketSeries.returns().rolling_mean().cross_signal(fast=10, slow=50).dropna()
    marketSeries.plot()
