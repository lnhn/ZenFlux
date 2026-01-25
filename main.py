"""
Day01 Quant Coding Challenge
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    np.random.seed(42)

    n = 200
    price = 100 + np.cumsum(np.random.randn(n))
    data = pd.DataFrame({'price': price})
    return data

def generate_signal(data):
    data = data.copy()

    # Short/long moving averages for a basic trend signal
    data["ma_short"] = data["price"].rolling(window=5).mean()
    data["ma_long"] = data["price"].rolling(window=20).mean()

    data["signal"] = 0
    # Go long when short MA is above long MA
    data.loc[data["ma_short"] > data["ma_long"], "signal"] = 1

    return data["signal"]

def backtest(data, signal):
    data = data.copy()
    data["signal"] = signal 

    data["return"] = data["price"].pct_change()  
    # Use prior signal to avoid look-ahead bias
    data['strategy_return'] = data["signal"].shift(1) * data["return"]

    data["cumulative_return"] = (1 + data["strategy_return"]).cumprod() 

    return data 


def evaluate(result):
     total_return = result["cumulative_return"].iloc[-1] - 1
     print(f"Total Strategy Return: {total_return:.2%}")
     result["cumulative_return"].plot(title="Cumulative Strategy Return")
     plt.show()   
    

if __name__ == "__main__":
    data = load_data()
    signal = generate_signal(data)
    result = backtest(data, signal)
    evaluate(result)

    
