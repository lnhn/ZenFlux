import pandas as pd 
import numpy as np
from abc import ABC, abstractmethod 
from dataclasses import dataclass

class  MarketData:
    def __init__(self,price):
        self.price = price 
    def build(self):
        df = pd.DataFrame({'price': self.price})
        df['ret'] = df['price'].pct_change()
        return df 
    
class Strategy(ABC):
    @abstractmethod
    def generate_signal(self,data):
        pass

class MovingAverageStrategy(Strategy):
    def __init__(self,short = 5, long = 20):
        self.short = short
        self.long = long

    def generate_signal(self,data):
        ma_s = data['price'].rolling(window=self.short).mean()   
        ma_l = data['price'].rolling(window = self.long).mean() 
        return (ma_s > ma_l).astype(int) 
    
class Backtester:
    def __init__(self,data,signal,execution):
        self.data = data 
        self.signal = signal
        self.execution = execution

    def run(self):
        self.data['position'] = self.signal.shift(self.execution.lag).fillna(0)
        self.data['strategy_return'] = self.data['position'] * self.data['ret']
        self.data['cum'] = (1+ self.data['strategy_return']).cumprod()
        return self.data 
    
class Evaluator:
    def __init__(self,result):
        self.result = result 

    def total_return(self):
        return self.result['cum'].iloc[-1] - 1
    
    def max_drawdown(self):
        peak = self.result['cum'].cummax()
        drawdown = (self.result['cum'] - peak) / peak
        return drawdown.min()
    
@dataclass 
class ExecutionModel:
    lag : int = 1
    cost : float = 0.0
    

if __name__ == "__main__":

    prices = 100 + np.cumsum(np.random.randn(200))
    market_data = MarketData(prices)
    data = market_data.build()

    strategy = MovingAverageStrategy(short=5, long=20)
    signal = strategy.generate_signal(data)

    execution = ExecutionModel(lag=1, cost=0.0)

    backtester = Backtester(data, signal, execution)
    result = backtester.run()

    evaluator = Evaluator(result)
    total_ret = evaluator.total_return()
    max_dd = evaluator.max_drawdown()

    print(f"Total Return: {total_ret:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")