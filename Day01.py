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
    #data.loc[data["price"] >80, "signal"] = 1
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
     #final return of the data 最终的收益
     total_return = result["cumulative_return"].iloc[-1] - 1

     print(f"Total Strategy Return: {total_return:.2%}")
     mdd = max_drawdown(result["cumulative_return"])
     print(f"Max Drawdown: {mdd:.2%}")
     annual_return  = annualized_return(result["cumulative_return"])
     print (f"Annualized Return: {annual_return:.2%}")
     result["cumulative_return"].plot(title="Cumulative Strategy Return")
     
     plt.show()   

def max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_dd = drawdown.min()
    return max_dd

def annualized_return(cumu):
    n_days = len(cumu)
    final_value = cumu.iloc[-1]
    return final_value ** (252 / n_days) - 1

    

if __name__ == "__main__":
    data = load_data()
    signal = generate_signal(data)
    result = backtest(data, signal)
   # plt.plot(data['price'])
    evaluate(result)

    

# --- Day 01 学习总结 ---
# 1) 数据与信号：
#    - 使用随机游走构造价格序列，形成最小可运行的数据集。
#    - 通过短期/长期均线计算交易信号，理解指标如何驱动策略。
#    - 注意信号规则（例如短均线与长均线的关系）会直接影响持仓与收益表现。
# 2) 回测逻辑：
#    - 用价格的百分比变化计算单期收益。
#    - 信号向后移一位（shift）以避免前视偏差，符合“信号生成后下一期执行”的交易假设。
#    - 累乘得到策略净值曲线，形成基础的绩效评估输入。
# 3) 绩效评估：
#    - 计算策略总收益、最大回撤与年化收益，建立最常用的策略评价指标框架。
#    - 通过净值曲线直观看到策略的整体走势与波动特征。
# 4) 进一步思考：
#    - 可以尝试加入手续费、滑点、仓位管理等更贴近真实交易的因素。
#    - 可以比较不同参数（均线窗口）或不同信号逻辑对结果的影响。
"""
========================
Day01 Quant Summary
========================

1. 量化系统的本质：
   - 策略的“思想核心”只存在于 generate_signal() 中
   - 它表达的是：我对市场结构的假设（趋势 / 均值回归 / 波动等）
   - 其余代码（回测、执行、评价）只是约束与裁判，不应随意改动

2. 时间因果与交易现实：
   - signal 使用前一天（已知）的数据生成决策（0 / 1）
   - position 在当期生效，真实承担当期的价格波动
   - 收益永远来自“持仓之后”的价格变化
   - signal.shift(1) 不是技巧，而是因果律本身，避免未来函数

3. close-to-close 逻辑：
   - 所有决策基于上一期收盘价
   - 所有收益基于当期收盘价变化
   - “看到收益”和“产生收益”必须严格区分

4. 收益的正确度量方式：
   - strategy_return：单期真实收益
   - cumulative_return：资金净值曲线（复利世界）
   - total_return = final_nav - 1
   - annualized_return 用于不同时间长度策略的可比性
   - Buy & Hold 是任何策略的第一基线

5. 风险视角：
   - 最大回撤（Max Drawdown）是底线指标
   - 策略是否值得研究，先看是否能在风险上站得住

Day01 结论：
- 回测系统已经“干净”：无未来函数、因果清晰、评价客观
- 当前策略表现不好，并非代码问题，而是策略假设与数据结构不匹配
- 后续研究的重心，应放在 generate_signal() 的设计与比较上
"""