import pandas as pd
import numpy as np
import os

# 1. 设置参数
# 如果是5分钟数据，请填入每天实际的K线数量（例如国内期货含夜盘大约是 69 根）
# 如果是日线数据，请将其设为 1
BARS_PER_DAY = 69  
TRADING_DAYS_PER_YEAR = 252

base_dir = os.path.dirname(os.path.abspath(__file__))
i_path = os.path.join(base_dir, 'I_daily.csv')
jm_path = os.path.join(base_dir, 'J_daily.csv')

print(f"尝试读取文件: {i_path}")
print(f"尝试读取文件: {jm_path}")

i_file = pd.read_csv(i_path)
jm_file = pd.read_csv(jm_path)

# 2. 计算基础指标 (注意处理 NaN)
i_file['return'] = i_file['close'].pct_change()
jm_file['return'] = jm_file['close'].pct_change()

i_returns = i_file['return'].dropna()
jm_returns = jm_file['return'].dropna()

# 3. 计算标准差（单根K线波动率）
i_volatility = i_returns.std()
jm_volatility = jm_returns.std()

# 4. 计算年化波动率 (加入日内数据调节因子)
annualization_factor = np.sqrt(TRADING_DAYS_PER_YEAR * BARS_PER_DAY)
i_annual_volatility = i_volatility * annualization_factor
jm_annual_volatility = jm_volatility * annualization_factor

print("\n=== 核心波动性比较结果 ===")
print(f"I_daily.csv  单根K线收益率标准差：{i_volatility:.6f} | 年化波动率：{i_annual_volatility:.4f}")
print(f"J_daily.csv 单根K线收益率标准差：{jm_volatility:.6f} | 年化波动率：{jm_annual_volatility:.4f}")

if i_volatility > jm_volatility:
    print("结论: I_5.csv 的整体波动性大于 J_5.csv")
elif i_volatility < jm_volatility:
    print("结论: J_5.csv 的整体波动性大于 I_5.csv")
else:
    print("结论: 两个文件的波动性相同")

# 5. 计算其他辅助波动性指标
print("\n=== 其他辅助波动性指标 ===")

# 平均绝对收益率
i_abs_return = abs(i_returns).mean()
jm_abs_return = abs(jm_returns).mean()
print(f"I_5.csv  平均绝对收益率：{i_abs_return:.6f}")
print(f"J_5.csv 平均绝对收益率：{jm_abs_return:.6f}")

# 标准化平均高低价差 (转化为百分比)
# 避免除以0的情况，加上一个极小值 1e-9，或者确保数据中没有 open 为 0 的情况
i_range_pct = ((i_file['high'] - i_file['low']) / (i_file['open'] + 1e-9)).dropna().mean()
jm_range_pct = ((jm_file['high'] - jm_file['low']) / (jm_file['open'] + 1e-9)).dropna().mean()

print(f"I_5.csv  标准化平均高低价差 (百分比)：{i_range_pct * 100:.4f}%")
print(f"J_5.csv 标准化平均高低价差 (百分比)：{jm_range_pct * 100:.4f}%")

"""
=== 核心波动性比较结果 ===
I_5.csv  单根K线收益率标准差：0.003055 | 年化波动率：0.4028
J_5.csv 单根K线收益率标准差：0.002748 | 年化波动率：0.3624


I_daily.csv  单根K线收益率标准差：0.025379 | 年化波动率：3.3466
J_daily.csv 单根K线收益率标准差：0.022404 | 年化波动率：2.9543


=== 其他辅助波动性指标 ===
I_5.csv  平均绝对收益率：0.001908
J_5.csv 平均绝对收益率：0.001704
I_5.csv  标准化平均高低价差 (百分比)：0.3799%
J_5.csv 标准化平均高低价差 (百分比)：0.3381%
"""
