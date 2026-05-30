import pandas as pd
import numpy as np
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
# 1. 读取原始真实值文件（5分钟K线，包含datetime和close列）
df_true = pd.read_csv(os.path.join(base_dir, 'JM_true.csv'))  # 替换为你的真实值文件路径

# 2. 统一datetime格式（确保和预测文件格式一致）
df_true['datetime'] = pd.to_datetime(df_true['datetime'], format='%Y/%m/%d %H:%M', errors='coerce')
# 删除无效datetime行
df_true = df_true.dropna(subset=['datetime']).reset_index(drop=True)

# 3. 构建12步真实值序列（核心修正：对齐pred_step_1）
# true_step_1 = 当前datetime的close（无偏移）
# true_step_2 = 当前datetime+5分钟的close（shift(-1)）
# true_step_3 = 当前datetime+10分钟的close（shift(-2)）
# ...
# true_step_12 = 当前datetime+55分钟的close（shift(-11)）
for step in range(1, 13):
    if step == 1:
        # Step1：当前datetime的真实值（无偏移）
        df_true[f'true_step_{step}'] = df_true['close']
    else:
        # StepN（N≥2）：当前datetime + (N-1)*5分钟的真实值（shift(-(step-1))）
        df_true[f'true_step_{step}'] = df_true['close'].shift(-(step-1))

# 4. 计算12步真实值的平均值（对应预测文件的pred_avg_1hr）
true_step_cols = [f'true_step_{i}' for i in range(1, 13)]
df_true['true_avg_1hr'] = df_true[true_step_cols].mean(axis=1)

# 5. 删除边界行（最后11行无法构建完整12步序列，会产生NaN）
# （因为true_step_12需要shift(-11)，最后11行的true_step_12会是NaN）
df_true_clean = df_true.dropna(subset=true_step_cols).reset_index(drop=True)

# 6. 保留目标格式列（和预测文件完全对应）
target_cols = ['datetime'] + true_step_cols + ['true_avg_1hr']
df_true_final = df_true_clean[target_cols].copy()

# 7. 验证对应关系（打印前3行，确认对齐）
print("✅ 修正后真实值文件结构（前3行）：")
print(df_true_final.head(3))
print(f"\n✅ 有效数据行数：{len(df_true_final)}（无NaN，可直接和预测文件对比）")

# 8. 保存修正后的真实值文件（命名和预测文件对应）
output_path = 'JM_真实.csv'
df_true_final.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✅ 修正后的真实值文件已保存：{output_path}")