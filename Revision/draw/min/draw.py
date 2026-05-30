import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# Fix: Set matplotlib backend (resolve tostring_rgb error in PyCharm)
# ==========================================
plt.switch_backend('Agg')

# ==========================================
# 1. Data Preparation
# ==========================================
models = ['Main Path', 'Main Path+Auxiliary Stream 1', 'Main Path+Auxiliary Stream 2', 'DENet']

data_metrics = {
    'MAE': [10.165336, 9.777685, 10.131724, 9.550982],
    'RMSE': [14.787234, 14.227218, 14.751037, 14.012361],
    'Step1_RMSE': [8.542504, 7.480993, 8.492024, 7.037260],
    'Step6_RMSE': [14.303150, 13.727676, 14.285870, 13.510129],
    'Step12_RMSE': [19.171486, 18.792068, 19.156235, 18.614632]
}

"""
data_metrics = {
    'MAE': [3.778342, 3.519448, 3.792314, 3.260461],
    'RMSE': [5.508369, 5.165097, 5.514597, 4.886584],
    'Step1_RMSE': [3.383799, 2.853163, 3.478150, 1.999686],
    'Step6_RMSE': [5.321812, 4.991502, 5.358332, 4.698548],
    'Step12_RMSE': [7.066971, 6.769794, 7.034065, 6.590704]
}
"""

df = pd.DataFrame(data_metrics, index=models)

# ==========================================
# 2. Calculate improvement rate (%) relative to Main Path
# ==========================================
baseline_vals = df.loc['Main Path']
df_improvement = (baseline_vals - df) / baseline_vals * 100
df_plot = df_improvement.drop('Main Path')

df_plot = df_plot.T.reset_index()
df_plot = pd.melt(df_plot, id_vars='index', var_name='Model', value_name='Improvement')

# ==========================================
# 3. SCI-style Plot Configuration
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 🔹 增加画布纵向高度，给顶部预留更多空间
fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

colors = ['#A9A9A9', '#CD5C5C', '#1E90FF']
palette = sns.color_palette(colors)

sns.barplot(
    data=df_plot,
    x='index',
    y='Improvement',
    hue='Model',
    palette=palette,
    edgecolor='black',
    linewidth=0.8,
    ax=ax
)

# ==========================================
# 4. Plot Beautification & 重叠修复
# ==========================================
ax.axhline(0, color='black', linewidth=1, linestyle='-')
ax.grid(axis='y', linestyle='--', alpha=0.5)

ax.set_ylabel('Performance Improvement vs. Main Path (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Metrics & Prediction Steps', fontsize=14, fontweight='bold')

ax.set_xticks(range(5))
ax.set_xticklabels(['MAE', 'RMSE', 'Step 1\nRMSE', 'Step 6\nRMSE', 'Step 12\nRMSE'], fontsize=12)

# 🔹 标题优化：大幅增加上下边距，避免和图例重叠
ax.set_title(
    'Minute-level ablation on Coke Futures',
    fontsize=16,
    fontweight='bold',
    pad=35  # 标题和图表主体的距离，彻底拉开
)

# 🔹 数值标注：增加padding，防止文字顶到图表最上沿
for container in ax.containers:
    if container.get_label() == 'DENet':
        ax.bar_label(container, fmt='%.2f%%', padding=6, fontsize=10, fontweight='bold', color='#1E90FF')

# 🔹 图例下移，和标题完全分离，不再重叠
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.07),  # 图例整体向下移动
    ncol=3,
    frameon=False,
    fontsize=11
)

# 🔹 强制增加图表顶部留白
plt.subplots_adjust(top=0.82)
plt.tight_layout()

# ==========================================
# 5. Save high-resolution images
# ==========================================
plt.savefig(
    'model_performance_improvement_denet_J.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

plt.savefig(
    'model_performance_improvement_denet_J.pdf',
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

print("Chart successfully saved, overlap issue completely fixed!")