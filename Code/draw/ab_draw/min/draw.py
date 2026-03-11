import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# Fix: Set matplotlib backend (resolve tostring_rgb error in PyCharm)
# ==========================================
plt.switch_backend('Agg')  # Use Agg backend for batch image saving, stable without errors

# ==========================================
# 1. Data Preparation (Revise model names to unique Stream 1/Stream 2)
# ==========================================
models = ['Main Path', 'Main Path+Auxiliary Stream 1', 'Main Path+Auxiliary Stream 2', 'DENet']

# Original data (unchanged)
data_metrics = {
    'MAE': [3.260413, 3.217573, 3.288225, 3.206708],
    'RMSE': [4.869308, 4.835596, 4.894866, 4.824818],
    'Step1_RMSE': [2.182600, 2.088613, 2.234749, 2.049422],
    'Step6_RMSE': [4.698711, 4.660552, 4.706723, 4.653096],
    'Step12_RMSE': [6.480537, 6.456815, 6.498059, 6.441393]
}

# Convert to DataFrame
df = pd.DataFrame(data_metrics, index=models)

# ==========================================
# 2. Calculate improvement rate (%) relative to Main Path
# ==========================================
baseline_vals = df.loc['Main Path']
df_improvement = (baseline_vals - df) / baseline_vals * 100

# Remove Main Path row (baseline with 0% improvement)
df_plot = df_improvement.drop('Main Path')

# Transpose data to fit seaborn plotting format
df_plot = df_plot.T.reset_index()
df_plot = pd.melt(df_plot, id_vars='index', var_name='Model', value_name='Improvement')

# ==========================================
# 3. SCI-style Plot Configuration (font settings fixed)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # Correct font parameter
plt.rcParams['axes.unicode_minus'] = False  # Resolve negative sign display issue

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Restore 3 colors (matching 3 models to plot)
colors = ['#A9A9A9', '#CD5C5C', '#1E90FF']  # Gray(Stream1), LightRed(Stream2), DarkBlue(DENet)
palette = sns.color_palette(colors)

# Plot bar chart
sns.barplot(
    data=df_plot,
    x='index',
    y='Improvement',
    hue='Model',
    palette=palette,
    edgecolor='black',  # Add black border to bars for clearer SCI figures
    linewidth=0.8,
    ax=ax
)

# ==========================================
# 4. Detailed Beautification (tick label warning fixed)
# ==========================================
# Add 0 baseline (distinguish performance improvement/decline)
ax.axhline(0, color='black', linewidth=1, linestyle='-')

# Horizontal grid lines (enhance readability)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Axis labels
ax.set_ylabel('Performance Improvement vs. Main Path (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Metrics & Prediction Steps', fontsize=14, fontweight='bold')

# Fix tick labels: set tick positions first, then labels
ax.set_xticks(range(5))  # 5 metrics (MAE/RMSE/Step1/6/12)
ax.set_xticklabels(['MAE', 'RMSE', 'Step 1\nRMSE', 'Step 6\nRMSE', 'Step 12\nRMSE'], fontsize=12)

# Annotate exact improvement rate values on DENet bars
for container in ax.containers:
    if container.get_label() == 'DENet':
        ax.bar_label(container, fmt='%.2f%%', padding=3, fontsize=10, fontweight='bold', color='#1E90FF')

# Legend settings (3 columns matching 3 models)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=11)

# Adjust layout to prevent label/legend truncation
plt.tight_layout()

# ==========================================
# 5. Save high-resolution images
# ==========================================
# Save PNG (for preview/submission, 300DPI meets SCI requirements)
plt.savefig(
    'model_performance_improvement_denet.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

# Save PDF (vector image, no jagged edges for journal layout)
plt.savefig(
    'model_performance_improvement_denet.pdf',
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

print("Chart successfully saved in PNG and PDF formats!")

# [Optional] For interactive image display (popup window), perform the following steps:
# 1. Install tkinter: conda install tk -y
# 2. Comment out plt.switch_backend('Agg') at the beginning
# 3. Uncomment the following two lines:
# plt.switch_backend('TkAgg')
# plt.show()