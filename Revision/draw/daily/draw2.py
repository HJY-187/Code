import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

# ==========================================
# 0. Backend Configuration
# ==========================================
plt.switch_backend('Agg')

# ==========================================
# 1. Data Preparation
# ==========================================
data = {
    'Model': ['Main Path', 'Main Path+Aux 1', 'Main Path+Aux 2', 'DENet'],
    'MAE': [12.4861, 12.4631, 12.4446, 12.4277],
    'RMSE': [16.7595, 16.6234, 16.7702, 16.6054],
    'R2': [0.9568, 0.9572, 0.9571, 0.9575]
}

df = pd.DataFrame(data)

# Long format
df_melt = df.melt(id_vars='Model', var_name='Metric', value_name='Value')

# ==========================================
# 2. Relative Improvement + Score²
# ==========================================
def calculate_score(row):
    metric_data = df[row['Metric']]

    if row['Metric'] in ['MAE', 'RMSE']:
        worst = metric_data.max()
        best = metric_data.min()
        return (worst - row['Value']) / (worst - best)
    else:
        worst = metric_data.min()
        best = metric_data.max()
        return (row['Value'] - worst) / (best - worst)

df_melt['Score'] = df_melt.apply(calculate_score, axis=1)

# 非线性增强（但保持克制）
df_melt['Plot_Size'] = 0.25 + 0.75 * (df_melt['Score'] ** 2)

# ==========================================
# 3. Plot Configuration
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)

metrics = ['MAE', 'RMSE', 'R2']
models = df['Model'].tolist()

x_mapping = {model: i for i, model in enumerate(models)}
y_mapping = {metric: i for i, metric in enumerate(metrics)}

x = df_melt['Model'].map(x_mapping)
y = df_melt['Metric'].map(y_mapping)
s = df_melt['Plot_Size'] * 2400
c = df_melt['Score']

scatter = ax.scatter(
    x, y,
    s=s,
    c=c,
    cmap='Blues',
    alpha=0.9,
    edgecolors='gray',
    linewidth=0.8,
    vmin=0, vmax=1
)

# ==========================================
# 4. Annotation (不加粗，适中字体)
# ==========================================
for i, row in df_melt.iterrows():
    text_color = 'white' if row['Score'] > 0.55 else 'black'

    ax.text(
        x_mapping[row['Model']],
        y_mapping[row['Metric']],
        f"{row['Value']:.4f}",
        ha='center',
        va='center',
        fontsize=11,   # ✅ 控制在论文常见范围
        color=text_color,
        fontname='Times New Roman'
    )

# Axis
ax.set_xlim(-0.5, len(models) - 0.5)
ax.set_ylim(-0.5, len(metrics) - 0.5)

ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=11)

ax.set_yticks(range(len(metrics)))
ax.set_yticklabels(metrics, fontsize=11)

ax.invert_yaxis()

# Labels（不加粗）
ax.set_xlabel('Model Variants', fontsize=12, labelpad=10)

# Grid
ax.set_axisbelow(True)
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

# Colorbar（简洁风）
cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.03)
cbar.set_label('Relative Improvement Score', fontsize=11)
cbar.ax.tick_params(labelsize=10)
cbar.outline.set_linewidth(0)

# Title（不过度强调）
ax.set_title(
    'Daily-level Ablation Study on Iron Ore Futures.',
    fontsize=18,
    pad=20
)

# ==========================================
# 5. Save
# ==========================================
plt.tight_layout()

output_file = 'bubble_heatmap_sci_I.png'
pdf_file = 'bubble_heatmap_sci_I.pdf'

plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.savefig(pdf_file, dpi=300, bbox_inches='tight')

print(f"Saved to: {output_file} & PDF")