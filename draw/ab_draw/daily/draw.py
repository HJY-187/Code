import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

# ==========================================
# 0. Backend Configuration (Avoid errors in server environment)
# ==========================================
plt.switch_backend('Agg')

# ==========================================
# 1. Data Preparation
# ==========================================
data = {
    'Model': ['Main Path', 'Main Path+Aux 1', 'Main Path+Aux 2', 'DENet'],
    'MAE': [12.4864, 12.4631, 12.4453, 12.4065],
    'RMSE': [16.7595, 16.6226, 16.7700, 16.6202],
    'R2': [0.9567, 0.9574, 0.9566, 0.9574]
}

df = pd.DataFrame(data)

# Convert data to long format for scatter plot visualization
df_melt = df.melt(id_vars='Model', var_name='Metric', value_name='Value')


# ==========================================
# 2. Calculate Normalized Scores (for bubble size and color control)
# ==========================================
# Scoring logic definition:
# MAE/RMSE: Smaller values are better -> (Max - x) / (Max - Min)
# R2:       Larger values are better -> (x - Min) / (Max - Min)
def calculate_score(row):
    metric_data = df[row['Metric']]
    if row['Metric'] in ['MAE', 'RMSE']:
        return (metric_data.max() - row['Value']) / (metric_data.max() - metric_data.min())
    else:  # R2
        return (row['Value'] - metric_data.min()) / (metric_data.max() - metric_data.min())


df_melt['Score'] = df_melt.apply(calculate_score, axis=1)

# Prevent bubbles from disappearing or being too small due to zero scores
# Mapping range: [0, 1] -> [0.2, 1.0] (Ensure worst model still has visible bubbles)
df_melt['Plot_Size'] = 0.2 + 0.8 * df_melt['Score']

# ==========================================
# 3. SCI-style Plot Configuration
# ==========================================
# Font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# Create figure (adjust aspect ratio to optimize spacing)
fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)

# Map coordinate axes
metrics = ['MAE', 'RMSE', 'R2']  # Y-axis order
models = df['Model'].tolist()  # X-axis order

# Generate coordinate grid
x_mapping = {model: i for i, model in enumerate(models)}
y_mapping = {metric: i for i, metric in enumerate(metrics)}

# Extract plotting data
x = df_melt['Model'].map(x_mapping)
y = df_melt['Metric'].map(y_mapping)
s = df_melt['Plot_Size'] * 2200  # Bubble base size coefficient, adjustable
c = df_melt['Score']  # Color mapping value

# Draw bubbles (Scatter Plot)
# cmap: Use gradient colors like 'Blues', 'GnBu' or 'PuBu'
scatter = ax.scatter(
    x, y,
    s=s,
    c=c,
    cmap='Blues',
    alpha=0.9,  # Slight transparency
    edgecolors='gray',  # Bubble edge color
    linewidth=1.0,  # Edge width
    vmin=-0.1, vmax=1.1  # Adjust color range to avoid invisible lightest color
)

# ==========================================
# 4. Detailed Beautification and Annotation
# ==========================================

# Add value labels
for i, row in df_melt.iterrows():
    # Determine text color: white text for dark bubbles, black text for light bubbles
    text_color = 'white' if row['Score'] > 0.5 else 'black'
    font_weight = 'bold' if row['Score'] > 0.9 else 'normal'  # Bold optimal results

    ax.text(
        x_mapping[row['Model']],
        y_mapping[row['Metric']],
        f"{row['Value']:.4f}",
        ha='center',
        va='center',
        fontsize=11,
        color=text_color,
        weight='bold',  # Unified bold for better clarity
        fontname='Times New Roman'
    )

# Set axis range (leave some margin)
ax.set_xlim(-0.5, len(models) - 0.5)
ax.set_ylim(-0.5, len(metrics) - 0.5)

# Set axis labels
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=12, weight='bold')
ax.set_yticks(range(len(metrics)))
ax.set_yticklabels(metrics, fontsize=12, weight='bold')

# Reverse Y-axis to place MAE at the top (consistent with table reading habits, optional)
ax.invert_yaxis()

# Axis labels
ax.set_xlabel('Model Variants', fontsize=14, weight='bold', labelpad=10)
# ax.set_ylabel('Metrics', fontsize=14, weight='bold', labelpad=10) # Ylabel can usually be omitted here

# Grid line settings (place at the bottom layer)
ax.set_axisbelow(True)
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Add Colorbar - Optional, for explaining color intensity
cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.03)
cbar.set_label('Normalized Performance Score', fontsize=11, weight='bold')
cbar.outline.set_linewidth(0)  # Remove colorbar border

# Title
ax.set_title(
    'Performance Comparison Bubble Chart',
    fontsize=16,
    weight='bold',
    pad=20
)

# ==========================================
# 5. Save Image
# ==========================================
plt.tight_layout()  # Automatic tight layout

output_file = 'bubble_heatmap_sci.png'
pdf_file = 'bubble_heatmap_sci.pdf'

plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=False)
plt.savefig(pdf_file, dpi=300, bbox_inches='tight', transparent=False)

print(f"SCI-level bubble heatmap saved to: {output_file} (PNG & PDF)")