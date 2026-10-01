"""Grouped bar chart: stage2 minus stage1, per model, across CLIP score,
aesthetic score, and LPIPS diversity - does the refinement pass in the
two-stage benchmark help or hurt each metric, and by how much.
"""
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent.parent / 'output_two_stage'
MODELS = ['sd-turbo', 'sdxl-turbo', 'dreamshaper-lcm']
METRICS = ['CLIP score', 'Aesthetic score', 'LPIPS diversity']
COLORS = {'sd-turbo': '#1B9E77', 'sdxl-turbo': '#D95F02', 'dreamshaper-lcm': '#7570B3'}

with open(RESULTS_DIR / 'per_image_results.csv', newline='') as f:
    rows = list(csv.DictReader(f))
with open(RESULTS_DIR / 'diversity_results.csv', newline='') as f:
    diversity_rows = list(csv.DictReader(f))

# delta[model][metric] = stage2 mean - stage1 mean
delta = {}
for model in MODELS:
    model_rows = [r for r in rows if r['model_key'] == model]
    clip_delta = (
        statistics.mean(float(r['stage2_clip_score']) for r in model_rows)
        - statistics.mean(float(r['stage1_clip_score']) for r in model_rows)
    )
    aesthetic_delta = (
        statistics.mean(float(r['stage2_aesthetic_score']) for r in model_rows)
        - statistics.mean(float(r['stage1_aesthetic_score']) for r in model_rows)
    )
    stage1_lpips = [float(r['lpips_mean']) for r in diversity_rows if r['model_key'] == model and r['stage'] == 'stage1']
    stage2_lpips = [float(r['lpips_mean']) for r in diversity_rows if r['model_key'] == model and r['stage'] == 'stage2']
    lpips_delta = statistics.mean(stage2_lpips) - statistics.mean(stage1_lpips)
    delta[model] = [clip_delta, aesthetic_delta, lpips_delta]

x = np.arange(len(METRICS))
group_width = 0.25
bar_width = 0.21

fig, ax = plt.subplots(figsize=(9, 6))
for i, model in enumerate(MODELS):
    ax.bar(x + (i - 1) * group_width, delta[model], bar_width, label=model,
           color=COLORS[model], edgecolor='black', linewidth=0.8)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Metric')
ax.set_ylabel('Stage2 minus stage1')
ax.set_title('Two-Stage Refinement Effect by Model and Metric')
ax.set_xticks(x)
ax.set_xticklabels(METRICS)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

fig.tight_layout()
out_path = Path(__file__).parent.parent / 'images' / 'stage_delta.png'
fig.savefig(out_path, dpi=150)
print(f'Wrote {out_path}')
