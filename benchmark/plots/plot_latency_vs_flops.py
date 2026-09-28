"""Scatter plot: median generation latency vs. FLOPs, one point per
(model, stage). stage1's x-position is just its own FLOPs; stage2's
x-position is stage1 + stage2 FLOPs combined, since producing a stage2
image requires stage1's compute first - it's the true cumulative cost to
reach a refined image, not stage2's cost in isolation. A dashed line joins
each model's two points. Run measure_flops.py first to produce flops.json.
"""
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PLOTS_DIR = Path(__file__).parent
RESULTS_CSV = PLOTS_DIR.parent / 'output_two_stage' / 'per_image_results.csv'
FLOPS_JSON = PLOTS_DIR / 'flops.json'
COLORS = {'sd-turbo': 'tab:blue', 'sdxl-turbo': 'tab:orange', 'dreamshaper-lcm': 'tab:green'}
MARKERS = {'stage1': 'o', 'stage2': '^'}

with open(RESULTS_CSV, newline='') as f:
    rows = list(csv.DictReader(f))

with open(FLOPS_JSON) as f:
    flops = json.load(f)

fig, ax = plt.subplots(figsize=(9, 6))

for model_key, stage_flops in flops.items():
    model_rows = [r for r in rows if r['model_key'] == model_key]
    xs = {
        'stage1': stage_flops['stage1'],
        'stage2': stage_flops['stage1'] + stage_flops['stage2'],
    }
    stage1_latency = statistics.median(float(r['stage1_latency_s']) for r in model_rows)
    stage2_latency = statistics.median(float(r['stage2_latency_s']) for r in model_rows)
    ys = {
        # Mirrors the x-axis's cumulative treatment: stage2's y is the total
        # time to reach a refined image (stage1 + stage2), not stage2 alone
        # in isolation - guarantees stage2 >= stage1, so every line slopes
        # upward rather than sometimes dipping (as raw stage2-alone latency
        # did for sdxl-turbo/sd-turbo).
        'stage1': stage1_latency,
        'stage2': stage1_latency + stage2_latency,
    }

    ax.plot([xs['stage1'], xs['stage2']], [ys['stage1'], ys['stage2']],
             linestyle='--', color=COLORS[model_key], alpha=0.6, zorder=1)
    for stage in ('stage1', 'stage2'):
        ax.scatter(xs[stage], ys[stage], color=COLORS[model_key], marker=MARKERS[stage],
                   s=140, edgecolor='black', linewidth=0.5, zorder=2)

ax.set_xlabel('FLOPs (stage1 alone; stage2 = stage1 + stage2 combined)')
ax.set_ylabel('Median latency (s) (stage1 alone; stage2 = stage1 + stage2 combined)')
ax.set_title('Median Latency vs. FLOPs - Two-Stage Diffusion Benchmark')
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

# Evenly-spaced, round-number ticks (5T-25T) rather than one tick per exact
# data point - easier to read at a glance than the data's own odd values.
nice_ticks = [5e12, 10e12, 15e12, 20e12, 25e12]
ax.set_xticks(nice_ticks)
ax.set_xticklabels([f'{v / 1e12:.0f}T' for v in nice_ticks])
ax.set_xlim(3e12, 27e12)

# Two-part legend: color -> model, marker shape -> stage.
model_handles = [
    Line2D([0], [0], marker='s', color='none', markerfacecolor=color,
           markeredgecolor='black', markersize=10, label=model)
    for model, color in COLORS.items()
]
stage_handles = [
    Line2D([0], [0], marker=marker, color='none', markerfacecolor='gray',
           markeredgecolor='black', markersize=10, label=stage)
    for stage, marker in MARKERS.items()
]
ax.legend(handles=model_handles + stage_handles, loc='upper left', title='Color = model, shape = stage')

fig.tight_layout()
out_path = PLOTS_DIR / 'latency_vs_flops.png'
fig.savefig(out_path, dpi=150)
print(f'Wrote {out_path}')
