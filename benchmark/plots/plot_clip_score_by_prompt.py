"""Grouped bar chart: mean CLIP score per prompt, one bar per model, from
the single-stage benchmark (benchmark/output/per_image_results.csv).
"""
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_CSV = Path(__file__).parent.parent / 'output' / 'per_image_results.csv'
PROMPTS = ['car', 'caucasian male', 'city landscape', 'northern lights', 'television with show playing']
MODELS = ['sd-turbo', 'sdxl-turbo', 'dreamshaper-lcm']
# ColorBrewer "Dark2" - muted, print/research-friendly qualitative palette.
COLORS = {'sd-turbo': '#1B9E77', 'sdxl-turbo': '#D95F02', 'dreamshaper-lcm': '#7570B3'}

with open(RESULTS_CSV, newline='') as f:
    rows = list(csv.DictReader(f))

# mean_clip[model][prompt] = mean clip_score across that (model, prompt)'s seeds
mean_clip = {
    model: {
        prompt: statistics.mean(
            float(r['clip_score']) for r in rows if r['model_key'] == model and r['prompt'] == prompt
        )
        for prompt in PROMPTS
    }
    for model in MODELS
}

x = np.arange(len(PROMPTS))
group_width = 0.25
bar_width = 0.21  # slightly narrower than the group spacing, so bars don't touch

fig, ax = plt.subplots(figsize=(10, 6))
for i, model in enumerate(MODELS):
    scores = [mean_clip[model][prompt] for prompt in PROMPTS]
    ax.bar(x + (i - 1) * group_width, scores, bar_width, label=model,
           color=COLORS[model], edgecolor='black', linewidth=0.8)

ax.set_xlabel('Prompt')
ax.set_ylabel('Mean CLIP score')
ax.set_title('Single-Stage CLIP Score by Prompt and Model')
ax.set_xticks(x)
ax.set_xticklabels(PROMPTS, rotation=15, ha='right')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

fig.tight_layout()
out_path = Path(__file__).parent / 'clip_score_by_prompt.png'
fig.savefig(out_path, dpi=150)
print(f'Wrote {out_path}')
