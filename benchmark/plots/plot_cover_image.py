"""Cover image: stage1 -> stage2 sample pairs, grouped one row per model,
each row boxed in that model's color (matching the other plots), centered
in the figure, and labeled with the model name above the box. Uses
manually placed axes rather than a uniform subplot grid, since rows have
different numbers of samples.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

IMAGES_DIR = Path(__file__).parent.parent / 'output_two_stage' / 'images'
# ColorBrewer "Dark2", same mapping used in the other plots.
COLORS = {'sd-turbo': '#1B9E77', 'sdxl-turbo': '#D95F02', 'dreamshaper-lcm': '#7570B3'}

# model_key -> list of (prompt slug, seed) samples shown in that model's row
ROWS = [
    ('dreamshaper-lcm', [('car', 65), ('northern-lights', 15)]),
    ('sd-turbo', [('city-landscape', 64), ('television-with-show-playing', 42)]),
    ('sdxl-turbo', [('caucasian-male', 2)]),
]

# All sizes in figure-fraction units, chosen so a 2-sample row fits with
# margin to spare and the arrow gap is wide enough that the arrow glyph
# never touches either image.
IMG_W, ARROW_GAP, SAMPLE_GAP = 0.16, 0.035, 0.06
ROW_H, ROW_VGAP = 0.22, 0.08
TOP_MARGIN, LABEL_OFFSET = 0.045, 0.008

fig = plt.figure(figsize=(8, 6.5))

for row, (model, samples) in enumerate(ROWS):
    row_top = 1 - TOP_MARGIN - row * (ROW_H + ROW_VGAP)
    row_bottom = row_top - ROW_H
    row_width = len(samples) * 2 * IMG_W + (len(samples) - 1) * SAMPLE_GAP + len(samples) * ARROW_GAP
    row_left = 0.5 - row_width / 2  # every row centered in the figure, not just sdxl-turbo's

    x = row_left
    for prompt_slug, seed in samples:
        stage1_ax = fig.add_axes([x, row_bottom, IMG_W, ROW_H])
        stage1_ax.imshow(Image.open(IMAGES_DIR / model / prompt_slug / f'stage1_seed_{seed}.png'))
        stage1_ax.axis('off')
        x += IMG_W

        arrow_center_x = x + ARROW_GAP / 2
        fig.text(arrow_center_x, (row_top + row_bottom) / 2, '→',
                  fontsize=20, ha='center', va='center')
        x += ARROW_GAP

        stage2_ax = fig.add_axes([x, row_bottom, IMG_W, ROW_H])
        stage2_ax.imshow(Image.open(IMAGES_DIR / model / prompt_slug / f'stage2_seed_{seed}.png'))
        stage2_ax.axis('off')
        x += IMG_W + SAMPLE_GAP

    pad = 0.008
    box = Rectangle(
        (row_left - pad, row_bottom - pad), row_width + 2 * pad, ROW_H + 2 * pad,
        fill=False, edgecolor=COLORS[model], linewidth=2.5, transform=fig.transFigure,
    )
    fig.add_artist(box)
    fig.text(row_left - pad, row_top + pad + LABEL_OFFSET, model, ha='left', va='bottom',
              fontsize=12, fontweight='bold', color=COLORS[model])

CAPTION = (
    'Figure 1. Left column of each image pair is the prompt-only pass, '
    'right column is same seed conditioned on the first pass.'
)
BOTTOM_GAP = 0.06
fig.text(0.5, row_bottom - BOTTOM_GAP, CAPTION, ha='center', va='top',
          fontsize=9, style='italic', wrap=True)

out_path = Path(__file__).parent / 'cover_image.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Wrote {out_path}')
