"""Two-stage variant of run_benchmark.py: stage 1 is the same plain
text2img generation as the single-stage benchmark, stage 2 feeds stage 1's
own output back into the same model via img2img with a "make it better"
refinement prompt, so both stages can be scored side-by-side to see whether
the refinement pass actually helps.

Shares MODEL_CHOICES, the standardized prompts/seeds, and all the scoring
machinery with run_benchmark.py (imported, not duplicated) - only the run()
loop and CSV/JSON schema differ, to hold both stages' images/scores per row.

Usage mirrors run_benchmark.py:
    python3 benchmark/run_benchmark_two_stage.py --output-dir ./benchmark/output_two_stage
Defaults to the three models already validated on this machine
(sd-turbo, sdxl-turbo, dreamshaper-lcm) - sd35-medium-turbo isn't included
by default since its own scope (seeds/steps) is still undecided.
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForImage2Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from run_benchmark import (
    CLIP_MODEL_ID,
    MODEL_CHOICES,
    PROMPTS,
    SEED_RANGE,
    build_seed_list,
    compute_lpips_diversity,
    empty_device_cache,
    load_aesthetic_predictor,
    load_diffusion_pipeline,
    pil_to_lpips_tensor,
    score_image,
    select_device,
    slugify,
)
import lpips

DEFAULT_MODELS = ['sd-turbo', 'sdxl-turbo', 'dreamshaper-lcm']

# strength controls how much stage 2 is allowed to change the image (the
# actual "intensity" dial); num_inference_steps is left equal to stage 1's
# rather than raised independently, since that would only make the schedule
# finer-grained without changing how much stage 2 actually changes - and for
# ADD/LCM-distilled models it isn't guaranteed to help anyway, as they're
# calibrated to specific low step counts rather than a general schedule.
REFINEMENT_STRENGTH = 0.8
REFINEMENT_PROMPT_TEMPLATE = (
    'this is bad. make {prompt} refined, highly detailed, sharper focus, and improved quality'
)


def load_existing_results(output_dir):
    per_image_rows = []
    per_image_path = output_dir / 'per_image_results.csv'
    if per_image_path.exists():
        with open(per_image_path, newline='') as f:
            for row in csv.DictReader(f):
                row['seed'] = int(row['seed'])
                for key in (
                    'stage1_clip_score', 'stage1_aesthetic_score', 'stage1_latency_s',
                    'stage2_clip_score', 'stage2_aesthetic_score', 'stage2_latency_s',
                ):
                    row[key] = float(row[key])
                per_image_rows.append(row)

    diversity_rows = []
    diversity_path = output_dir / 'diversity_results.csv'
    if diversity_path.exists():
        with open(diversity_path, newline='') as f:
            for row in csv.DictReader(f):
                row['lpips_mean'] = float(row['lpips_mean']) if row['lpips_mean'] else None
                row['lpips_std'] = float(row['lpips_std'])
                row['num_pairs'] = int(row['num_pairs'])
                diversity_rows.append(row)

    return per_image_rows, diversity_rows


def run(args):
    device = select_device()
    default_dtype = torch.float16 if device in ('cuda', 'mps') else torch.float32
    print(f'Device: {device} (default dtype: {default_dtype})')

    output_dir = Path(args.output_dir)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    seeds = build_seed_list(args.seeds_per_prompt)

    scoring_device = 'cpu'
    print('Loading CLIP + aesthetic predictor + LPIPS (shared across all models, kept on CPU)...')
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(scoring_device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    aesthetic_model = load_aesthetic_predictor(scoring_device)
    lpips_model = lpips.LPIPS(net='alex').to(scoring_device).eval()

    per_image_rows, diversity_rows = load_existing_results(output_dir)
    if per_image_rows:
        print(f'Loaded {len(per_image_rows)} existing per-image results from {output_dir}.')

    for model_key in (args.models or DEFAULT_MODELS):
        model = MODEL_CHOICES[model_key]
        dtype = model.get('dtype') or default_dtype
        print(f'\n=== {model_key} ({model["repo_id"]}, {dtype}) ===')
        try:
            pipeline = load_diffusion_pipeline(model['repo_id'], device, dtype, model.get('load_kwargs'))
            # Shares the already-loaded weights/components with the text2img
            # pipeline above (no duplicate memory) - same pattern
            # api_data/image_model.py's get_img2img_pipeline() uses.
            img2img_pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)
        except Exception as exc:
            print(f'  Skipping {model_key}: failed to load - {exc}')
            continue

        per_image_rows = [r for r in per_image_rows if r['model_key'] != model_key]
        diversity_rows = [r for r in diversity_rows if r['model_key'] != model_key]

        # Floors strength the same way api_data/image_model.py does for its
        # own img2img path, in case a future --num-inference-steps override
        # would otherwise round int(steps * strength) down to 0 and crash.
        strength = REFINEMENT_STRENGTH
        if int(args.num_inference_steps * strength) < 1:
            strength = 1 / args.num_inference_steps

        try:
            unlimited = args.keep_images <= 0
            for prompt in PROMPTS:
                prompt_dir = images_dir / model_key / slugify(prompt)
                prompt_dir.mkdir(parents=True, exist_ok=True)
                refined_prompt = REFINEMENT_PROMPT_TEMPLATE.format(prompt=prompt)

                stage1_tensors = []
                stage2_tensors = []
                for idx, seed in enumerate(tqdm(seeds, desc=f'{model_key} · {prompt}', unit='img')):
                    generator1 = torch.Generator(device=device).manual_seed(seed)
                    start = time.perf_counter()
                    stage1_image = pipeline(
                        prompt=prompt, num_inference_steps=args.num_inference_steps,
                        guidance_scale=model['guidance_scale'], generator=generator1,
                    ).images[0]
                    stage1_latency = time.perf_counter() - start

                    # Fresh generator seeded the same way, not a continuation
                    # of generator1's stream - keeps stage 2 independently
                    # reproducible regardless of how stage 1's call consumed
                    # its own generator internally.
                    generator2 = torch.Generator(device=device).manual_seed(seed)
                    start = time.perf_counter()
                    stage2_image = img2img_pipeline(
                        prompt=refined_prompt, image=stage1_image, strength=strength,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=model['guidance_scale'], generator=generator2,
                    ).images[0]
                    stage2_latency = time.perf_counter() - start

                    keep = unlimited or idx < args.keep_images
                    stage1_path = prompt_dir / f'stage1_seed_{seed}.png' if keep else None
                    stage2_path = prompt_dir / f'stage2_seed_{seed}.png' if keep else None
                    if stage1_path:
                        stage1_image.save(stage1_path)
                    if stage2_path:
                        stage2_image.save(stage2_path)

                    stage1_clip, stage1_aesthetic = score_image(
                        stage1_image, prompt, clip_model, clip_processor, aesthetic_model, scoring_device
                    )
                    stage2_clip, stage2_aesthetic = score_image(
                        stage2_image, prompt, clip_model, clip_processor, aesthetic_model, scoring_device
                    )
                    stage1_tensors.append(pil_to_lpips_tensor(stage1_image, scoring_device))
                    stage2_tensors.append(pil_to_lpips_tensor(stage2_image, scoring_device))

                    per_image_rows.append({
                        'model_key': model_key, 'prompt': prompt, 'seed': seed,
                        'stage1_clip_score': stage1_clip, 'stage1_aesthetic_score': stage1_aesthetic,
                        'stage1_latency_s': stage1_latency,
                        'stage1_image_path': str(stage1_path) if stage1_path else '',
                        'stage2_clip_score': stage2_clip, 'stage2_aesthetic_score': stage2_aesthetic,
                        'stage2_latency_s': stage2_latency,
                        'stage2_image_path': str(stage2_path) if stage2_path else '',
                    })

                    empty_device_cache(device)

                for stage, tensors in (('stage1', stage1_tensors), ('stage2', stage2_tensors)):
                    distances = compute_lpips_diversity(lpips_model, tensors, args.lpips_max_pairs)
                    diversity_rows.append({
                        'model_key': model_key, 'prompt': prompt, 'stage': stage,
                        'lpips_mean': statistics.mean(distances) if distances else None,
                        'lpips_std': statistics.stdev(distances) if len(distances) > 1 else 0.0,
                        'num_pairs': len(distances),
                    })
                mean1 = diversity_rows[-2]['lpips_mean']
                mean2 = diversity_rows[-1]['lpips_mean']
                mean1_str = f'{mean1:.4f}' if mean1 is not None else 'n/a'
                mean2_str = f'{mean2:.4f}' if mean2 is not None else 'n/a'
                print(f'  {prompt!r}: {len(seeds)} images, '
                      f'stage1 lpips_mean={mean1_str}, stage2 lpips_mean={mean2_str}')
        except Exception as exc:
            print(f'  Stopping {model_key} early - {exc}')
        finally:
            del pipeline
            del img2img_pipeline
            empty_device_cache(device)
            write_results(output_dir, per_image_rows, diversity_rows, args, seeds, strength)


def write_results(output_dir, per_image_rows, diversity_rows, args, seeds, strength):
    if not per_image_rows:
        print('\nNo images were generated (every model failed to load) - nothing to write.')
        return

    with open(output_dir / 'per_image_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    with open(output_dir / 'diversity_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(diversity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diversity_rows)

    models_run = sorted({row['model_key'] for row in per_image_rows})
    results = {}
    for model_key in models_run:
        model_rows = [r for r in per_image_rows if r['model_key'] == model_key]
        for stage in ('stage1', 'stage2'):
            stage_diversity = [
                r['lpips_mean'] for r in diversity_rows
                if r['model_key'] == model_key and r['stage'] == stage and r['lpips_mean'] is not None
            ]
            results.setdefault(model_key, {})[stage] = {
                'num_images': len(model_rows),
                'clip_score_mean': statistics.mean(r[f'{stage}_clip_score'] for r in model_rows),
                'aesthetic_score_mean': statistics.mean(r[f'{stage}_aesthetic_score'] for r in model_rows),
                'latency_s_mean': statistics.mean(r[f'{stage}_latency_s'] for r in model_rows),
                'lpips_diversity_mean': statistics.mean(stage_diversity) if stage_diversity else None,
            }

    summary = {
        'config': {
            'models_run': models_run,
            'prompts': PROMPTS,
            'seeds': seeds,
            'num_inference_steps': args.num_inference_steps,
            'refinement_strength': strength,
            'refinement_prompt_template': REFINEMENT_PROMPT_TEMPLATE,
            'keep_images_per_prompt': args.keep_images,
        },
        'results': results,
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n=== Summary ===')
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='*', choices=list(MODEL_CHOICES),
                         help=f'Subset of models to run (default: {DEFAULT_MODELS}).')
    parser.add_argument('--seeds-per-prompt', type=int, default=100,
                         help=f'Size of the single standardized seed list (sampled from the range '
                              f'{SEED_RANGE}) shared across every prompt and every model '
                              f'(default: 100).')
    parser.add_argument('--num-inference-steps', type=int, default=4,
                         help='Used for both stages (default: 4).')
    parser.add_argument('--lpips-max-pairs', type=int, default=500,
                         help='Cap on pairwise LPIPS comparisons per (model, prompt, stage) group '
                              '(default: 500).')
    parser.add_argument('--keep-images', type=int, default=5,
                         help='Only save this many stage1+stage2 image pairs per (model, prompt) '
                              'group to disk, in seed order. Every image is still generated and '
                              'used for CLIP/aesthetic/LPIPS scoring regardless - this only trims '
                              'what gets written as a PNG. Pass 0 or a negative number to keep '
                              'every image (default: 5).')
    parser.add_argument('--output-dir', default='./benchmark/output_two_stage')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
