"""Benchmarks Reffusion's four MODEL_CHOICES checkpoints against a fixed set
of prompts and seeds, scoring each generated image on CLIP text-image
alignment and LAION aesthetic quality, and each (model, prompt) group on
LPIPS seed diversity and mean generation latency.

On a CUDA machine, run inside the accompanying Dockerfile (covers all four
models, flux2-klein included):
    docker build -t reffusion-benchmark benchmark/
    docker run --gpus all \
        -v $(pwd)/benchmark_output:/output \
        -v $(pwd)/hf_cache:/cache/huggingface \
        reffusion-benchmark

On Apple Silicon, skip the container - Docker Desktop on macOS can't reach
Metal/MPS at all (Apple doesn't expose it to containers), so running the
Dockerfile there means slow CPU-only inference for no reason. Run natively
against the same env the Django app uses instead:
    pip install -r benchmark/requirements.txt
    python3 benchmark/run_benchmark.py --output-dir ./benchmark_output
Device selection (cuda > mps > cpu) is automatic either way, matching
api_data/image_model.py's own logic. All four models now load on mps:
flux2-klein points at the bf16 black-forest-labs/FLUX.2-klein-4B rather than
the CUDA-only -4b-fp8 checkpoint (its README documents mps support
directly). A model that still fails to load for some other reason is
skipped with a logged message rather than crashing the run.

Deliberately standalone from the Django app - it only needs the model
registry (repo ids + guidance scales), not any of api_data's chat/account
machinery, so it mirrors api_data/image_model.py's MODEL_CHOICES rather than
importing it.
"""

import argparse
import csv
import itertools
import json
import logging
import random
import statistics
import time
import urllib.request
from pathlib import Path

import lpips
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# diffusers' ModelMixin.to() warns "should be kept in float32: []" whenever
# from_pretrained(torch_dtype=...) internally casts a model that doesn't
# declare `_keep_in_fp32_modules` (AutoencoderKL doesn't) - its check treats
# an empty list the same as a real one, so this fires unconditionally and
# isn't a sign of an actual precision problem. Silenced at the source logger
# rather than globally so other diffusers warnings still surface.
logging.getLogger('diffusers.models.modeling_utils').setLevel(logging.ERROR)

# Mirrors api_data/image_model.py's MODEL_CHOICES (repo_id + guidance_scale
# - kept separate on purpose, see module docstring). guidance_scale is a
# property of how each model was trained/distilled, not a free parameter:
# ADD-distilled turbo models want 0.0 (no classifier-free guidance), LCM
# distillation bakes guidance into the model itself so it stays cheap even
# at a high-looking value, and non-distilled/lightly-distilled models need
# real CFG (an actual second forward pass per step) at their own recommended
# scale.
MODEL_CHOICES = {
    'sd-turbo': {
        'repo_id': 'stabilityai/sd-turbo',
        'guidance_scale': 0.0,
    },
    'sdxl-turbo': {
        'repo_id': 'stabilityai/sdxl-turbo',
        'guidance_scale': 0.0,
    },
    'flux2-klein': {
        # Unquantized bf16 release, not the app's CUDA-only fp8 checkpoint
        # (fp8 kernels don't exist on mps). Its own usage example recommends
        # guidance_scale=1.0 rather than the ADD-distilled 0.0 above. Needs
        # more memory than a 16GB unified-memory machine can spare even with
        # VAE tiling, so treat it as CUDA-only.
        'repo_id': 'black-forest-labs/FLUX.2-klein-4B',
        'guidance_scale': 1.0,
        'dtype': torch.bfloat16,
    },
    'dreamshaper-lcm': {
        # LCM distillation, native 768x768 - bigger than sd-turbo/sdxl-
        # turbo's 512, smaller than a standard SDXL finetune's 1024.
        # guidance_scale=8.0 looks high but LCM bakes guidance into the
        # model as a conditioning embedding instead of running a second
        # unconditional forward pass, so it's still single-pass-per-step.
        'repo_id': 'SimianLuo/LCM_Dreamshaper_v7',
        'guidance_scale': 8.0,
    },
    'sd35-medium-turbo': {
        # SD3.5 Medium's MMDiT-X transformer, distilled for few-step
        # generation (tensorart's checkpoint recommends guidance_scale=1.5,
        # ~8 steps, vs. the base model's non-distilled 40-step/4.5
        # recommendation). text_encoder_3/tokenizer_3 dropped via
        # load_kwargs: the optional T5-XXL encoder alone is ~5GB, more than
        # this benchmark's memory budget can spare alongside the rest of
        # the pipeline.
        'repo_id': 'tensorart/stable-diffusion-3.5-medium-turbo',
        'guidance_scale': 1.5,
        'load_kwargs': {'text_encoder_3': None, 'tokenizer_3': None},
    },
}

# The standardized prompt set. All 5 prompts share the exact same 100 seeds
# (sampled once below), and every model uses that same set too - so a given
# seed means the same RNG state everywhere it appears, making outputs
# directly comparable across the whole (model, prompt, seed) grid.
PROMPTS = [
    'car',
    'caucasian male',
    'city landscape',
    'northern lights',
    'television with show playing',
]

# Sampling the seed list once, from a fixed RNG, makes reruns reproducible
# and guarantees every prompt and every model sees the exact same seeds.
SEED_SAMPLING_RNG_SEED = 0
SEED_RANGE = (1, 1000)

# ViT-L/14 specifically: the aesthetic predictor below was trained on this
# model's 768-dim image embeddings, so CLIP score and aesthetic score share
# one forward pass through the same encoder.
CLIP_MODEL_ID = 'openai/clip-vit-large-patch14'
AESTHETIC_PREDICTOR_PATH = Path(__file__).parent / 'aesthetic-predictor.pth'
AESTHETIC_PREDICTOR_URL = (
    'https://raw.githubusercontent.com/christophschuhmann/'
    'improved-aesthetic-predictor/main/sac+logos+ava1-l14-linearMSE.pth'
)


def build_seed_list(num_seeds):
    """One standardized seed list, shared across every prompt and every
    model - not sampled per-prompt, so results line up seed-for-seed across
    the whole grid."""
    rng = random.Random(SEED_SAMPLING_RNG_SEED)
    return sorted(rng.sample(range(SEED_RANGE[0], SEED_RANGE[1] + 1), num_seeds))


def slugify(text):
    return '-'.join(text.lower().split())


class AestheticPredictor(torch.nn.Module):
    """LAION's 'improved aesthetic predictor': a 5-layer MLP regressing
    L2-normalized CLIP ViT-L/14 image embeddings onto human aesthetic
    ratings (roughly 1-10). Architecture and checkpoint from
    https://github.com/christophschuhmann/improved-aesthetic-predictor -
    reimplemented as a plain nn.Module (the original is a
    pytorch_lightning.LightningModule) since only its `layers` state dict
    is needed here."""

    def __init__(self, input_size=768):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def load_aesthetic_predictor(device):
    if not AESTHETIC_PREDICTOR_PATH.exists():
        # Dockerfile bakes this in at build time; the native run path has no
        # such step, so fetch it here on first use instead of crashing.
        print(f'Downloading aesthetic predictor checkpoint to {AESTHETIC_PREDICTOR_PATH}...')
        urllib.request.urlretrieve(AESTHETIC_PREDICTOR_URL, AESTHETIC_PREDICTOR_PATH)

    model = AestheticPredictor()
    state_dict = torch.load(AESTHETIC_PREDICTOR_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def select_device():
    """cuda > mps > cpu - mirrors api_data/image_model.py's _load_pipeline
    so a run on this Mac exercises the same device path the app itself
    would use."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def empty_device_cache(device):
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


def load_diffusion_pipeline(repo_id, device, dtype, load_kwargs=None):
    pipeline = AutoPipelineForText2Image.from_pretrained(repo_id, torch_dtype=dtype, **(load_kwargs or {}))
    pipeline.to(device)
    # Replaced below by one tqdm bar per (model, prompt) over the seed list
    # (e.g. 43/100) - diffusers' own per-step bar (4/4, repeated once per
    # image) is noise at this scale.
    pipeline.set_progress_bar_config(disable=True)
    # VAE decode upsamples latents to full pixel resolution in one shot,
    # which is the single biggest memory spike in the whole forward pass -
    # tiling/slicing caps that on unified-memory machines (16GB Macs OOM
    # on flux2-klein's decode otherwise, since CLIP/aesthetic/LPIPS also
    # stay resident on the same device for the whole run).
    if hasattr(pipeline, 'vae'):
        pipeline.vae.enable_tiling()
        pipeline.vae.enable_slicing()
    # Some repos (e.g. LCM_Dreamshaper_v7) bundle the default SD NSFW safety
    # checker, which silently replaces any flagged image with a solid black
    # square rather than raising - discovered because it false-positived on
    # ~31% of one prompt's outputs (shirtless-figure prompts, the same kind
    # of content the other models here already generate cleanly), quietly
    # corrupting that prompt's CLIP/aesthetic/LPIPS scores with duplicate
    # black-image rows. Not appropriate for a benchmark measuring model
    # quality, so disabled uniformly for every model.
    if getattr(pipeline, 'safety_checker', None) is not None:
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
    return pipeline


def pil_to_lpips_tensor(image, device):
    """[-1, 1]-normalized CHW tensor, the format lpips.LPIPS() expects."""
    arr = np.asarray(image.convert('RGB'), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def score_image(image, prompt, clip_model, clip_processor, aesthetic_model, device):
    inputs = clip_processor(text=[prompt], images=[image], return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        # torchmetrics' CLIPScore convention: 100 * max(cosine_similarity, 0).
        cosine_sim = (image_embeds @ text_embeds.T).item()
        clip_score = 100 * max(cosine_sim, 0.0)
        aesthetic_score = aesthetic_model(image_embeds).item()
    return clip_score, aesthetic_score


def compute_lpips_diversity(lpips_model, tensors, max_pairs):
    """Mean/stdev pairwise LPIPS distance across a group of same-prompt
    images - higher means the model gives visually distinct outputs across
    seeds, not just near-duplicates with prompt-following noise."""
    pairs = list(itertools.combinations(range(len(tensors)), 2))
    if len(pairs) > max_pairs:
        pairs = random.Random(SEED_SAMPLING_RNG_SEED).sample(pairs, max_pairs)

    distances = []
    with torch.no_grad():
        for i, j in pairs:
            distances.append(lpips_model(tensors[i], tensors[j]).item())
    return distances


def load_existing_results(output_dir):
    """Reloads a prior run's CSVs (if present) so this run's writes merge
    with them instead of starting from an empty in-memory slate and
    clobbering other models' already-computed results."""
    per_image_rows = []
    per_image_path = output_dir / 'per_image_results.csv'
    if per_image_path.exists():
        with open(per_image_path, newline='') as f:
            for row in csv.DictReader(f):
                row['seed'] = int(row['seed'])
                row['clip_score'] = float(row['clip_score'])
                row['aesthetic_score'] = float(row['aesthetic_score'])
                row['latency_s'] = float(row['latency_s'])
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

    # Kept on CPU deliberately, not `device`: these three stay resident for
    # the entire run (all four diffusion models share them), so leaving them
    # on mps permanently taxes the same unified-memory pool the current
    # diffusion pipeline needs - that's what left flux2-klein's VAE decode
    # only ~2GB of headroom before OOMing. They're small relative to a
    # diffusion model, so CPU inference on them is cheap next to a single
    # image generation.
    scoring_device = 'cpu'
    print('Loading CLIP + aesthetic predictor + LPIPS (shared across all models, kept on CPU)...')
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(scoring_device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    aesthetic_model = load_aesthetic_predictor(scoring_device)
    lpips_model = lpips.LPIPS(net='alex').to(scoring_device).eval()

    # Picks up where a prior process left off, not just a prior crash within
    # this one - e.g. rerunning a single --models after swapping its config
    # (a different checkpoint, guidance_scale, step count, ...) merges with
    # whatever other models' results are already on disk instead of wiping
    # them, since write_results() below always overwrites both CSVs with
    # whatever's in these two lists.
    per_image_rows, diversity_rows = load_existing_results(output_dir)
    if per_image_rows:
        print(f'Loaded {len(per_image_rows)} existing per-image results from {output_dir}.')

    for model_key in (args.models or list(MODEL_CHOICES)):
        model = MODEL_CHOICES[model_key]
        dtype = model.get('dtype') or default_dtype
        print(f'\n=== {model_key} ({model["repo_id"]}, {dtype}) ===')
        try:
            pipeline = load_diffusion_pipeline(model['repo_id'], device, dtype, model.get('load_kwargs'))
        except Exception as exc:
            print(f'  Skipping {model_key}: failed to load - {exc}')
            continue

        # Drop any rows this exact model_key contributed on a prior run
        # before regenerating it fresh - otherwise a rerun (e.g. after
        # swapping its checkpoint) would end up with both the old and new
        # versions' rows mixed together in the merged results.
        per_image_rows = [r for r in per_image_rows if r['model_key'] != model_key]
        diversity_rows = [r for r in diversity_rows if r['model_key'] != model_key]

        # A crash partway through generation (e.g. an OOM on a later prompt)
        # used to take the whole run down with it, discarding every already-
        # completed model's results along with it since they were only ever
        # written to disk once at the very end. Now it only aborts the
        # current model - whatever it already generated stays in
        # per_image_rows/diversity_rows, and the write below persists
        # everything (this model's partial results plus every prior model's)
        # before moving on.
        try:
            unlimited = args.keep_images <= 0
            for prompt in PROMPTS:
                prompt_dir = images_dir / model_key / slugify(prompt)
                prompt_dir.mkdir(parents=True, exist_ok=True)

                lpips_tensors = []
                for idx, seed in enumerate(tqdm(seeds, desc=f'{model_key} · {prompt}', unit='img')):
                    generator = torch.Generator(device=device).manual_seed(seed)
                    start = time.perf_counter()
                    image = pipeline(
                        prompt=prompt, num_inference_steps=args.num_inference_steps,
                        guidance_scale=model['guidance_scale'], generator=generator,
                    ).images[0]
                    latency = time.perf_counter() - start

                    # Every image is scored and fed to LPIPS regardless of
                    # --keep-images - only whether it's written to disk changes.
                    keep = unlimited or idx < args.keep_images
                    image_path = prompt_dir / f'seed_{seed}.png' if keep else None
                    if image_path:
                        image.save(image_path)

                    clip_score, aesthetic_score = score_image(
                        image, prompt, clip_model, clip_processor, aesthetic_model, scoring_device
                    )
                    lpips_tensors.append(pil_to_lpips_tensor(image, scoring_device))

                    per_image_rows.append({
                        'model_key': model_key, 'prompt': prompt, 'seed': seed,
                        'clip_score': clip_score, 'aesthetic_score': aesthetic_score,
                        'latency_s': latency, 'image_path': str(image_path) if image_path else '',
                    })

                    # mps's caching allocator doesn't reclaim fragments on its
                    # own across hundreds of sequential generations - left
                    # unchecked this is the classic cause of a run getting
                    # progressively slower (and eventually swapping) the longer
                    # it goes, independent of the OOM fixed above.
                    empty_device_cache(device)

                distances = compute_lpips_diversity(lpips_model, lpips_tensors, args.lpips_max_pairs)
                diversity_rows.append({
                    'model_key': model_key, 'prompt': prompt,
                    'lpips_mean': statistics.mean(distances) if distances else None,
                    'lpips_std': statistics.stdev(distances) if len(distances) > 1 else 0.0,
                    'num_pairs': len(distances),
                })
                mean_str = f'{diversity_rows[-1]["lpips_mean"]:.4f}' if distances else 'n/a'
                print(f'  {prompt!r}: {len(seeds)} images, lpips_mean={mean_str}')
        except Exception as exc:
            print(f'  Stopping {model_key} early - {exc}')
        finally:
            del pipeline
            empty_device_cache(device)
            write_results(output_dir, per_image_rows, diversity_rows, args, seeds)


def write_results(output_dir, per_image_rows, diversity_rows, args, seeds):
    if not per_image_rows:
        print('\nNo images were generated (every model failed to load) - nothing to write.')
        return

    # One row per generated image regardless of --keep-images - image_path
    # is '' for any image that was scored but not written to disk.
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
        model_diversity = [
            r['lpips_mean'] for r in diversity_rows
            if r['model_key'] == model_key and r['lpips_mean'] is not None
        ]
        results[model_key] = {
            'num_images': len(model_rows),
            'clip_score_mean': statistics.mean(r['clip_score'] for r in model_rows),
            'aesthetic_score_mean': statistics.mean(r['aesthetic_score'] for r in model_rows),
            'latency_s_mean': statistics.mean(r['latency_s'] for r in model_rows),
            'lpips_diversity_mean': statistics.mean(model_diversity) if model_diversity else None,
        }

    # The run's own config lives alongside its results so summary.json is a
    # self-contained record: which models/prompts/seeds produced these
    # numbers, not just the numbers themselves.
    summary = {
        'config': {
            'models_run': models_run,
            'prompts': PROMPTS,
            'seeds': seeds,
            'num_inference_steps': args.num_inference_steps,
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
                         help='Subset of models to run (default: all four).')
    parser.add_argument('--seeds-per-prompt', type=int, default=100,
                         help=f'Size of the single standardized seed list (sampled from the range '
                              f'{SEED_RANGE}) shared across every prompt and every model '
                              f'(default: 100).')
    parser.add_argument('--num-inference-steps', type=int, default=4,
                         help='Matches the app default in Chat.num_inference_steps (default: 4).')
    parser.add_argument('--lpips-max-pairs', type=int, default=500,
                         help='Cap on pairwise LPIPS comparisons per (model, prompt) group, to '
                              'bound runtime when seeds-per-prompt is large (default: 500).')
    parser.add_argument('--keep-images', type=int, default=5,
                         help='Only save this many images per (model, prompt) group to disk, '
                              'in seed order - the saved images are the same seeds across every '
                              'prompt and model, since the seed list is standardized. Every '
                              'image is still generated and used for CLIP/aesthetic/LPIPS '
                              'scoring regardless - this only trims what gets written as a PNG. '
                              'Pass 0 or a negative number to keep every image (default: 5).')
    parser.add_argument('--output-dir', default='/output')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
