"""Measures real FLOPs for one stage1 (text2img) call and one stage2
(img2img) call per model, using the exact same settings the two-stage
benchmark used (num_inference_steps=4, strength=0.8) - so the numbers here
are directly comparable to that run's median latencies, not an abstract
per-step estimate. Writes benchmark/plots/flops.json.
"""
import json
import sys
from pathlib import Path

import torch
from diffusers import AutoPipelineForImage2Image
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_benchmark import MODEL_CHOICES, empty_device_cache, load_diffusion_pipeline, select_device
from run_benchmark_two_stage import DEFAULT_MODELS, REFINEMENT_PROMPT_TEMPLATE, REFINEMENT_STRENGTH

PROMPT = 'car'
NUM_INFERENCE_STEPS = 4


def run_and_count_flops(call):
    """FlopCounterMode only observes ops, it doesn't change what they
    compute - so the real result and its FLOP count come from one call."""
    with FlopCounterMode(display=False) as counter:
        result = call()
    return result, counter.get_total_flops()


def main():
    device = select_device()
    default_dtype = torch.float16 if device in ('cuda', 'mps') else torch.float32
    results = {}

    for model_key in DEFAULT_MODELS:
        print(f'=== {model_key} ===')
        model = MODEL_CHOICES[model_key]
        dtype = model.get('dtype') or default_dtype
        pipeline = load_diffusion_pipeline(model['repo_id'], device, dtype, model.get('load_kwargs'))
        img2img_pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)

        stage1_image, stage1_flops = run_and_count_flops(lambda: pipeline(
            prompt=PROMPT, num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=model['guidance_scale'],
            generator=torch.Generator(device=device).manual_seed(0),
        ).images[0])
        print(f'  stage1: {stage1_flops:,} FLOPs')

        refined_prompt = REFINEMENT_PROMPT_TEMPLATE.format(prompt=PROMPT)
        _, stage2_flops = run_and_count_flops(lambda: img2img_pipeline(
            prompt=refined_prompt, image=stage1_image, strength=REFINEMENT_STRENGTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=model['guidance_scale'],
            generator=torch.Generator(device=device).manual_seed(0),
        ).images[0])
        print(f'  stage2: {stage2_flops:,} FLOPs')

        results[model_key] = {'stage1': stage1_flops, 'stage2': stage2_flops}

        del pipeline, img2img_pipeline
        empty_device_cache(device)

    out_path = Path(__file__).parent / 'flops.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
