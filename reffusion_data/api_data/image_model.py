"""Loads text-to-image models on demand and generates images from prompts."""

import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

# Checkpoints a chat can be set to use. `guidance_scale` is a property of how
# each was trained/distilled, not something a user should be tuning per-request:
# - sd-turbo/sdxl-turbo are ADD-distilled for real-time, 1-4 step, *no*
#   classifier-free guidance generation (guidance_scale=0.0 is correct for
#   them, not a placeholder - a nonzero value actively hurts them).
# - flux2-klein is FLUX.2's distilled fast variant (same family as
#   FLUX.1-schnell), assumed guidance-distilled like the turbo pair above -
#   NOT verified locally: the fp8 checkpoint needs a CUDA GPU with >=13GB
#   VRAM (see IMPLEMENTATION_SPEC.md); this Mac (CPU/MPS only) cannot load
#   fp8 weights, so selecting this model here will fail until run on
#   suitable hardware.
# - realvisxl-v5 is an ordinary (non-distilled) SDXL finetune, so it needs
#   real CFG guidance; 5.0 is a reasonable starting point from the RealVisXL
#   family's typical recommended range (~3-7), not a value tuned for V5.0
#   specifically.
MODEL_CHOICES = {
    'sd-turbo': {
        'repo_id': 'stabilityai/sd-turbo',
        'label': 'SD-Turbo',
        'guidance_scale': 0.0,
    },
    'sdxl-turbo': {
        'repo_id': 'stabilityai/sdxl-turbo',
        'label': 'SDXL-Turbo',
        'guidance_scale': 0.0,
    },
    'flux2-klein': {
        'repo_id': 'black-forest-labs/FLUX.2-klein-4b-fp8',
        'label': 'FLUX.2 Klein 4B (fp8)',
        'guidance_scale': 0.0,
    },
    'realvisxl-v5': {
        'repo_id': 'SG161222/RealVisXL_V5.0',
        'label': 'RealVisXL V5.0',
        'guidance_scale': 5.0,
    },
}
DEFAULT_MODEL_KEY = 'sd-turbo'

_text2img_pipelines = {}
_img2img_pipelines = {}


def _load_pipeline(repo_id):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    pipeline = AutoPipelineForText2Image.from_pretrained(repo_id, torch_dtype=dtype)
    pipeline.to(device)
    return pipeline


def get_text2img_pipeline(repo_id):
    """Returns the cached text2img pipeline for `repo_id`, downloading and
    loading it first if this is the first time it's been used."""
    if repo_id not in _text2img_pipelines:
        _text2img_pipelines[repo_id] = _load_pipeline(repo_id)
    return _text2img_pipelines[repo_id]


def get_img2img_pipeline(repo_id):
    """Returns the cached img2img pipeline for `repo_id`, sharing weights
    with (and loading, if needed) that repo's text2img pipeline."""
    if repo_id not in _img2img_pipelines:
        _img2img_pipelines[repo_id] = AutoPipelineForImage2Image.from_pipe(
            get_text2img_pipeline(repo_id)
        )
    return _img2img_pipelines[repo_id]


def generate_image(
    prompt, reference_image=None, seed=None, strength=0.8, num_inference_steps=4,
    model_key=DEFAULT_MODEL_KEY,
):
    """Returns a PIL.Image generated from the given text prompt, using the
    model identified by `model_key` (see MODEL_CHOICES).

    If reference_image (a PIL.Image) is given, runs img2img using it as the
    starting point instead of pure text-to-image, with `strength` controlling
    how much of the reference survives. If seed is given, the generation is
    reproducible against another call using the same seed.
    """
    model = MODEL_CHOICES[model_key]
    repo_id = model['repo_id']
    guidance_scale = model['guidance_scale']

    if reference_image is None:
        pipeline = get_text2img_pipeline(repo_id)
    else:
        pipeline = get_img2img_pipeline(repo_id)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    if reference_image is None:
        result = pipeline(
            prompt=prompt, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, generator=generator,
        )
    else:
        # int(num_inference_steps * strength) is the number of denoising
        # steps diffusers actually runs; at 0 it crashes on an empty latent
        # tensor instead of a no-op, so floor strength to guarantee >=1 step
        # regardless of what strength/num_inference_steps were passed in.
        if int(num_inference_steps * strength) < 1:
            strength = 1 / num_inference_steps
        result = pipeline(
            prompt=prompt, image=reference_image,
            num_inference_steps=num_inference_steps, strength=strength,
            guidance_scale=guidance_scale, generator=generator,
        )
    return result.images[0]
