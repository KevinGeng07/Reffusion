# Design: Image-to-Image Chaining Across a Chat's Log History

## Purpose

Currently every image generation is stateless: `generate_image(prompt)` only
ever sees the current turn's prompt text, with no awareness of a chat's prior
log entries. This makes each prompt in a chat produce a completely
unrelated image, even though the whole point of a "chat" (as opposed to a
one-off generation) is a sequence of related edits/iterations.

This design makes generation #2 and #3 in a chat build on what came before:
the most recent prior image is used as an img2img reference, and all prior
prompts are folded into the new prompt's text, so both the visual and
textual history inform each new generation.

## Decisions

- **Reference image:** only the *most recent* prior log's image is passed to
  img2img (not all prior images — diffusers img2img takes a single `image=`
  input). Visual influence from earlier turns still carries forward
  transitively, since each image was itself derived from the one before it.
- **Prompt text:** all prior prompts in the chat, plus the new one, are
  concatenated (comma-joined) into a single prompt string sent to the model
  every time — not just the latest instruction on its own.
- **First log in a chat:** no prior image exists, so it always uses plain
  text-to-image, unchanged from today's behavior.
- **Step count / strength:** kept at Stability AI's documented recipe for
  `sd-turbo` — text2img `num_inference_steps=1, guidance_scale=0.0`
  (unchanged); img2img `num_inference_steps=2, strength=0.5,
  guidance_scale=0.0`. `sd-turbo` is distilled specifically for a 1-4 step
  regime and does not get more accurate with more steps — going higher (e.g.
  10-15) falls outside that regime and can degrade results rather than
  improve them, for no benefit. `effective_steps = int(num_inference_steps ×
  strength)`, so `strength=0.5` requires `num_inference_steps=2` as the
  minimum that does any denoising at all — `num_inference_steps=1` at that
  strength would compute to 0 effective steps and silently return the
  reference image unchanged.

## Backend

**`reffusion_data/api_data/image_model.py`** — `generate_image` gains an
optional `reference_image` parameter. Two module-level pipeline caches
instead of one, so both variants stay loaded without doubling memory use
(`AutoPipelineForImage2Image.from_pipe()` reuses the text2img pipeline's
already-loaded weights):

```python
_text2img_pipeline = None
_img2img_pipeline = None


def generate_image(prompt, reference_image=None):
    global _text2img_pipeline, _img2img_pipeline

    if reference_image is None:
        if _text2img_pipeline is None:
            _text2img_pipeline = _load_pipeline()
        result = _text2img_pipeline(
            prompt=prompt, num_inference_steps=1, guidance_scale=0.0
        )
    else:
        if _img2img_pipeline is None:
            if _text2img_pipeline is None:
                _text2img_pipeline = _load_pipeline()
            _img2img_pipeline = AutoPipelineForImage2Image.from_pipe(_text2img_pipeline)
        result = _img2img_pipeline(
            prompt=prompt, image=reference_image,
            num_inference_steps=2, strength=0.5, guidance_scale=0.0,
        )
    return result.images[0]
```

`image_model.py` stays framework-agnostic — it takes a prompt string and an
optional `PIL.Image`, with no knowledge of `Chat`/`ChatImages` or the
Django ORM.

**`reffusion_data/api_data/views.py`** — `chat_detail`'s POST branch builds
the concatenated prompt and loads the reference image before calling
`generate_image`, replacing the current `prompt = serial.validated_data['text']`
/ `image = generate_image(prompt)` pair:

```python
prompt = serial.validated_data['text']
prior_logs = list(chat.chat_log.order_by('log_id'))

if prior_logs:
    full_prompt = ', '.join(log.text for log in prior_logs) + ', ' + prompt
    reference_image = Image.open(prior_logs[-1].image.path)
    image = generate_image(full_prompt, reference_image=reference_image)
else:
    image = generate_image(prompt)
```

(`Image` here is `PIL.Image`, already imported transitively via
`image_model.py`'s return type — `views.py` needs its own
`from PIL import Image` added.)

## Testing

Existing tests mock `api_data.views.generate_image` entirely
(`@patch('api_data.views.generate_image')`), so they never touch the real
ML pipeline — that pattern continues to work unchanged since the call site
still goes through the same importable name, just with a new optional
kwarg. New/updated coverage in `reffusion_data/api_data/tests.py`:

- A test creating two existing log entries with real (tiny, in-memory) PNG
  images, then POSTing a third prompt, asserting `generate_image` was called
  with the three prompts comma-joined and a `reference_image` matching the
  second log's image.
- A test confirming the *first* log in a chat still calls `generate_image`
  with `reference_image=None` (i.e. positional prompt only, no second
  argument), preserving today's behavior.

`image_model.py`'s internals (the real diffusers/torch pipeline calls) are
not unit tested, consistent with the existing codebase — it has zero test
coverage today and this design doesn't change that; only the call boundary
in `views.py` is tested.

## Out of scope

- No frontend changes — `chat.html` already just POSTs `{text: prompt}` and
  displays whatever image comes back; nothing about the request/response
  shape changes.
- No new configuration surface (no per-request `strength` control, no model
  swap). If accuracy turns out to matter more than speed later, that's a
  separate decision (a non-turbo model, different hardware requirements).
