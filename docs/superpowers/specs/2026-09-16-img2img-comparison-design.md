# Design: Side-by-Side Img2Img Comparison Image

Extends [[2026-09-16-img2img-chaining-design]] (`docs/superpowers/specs/2026-09-16-img2img-chaining-design.md`), which is already implemented. That design made log #2+ in a chat use img2img (concatenated prior prompts + most-recent prior image as reference). This design adds, alongside that primary image, a second "what would this look like with the same random noise but no img2img conditioning" comparison image, so the two can be viewed side by side.

## Purpose

Let the user visually isolate what img2img conditioning is actually doing, by generating a second image from the same initial random noise but through the plain text2img pipeline instead.

## Decisions

- **"Same initial noise"** means a shared seeded `torch.Generator`, not literally identical starting latents — text2img and img2img start from architecturally different points (pure noise at max timestep vs. the reference image's encoded latents plus scaled noise at a timestep set by `strength`), so a shared seed is the closest honest equivalent, and isolates the random source as a controlled variable.
- **Comparison prompt:** the current turn's own prompt text alone — *not* the full concatenated history used by the primary img2img call. This is a deliberate choice: the comparison varies both the prompt and the reference-image conditioning, showing "what this turn alone would produce fresh" rather than a strict single-variable ablation.
- **Scope:** only log #2+ gets a comparison image. Log #1 has no img2img reference to compare against and is unchanged — single image, no comparison, no label.
- **Storage:** new `ChatImages.comparison_image` field, a real migration.

## Backend

**`reffusion_data/api_data/image_model.py`** — `generate_image` gains an optional `seed` parameter. Return type is unchanged (still a single `PIL.Image` per call); the seed just makes a given call reproducible against another call using the same seed.

```python
def generate_image(prompt, reference_image=None, seed=None):
    global _text2img_pipeline, _img2img_pipeline

    if reference_image is None:
        if _text2img_pipeline is None:
            _text2img_pipeline = _load_pipeline()
        pipeline = _text2img_pipeline
    else:
        if _img2img_pipeline is None:
            if _text2img_pipeline is None:
                _text2img_pipeline = _load_pipeline()
            _img2img_pipeline = AutoPipelineForImage2Image.from_pipe(_text2img_pipeline)
        pipeline = _img2img_pipeline

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    if reference_image is None:
        result = pipeline(
            prompt=prompt, num_inference_steps=1, guidance_scale=0.0,
            generator=generator,
        )
    else:
        result = pipeline(
            prompt=prompt, image=reference_image,
            num_inference_steps=2, strength=0.5, guidance_scale=0.0,
            generator=generator,
        )
    return result.images[0]
```

**`reffusion_data/api_data/views.py`**'s `chat_detail` POST branch, replacing the
`if prior_logs: ... else: ...` block from the img2img design:

```python
comparison_image = None

if prior_logs:
    full_prompt = ', '.join(log.text for log in prior_logs) + ', ' + prompt
    reference_image = Image.open(prior_logs[-1].image.path)
    seed = random.randint(0, 2**32 - 1)
    image = generate_image(full_prompt, reference_image=reference_image, seed=seed)
    comparison_image = generate_image(prompt, seed=seed)
else:
    image = generate_image(prompt)
```

Two separate `manual_seed(seed)` calls happen inside the two `generate_image`
calls (one per call, via the `seed` param) — passing a `torch.Generator`
into a pipeline call consumes/mutates its internal state, so the *same
object* can't be reused across both calls and still produce matching noise
the second time. Using the same seed *value* twice, on two freshly-seeded
generators, is what makes the noise match.

Saving both images onto the log entry:

```python
buffer = io.BytesIO()
image.save(buffer, format='PNG')
image_file = ContentFile(buffer.getvalue(), name=f'{chat.chat_id}_{l_id}.png')

comparison_image_file = None
if comparison_image is not None:
    comparison_buffer = io.BytesIO()
    comparison_image.save(comparison_buffer, format='PNG')
    comparison_image_file = ContentFile(
        comparison_buffer.getvalue(), name=f'{chat.chat_id}_{l_id}_comparison.png'
    )

serial.save(chat=chat, log_id=l_id, image=image_file, comparison_image=comparison_image_file)
```

`random` needs importing in `views.py`.

**`reffusion_data/api_data/models.py`** — new field on `ChatImages`:

```python
comparison_image = models.ImageField(upload_to='gens/', blank=True, null=True)
```

`null=True` here (unlike `image`, which has no `null=True`) because "no
comparison" is a genuine third state for log #1 entries, not just "not
uploaded yet."

**Migration**: a straightforward `AddField` schema migration — no data
migration needed, since existing rows correctly get `NULL`/empty for a
field that didn't exist for them.

**`reffusion_data/api_data/serializer.py`** — `ChatImagesSerial.Meta.fields`
gains `'comparison_image'`, following the same (not-read-only,
view-controlled) treatment as the existing `'image'` field.

**Chat deletion cleanup** — `chat_detail`'s `DELETE` branch's existing loop
(`for log in chat.chat_log.all(): if log.image: log.image.delete(save=False)`)
also deletes `comparison_image` files when present:

```python
if log.comparison_image:
    log.comparison_image.delete(save=False)
```

## Frontend

In `reffusion_data/api_data/templates/api_data/chat.html`, `renderLogs()`'s
per-entry markup: when `log.comparison_image` is present, render both images
side by side with captions; otherwise render exactly as today (single
unlabeled image, log #1's case).

```javascript
const hasComparison = Boolean(log.comparison_image);
div.innerHTML = `
  <p>${log.text}</p>
  ${hasComparison ? `
    <div class="log-images">
      <div class="log-image">
        <img src="${log.image}" alt="${log.text}">
        <div class="log-image-label">With reference</div>
      </div>
      <div class="log-image">
        <img src="${log.comparison_image}" alt="${log.text} (no reference)">
        <div class="log-image-label">Without reference (same seed)</div>
      </div>
    </div>
  ` : (log.image ? `<img src="${log.image}" alt="${log.text}">` : '')}
`;
```

New CSS:

```css
.log-images { display: flex; gap: 8px; }
.log-image { flex: 1; min-width: 0; }
.log-image img { max-width: 100%; border-radius: 4px; display: block; }
.log-image-label { font-size: 12px; color: #666; margin-top: 4px; text-align: center; }
```

## Testing

Following the existing `@patch('api_data.views.generate_image')` pattern:

- Log #1 (no prior logs): `generate_image` called exactly once, positional
  prompt only, no `reference_image`/`seed` kwargs — `comparison_image`
  stays empty on the saved row.
- Log #2+: `generate_image` called twice via `side_effect` (two distinct
  return images). Assert: first call gets the full concatenated prompt +
  `reference_image` + some `seed`; second call gets only the current
  turn's prompt + the *same* `seed` value, no `reference_image`. Assert
  the saved row's `comparison_image` is populated and its bytes match the
  second call's return value.
- Chat-delete test extended to also assert a `comparison_image` file is
  removed from disk when present.

`image_model.py`'s internals (the real `torch.Generator`/pipeline
mechanics) stay untested, consistent with the rest of the codebase — only
the `views.py` call boundary is tested.

## Out of scope

- No UI control over the seed (always random per request).
- No way to request a comparison for log #1.
- No change to the `strength`/step-count recipe established in the img2img
  chaining design.
