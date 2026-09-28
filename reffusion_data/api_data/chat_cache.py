"""Mirrors each chat's config, prompt history, and generated images into Redis.

sqlite (via the Django ORM) stays the source of truth; this is a
best-effort write-through cache next to it, so a chat's full context can be
read back with a handful of GETs instead of a DB query + image file reads.
Redis being unreachable never fails a request: writes are wrapped and only
log a warning.

Per chat, this writes:
- ``chat:<chat_id>:meta`` -- one YAML document with the chat's name,
  generation config, and prompt history (log_id + text per turn). Text
  only, so it stays small and human-readable via `redis-cli GET`.
- ``chat:<chat_id>:image:<log_id>`` -- that turn's generated image, raw
  PNG bytes.
- ``chat:<chat_id>:image:<log_id>:comparison`` -- that turn's comparison
  image, if it has one.
Images are kept out of the YAML doc rather than inlined as base64, so the
config/prompt doc stays cheap to read and the binary data stays in
Redis's native byte-string form.
"""

import logging

import redis
import yaml
from django.conf import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.REDIS_URL)
    return _client


def _meta_key(chat_id):
    return f'chat:{chat_id}:meta'


def _image_key(chat_id, log_id):
    return f'chat:{chat_id}:image:{log_id}'


def _comparison_key(chat_id, log_id):
    return f'chat:{chat_id}:image:{log_id}:comparison'


def sync_chat(chat):
    """Writes this chat's current name/config/prompts/images to Redis."""
    logs = list(chat.chat_log.order_by('log_id'))

    meta = {
        'chat_id': chat.chat_id,
        'name': chat.name,
        'config': {
            'use_conditioning': chat.use_conditioning,
            'conditioning_strength': chat.conditioning_strength,
            'num_inference_steps': chat.num_inference_steps,
        },
        'prompts': [{'log_id': log.log_id, 'text': log.text} for log in logs],
    }

    try:
        client = _get_client()
        pipe = client.pipeline()
        pipe.set(_meta_key(chat.chat_id), yaml.safe_dump(meta, sort_keys=False))
        for log in logs:
            if log.image:
                with log.image.open('rb') as f:
                    pipe.set(_image_key(chat.chat_id, log.log_id), f.read())
            if log.comparison_image:
                with log.comparison_image.open('rb') as f:
                    pipe.set(_comparison_key(chat.chat_id, log.log_id), f.read())
        pipe.execute()
    except redis.RedisError:
        logger.warning('Could not sync chat %s to Redis', chat.chat_id, exc_info=True)


def delete_chat(chat_id):
    """Removes every Redis key for this chat (mirrors a Chat.delete())."""
    try:
        client = _get_client()
        keys = client.keys(f'chat:{chat_id}:*')
        if keys:
            client.delete(*keys)
    except redis.RedisError:
        logger.warning('Could not delete chat %s from Redis', chat_id, exc_info=True)
