"""Background downloading of MODEL_CHOICES checkpoints, with real progress
(actual bytes transferred, not a simulated animation) so the UI can show
more than a blind spinner while a multi-GB model downloads.

Downloads are serialized through one worker thread - only one model
downloads at a time - so there's never ambiguity about which download a
given progress bar belongs to.

Getting real bytes out of huggingface_hub took two tries. It has two
separate download code paths with two separate, independently-patchable
"which tqdm class to use" hooks:
  - `file_download.http_get` (small/plain files) resolves its progress bar
    class via a free variable looked up fresh from `huggingface_hub.utils.
    tqdm`'s own module globals on every call, so patching that module's
    `tqdm` attribute affects it immediately.
  - `snapshot_download` (used for every multi-file repo, i.e. every model
    here) imported its own frozen copy of that class as `hf_tqdm` at
    package load time - patching `huggingface_hub.utils.tqdm.tqdm` doesn't
    touch that copy at all. Its two aggregate "whole repo" bars (transfer +
    reconstruction) only pick up a patched class if `huggingface_hub.
    _snapshot_download.hf_tqdm` itself is reassigned.
Both are patched below. Only `__init__` is overridden (to record the
instance) - NOT `update()`: huggingface_hub's own Xet-transfer code calls
`.update()` on these bar objects from worker threads in ways that, at least
in this diffusers/huggingface_hub version, sometimes touch a partially
disabled bar and raise AttributeError. Reading `.n`/`.total` back
afterwards (with getattr fallbacks, in case that same partial-init state is
ever observed here) sidesteps that entirely - it relies on tqdm's own
bookkeeping instead of reimplementing it.
"""

import importlib
import queue
import threading

from . import image_model

_hf_tqdm_module = importlib.import_module('huggingface_hub.utils.tqdm')
_snapshot_download_module = importlib.import_module('huggingface_hub._snapshot_download')

_lock = threading.Lock()
_status = {
    key: {'status': 'not_downloaded', 'downloaded_bytes': 0, 'total_bytes': 0}
    for key in image_model.MODEL_CHOICES
}
_done_events = {key: threading.Event() for key in image_model.MODEL_CHOICES}
_queue = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()


class _ProgressTqdm(_hf_tqdm_module.tqdm):
    """Swapped in for huggingface_hub's internal progress bar class(es)
    while a download runs. Every instance created while a download is
    active gets recorded, so its live .n/.total can be read back - see the
    module docstring for why this doesn't also override update()."""

    active_key = None
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _ProgressTqdm.active_key is not None:
            _ProgressTqdm.instances.append(self)


def _read_live_progress():
    """Best-effort (downloaded, total) bytes for the active download, read
    directly off whichever tracked bar currently has the largest total.
    huggingface_hub's aggregate "whole repo" bars end up with the true
    total repo size, dwarfing any single small file's bar, so this picks
    out the meaningful one without depending on its exact internal name.
    """
    try:
        instances = [b for b in _ProgressTqdm.instances if (getattr(b, 'total', 0) or 0) > 0]
        if not instances:
            return 0, 0
        bar = max(instances, key=lambda b: (getattr(b, 'total', 0) or 0))
        return (getattr(bar, 'n', 0) or 0), (getattr(bar, 'total', 0) or 0)
    except Exception:
        return 0, 0


def _download_and_load(model_key):
    repo_id = image_model.MODEL_CHOICES[model_key]['repo_id']
    with _lock:
        _status[model_key] = {'status': 'downloading', 'downloaded_bytes': 0, 'total_bytes': 0}

    _ProgressTqdm.active_key = model_key
    _ProgressTqdm.instances = []
    _hf_tqdm_module.tqdm = _ProgressTqdm
    _snapshot_download_module.hf_tqdm = _ProgressTqdm
    try:
        image_model.get_text2img_pipeline(repo_id)
        _, total = _read_live_progress()
        with _lock:
            _status[model_key]['status'] = 'ready'
            _status[model_key]['downloaded_bytes'] = total
            _status[model_key]['total_bytes'] = total
    except Exception:
        with _lock:
            _status[model_key]['status'] = 'error'
        raise
    finally:
        _ProgressTqdm.active_key = None
        _ProgressTqdm.instances = []
        _done_events[model_key].set()


def _worker():
    while True:
        model_key = _queue.get()
        try:
            _download_and_load(model_key)
        except Exception:
            pass  # status is already recorded as 'error'; nothing else to do
        finally:
            _queue.task_done()


def _ensure_worker():
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=_worker, daemon=True).start()
            _worker_started = True


def ensure_ready(model_key, background=False):
    """Makes sure `model_key`'s pipeline is downloaded and loaded.

    Already ready -> returns immediately. Already downloading (from an
    earlier call) -> joins that one instead of starting a second. Otherwise
    starts a download - in the background (returns immediately) if
    `background`, or blocking until it finishes otherwise.
    """
    if model_key not in image_model.MODEL_CHOICES:
        raise KeyError(model_key)

    with _lock:
        state = _status[model_key]['status']
        if state == 'ready':
            return
        already_running = state in ('queued', 'downloading')
        if not already_running:
            _status[model_key]['status'] = 'queued'
            _done_events[model_key].clear()

    _ensure_worker()
    if not already_running:
        _queue.put(model_key)

    if not background:
        _done_events[model_key].wait()


def get_statuses():
    with _lock:
        statuses = {key: dict(value) for key, value in _status.items()}

    active_key = _ProgressTqdm.active_key
    if active_key is not None and active_key in statuses:
        downloaded, total = _read_live_progress()
        if total:
            statuses[active_key]['downloaded_bytes'] = downloaded
            statuses[active_key]['total_bytes'] = total

    return statuses
