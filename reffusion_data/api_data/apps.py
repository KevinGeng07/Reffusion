import os
import sys

from django.apps import AppConfig


class DataConfig(AppConfig):
    name = 'api_data'

    def ready(self):
        # Only in the actual serving process (not `migrate`/`test`/etc, and
        # not the reloader's watcher process, which never handles requests).
        if 'runserver' not in sys.argv or os.environ.get('RUN_MAIN') != 'true':
            return

        from . import model_downloads
        from .image_model import DEFAULT_MODEL_KEY
        model_downloads.ensure_ready(DEFAULT_MODEL_KEY, background=True)
