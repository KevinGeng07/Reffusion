"""A DRF token auth class with a rolling inactivity timeout: a token stops
working once SESSION_TIMEOUT has passed since the last request that used
it, rather than being valid forever (the default) or for a fixed lifetime
from login."""

from datetime import timedelta

from django.utils import timezone
from rest_framework.authentication import TokenAuthentication
from rest_framework.exceptions import AuthenticationFailed

from .models import TokenActivity

SESSION_TIMEOUT = timedelta(minutes=5)


class ExpiringTokenAuthentication(TokenAuthentication):
    def authenticate_credentials(self, key):
        user, token = super().authenticate_credentials(key)

        activity, _ = TokenActivity.objects.get_or_create(token_key=key)
        if timezone.now() - activity.last_seen > SESSION_TIMEOUT:
            raise AuthenticationFailed('Session expired due to inactivity.')

        activity.save(update_fields=['last_seen'])  # auto_now bumps it to now
        return user, token
