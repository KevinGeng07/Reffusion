from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.forms import ValidationError

from .image_model import DEFAULT_MODEL_KEY, MODEL_CHOICES

MAX_CHATS = 5
MAX_CHAT_HIST = 5

# Create your models here.
class Account(models.Model):
    name = models.CharField() # Extract from user account.
    user_id = models.CharField(max_length=32, unique=True)
    next_chat_n = models.PositiveIntegerField(default=1)

    def __str__(self):
        return self.user_id

class Chat(models.Model):
    account = models.ForeignKey(
        to=Account, on_delete=models.CASCADE,
        related_name='chat'
    )

    chat_id = models.CharField(max_length=32) # Remove 'unique' when separate unique chat constraint below.
    name = models.CharField(max_length=50, blank=True) # User-set display name; falls back to chat_id when blank.

    # Per-chat img2img settings, applied from log #2 onward.
    use_conditioning = models.BooleanField(default=True)
    conditioning_strength = models.FloatField(
        default=0.8, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    num_inference_steps = models.PositiveSmallIntegerField(
        default=4, validators=[MinValueValidator(1), MaxValueValidator(10)]
    )
    model_key = models.CharField(
        max_length=32, default=DEFAULT_MODEL_KEY,
        choices=[(key, model['label']) for key, model in MODEL_CHOICES.items()],
    )

    def __str__(self):
        return self.chat_id

    def clean(self):
        if self.chat.count() >= MAX_CHATS:
            raise ValidationError(f'{MAX_CHATS} chats reached.')

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['account', 'chat_id'],
                name='unique_per_account'
            )
        ]

class TokenActivity(models.Model):
    """Tracks when a DRF auth token was last used, so ExpiringTokenAuthentication
    can reject it once too much time has passed since the last request -
    i.e. a rolling inactivity timeout, not a fixed token lifetime."""
    token_key = models.CharField(max_length=40, unique=True)
    last_seen = models.DateTimeField(auto_now=True)

class ChatImages(models.Model):
    chat = models.ForeignKey(
        to=Chat, on_delete=models.CASCADE,
        related_name='chat_log'
    )

    log_id = models.PositiveIntegerField()
    image = models.ImageField(upload_to='gens/', blank=True)
    comparison_image = models.ImageField(upload_to='gens/', blank=True, null=True)
    text = models.CharField(max_length=250)

    def __str__(self):
        return self

    def clean(self):
        if self.chat.chat_log.count() >= MAX_CHAT_HIST:
            raise ValidationError(f'{MAX_CHAT_HIST} logs reached.')

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['chat', 'log_id'],
                name='chat_unique'
            )
        ] 
