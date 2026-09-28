### TODO: BUILD DJANGO USERS CRUD.

import io
import random

from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.db import transaction
from django.shortcuts import get_object_or_404
from PIL import Image
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.decorators import api_view, permission_classes
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from . import chat_cache, model_downloads
from .models import *
from .serializer import *
from .image_model import MODEL_CHOICES, generate_image


class LoginView(ObtainAuthToken):
    """POST {username, password} -> {token}. Open to anyone (that's the
    point - everything else requires the token this returns)."""
    permission_classes = [AllowAny]


@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    """POST {username, password} -> {token}. Creates the Django login and
    its matching Account together, atomically, so there's never a login
    that lacks the Account record the rest of the API assumes exists."""
    username = (request.data.get('username') or '').strip()
    password = request.data.get('password') or ''

    if not username or not password:
        return Response(
            {'detail': 'username and password are required.'}, status.HTTP_400_BAD_REQUEST
        )
    if User.objects.filter(username=username).exists():
        return Response(
            {'detail': 'That username is already taken.'}, status.HTTP_400_BAD_REQUEST
        )

    with transaction.atomic():
        user = User.objects.create_user(username=username, password=password)
        Account.objects.create(user_id=username, name=username)
        token = Token.objects.create(user=user)

    return Response({'token': token.key}, status.HTTP_201_CREATED)


@api_view(['POST'])
def logout(request):
    """Deletes the caller's own token, so it stops working immediately
    (rather than relying solely on the inactivity timeout eventually
    catching up)."""
    TokenActivity.objects.filter(token_key=request.auth.key).delete()
    request.auth.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


def _ensure_owns_account(request, user_id):
    if request.user.username != user_id:
        raise PermissionDenied('Not your account.')


@api_view(['GET'])
def list_models(request):
    statuses = model_downloads.get_statuses()
    return Response([
        {'key': key, 'label': model['label'], **statuses[key]}
        for key, model in MODEL_CHOICES.items()
    ])


@api_view(['POST'])
def warm_model(request, model_key):
    if model_key not in MODEL_CHOICES:
        return Response({'detail': 'Unknown model.'}, status.HTTP_404_NOT_FOUND)
    model_downloads.ensure_ready(model_key, background=True)
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['GET'])
def get_accounts(request):
    accounts = Account.objects.filter(user_id=request.user.username)

    serial = AccountSerial(accounts, many=True)
    return Response(serial.data)

@api_view(['GET'])
def get_account(request, user_id):
    _ensure_owns_account(request, user_id)
    account = get_object_or_404(Account, user_id=user_id)

    serial = AccountSerial(account, many=False)
    return Response(serial.data)


# PUT changes the entire resource, PATCH changes one field.
@api_view(['POST'])
def create_chat(request, user_id):
    _ensure_owns_account(request, user_id)
    account = get_object_or_404(Account, user_id=user_id)

    if account.chat.count() >= MAX_CHATS:
        return Response(
            {'detail': f'{MAX_CHATS} chats reached.'}, status.HTTP_400_BAD_REQUEST
        )

    serial = ChatSerial(data=request.data)

    if serial.is_valid():
        chat_id = f'chat-{account.next_chat_n}'
        chat = serial.save(account=account, chat_id=chat_id)
        account.next_chat_n += 1
        account.save(update_fields=['next_chat_n'])
        chat_cache.sync_chat(chat)
        model_downloads.ensure_ready(chat.model_key, background=True)
        return Response(serial.data, status.HTTP_201_CREATED)
    return Response(serial.errors, status.HTTP_400_BAD_REQUEST)


@api_view(['POST', 'DELETE'])
def chat_detail(request, user_id, chat_id):
    _ensure_owns_account(request, user_id)
    account = get_object_or_404(Account, user_id=user_id)
    chat = get_object_or_404(account.chat, chat_id=chat_id)

    if request.method == 'DELETE':
        for log in chat.chat_log.all():
            if log.image:
                log.image.delete(save=False)
            if log.comparison_image:
                log.comparison_image.delete(save=False)
        chat.delete()
        chat_cache.delete_chat(chat_id)
        return Response(status=status.HTTP_204_NO_CONTENT)

    if chat.chat_log.count() >= MAX_CHAT_HIST:
        return Response(
            {'detail': f'{MAX_CHAT_HIST} logs reached.'}, status.HTTP_400_BAD_REQUEST
        )

    serial = ChatImagesSerial(data=request.data)

    if serial.is_valid():
        last_log_id = chat.chat_log.order_by('log_id').last()

        if not last_log_id:
            l_id = 1
        else:
            l_id = last_log_id.log_id + 1

        prompt = serial.validated_data['text']
        prior_logs = list(chat.chat_log.order_by('log_id'))
        comparison_image = None

        # No-op if already downloaded; otherwise blocks here (rather than
        # inside generate_image) so a chat's first generation waits on the
        # same tracked/queued download a client may already be polling
        # progress for, instead of starting a second one.
        model_downloads.ensure_ready(chat.model_key)

        if prior_logs:
            full_prompt = ', '.join(log.text for log in prior_logs) + ', ' + prompt

            if chat.use_conditioning:
                reference_image = Image.open(prior_logs[-1].image.path)
                seed = random.randint(0, 2**32 - 1)
                image = generate_image(
                    full_prompt, reference_image=reference_image, seed=seed,
                    strength=chat.conditioning_strength,
                    num_inference_steps=chat.num_inference_steps,
                    model_key=chat.model_key,
                )
                comparison_image = generate_image(
                    full_prompt, seed=seed,
                    num_inference_steps=chat.num_inference_steps, model_key=chat.model_key,
                )
            else:
                image = generate_image(
                    full_prompt,
                    num_inference_steps=chat.num_inference_steps, model_key=chat.model_key,
                )
        else:
            image = generate_image(
                prompt,
                num_inference_steps=chat.num_inference_steps, model_key=chat.model_key,
            )

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

        serial.save(
            chat=chat, log_id=l_id, image=image_file,
            comparison_image=comparison_image_file,
        )
        chat_cache.sync_chat(chat)

        return Response(serial.data, status.HTTP_201_CREATED)
    return Response(serial.errors, status.HTTP_400_BAD_REQUEST)
