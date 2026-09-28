import io
import os
import tempfile
import time
from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.core.files.base import ContentFile
from django.utils import timezone
from PIL import Image
from rest_framework.test import APIClient

from .chat_id_utils import next_chat_n_for
from .models import Account, Chat, ChatImages, TokenActivity


class NextChatNForTests(TestCase):
    def test_no_existing_chats_starts_at_one(self):
        self.assertEqual(next_chat_n_for([]), 1)

    def test_ignores_non_matching_chat_ids(self):
        self.assertEqual(next_chat_n_for(['main']), 1)

    def test_returns_max_plus_one(self):
        self.assertEqual(next_chat_n_for(['chat-2', 'chat-3', 'chat-5', 'main']), 6)

    def test_handles_gap_from_deletion(self):
        self.assertEqual(next_chat_n_for(['chat-1', 'chat-3']), 4)


class AuthTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.account = Account.objects.create(user_id='tester', name='Tester')
        self.user = User.objects.create_user('tester', password='hunter2')

    def test_login_returns_token_for_correct_password(self):
        response = self.client.post(
            '/api/login/', {'username': 'tester', 'password': 'hunter2'}, format='json',
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('token', response.data)

    def test_login_rejects_wrong_password(self):
        response = self.client.post(
            '/api/login/', {'username': 'tester', 'password': 'wrong'}, format='json',
        )

        self.assertEqual(response.status_code, 400)

    def test_unauthenticated_request_is_rejected(self):
        response = self.client.get(f'/api/accounts/{self.account.user_id}/')

        self.assertEqual(response.status_code, 401)

    def test_cannot_access_another_users_account(self):
        User.objects.create_user('other', password='hunter2')
        self.client.force_authenticate(user=User.objects.get(username='other'))

        response = self.client.get(f'/api/accounts/{self.account.user_id}/')

        self.assertEqual(response.status_code, 403)

    def test_accounts_list_only_returns_own_account(self):
        other_account = Account.objects.create(user_id='other', name='Other')
        self.client.force_authenticate(user=User.objects.get_or_create(username='tester')[0])

        response = self.client.get('/api/accounts/')

        self.assertEqual(response.status_code, 200)
        returned_ids = {a['user_id'] for a in response.data}
        self.assertEqual(returned_ids, {'tester'})
        self.assertNotIn(other_account.user_id, returned_ids)

    def test_missing_account_is_404_not_500(self):
        # A login with no matching Account (e.g. a Django user created
        # outside signup, like a manage.py-created superuser) must get a
        # clean 404, not an unhandled DoesNotExist -> 500.
        User.objects.create_user('ghost', password='hunter2')
        self.client.force_authenticate(user=User.objects.get(username='ghost'))

        response = self.client.get('/api/accounts/ghost/')

        self.assertEqual(response.status_code, 404)


class SessionExpiryTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.account = Account.objects.create(user_id='tester', name='Tester')
        self.user = User.objects.create_user('tester', password='hunter2')
        response = self.client.post(
            '/api/login/', {'username': 'tester', 'password': 'hunter2'}, format='json',
        )
        self.token = response.data['token']
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token}')
        # One authenticated request so a TokenActivity row exists to
        # backdate below - it's only created lazily, on first use.
        self.client.get(f'/api/accounts/{self.account.user_id}/')

    def test_token_still_works_within_the_timeout(self):
        response = self.client.get(f'/api/accounts/{self.account.user_id}/')

        self.assertEqual(response.status_code, 200)

    def test_token_rejected_after_5_minutes_of_inactivity(self):
        TokenActivity.objects.filter(token_key=self.token).update(
            last_seen=timezone.now() - timedelta(minutes=5, seconds=1),
        )

        response = self.client.get(f'/api/accounts/{self.account.user_id}/')

        self.assertEqual(response.status_code, 401)

    def test_activity_within_the_window_keeps_the_session_alive(self):
        # Last request was 4 minutes ago - still inside the 5 minute window.
        TokenActivity.objects.filter(token_key=self.token).update(
            last_seen=timezone.now() - timedelta(minutes=4),
        )

        response = self.client.get(f'/api/accounts/{self.account.user_id}/')

        self.assertEqual(response.status_code, 200)

    def test_request_resets_the_inactivity_clock(self):
        self.client.get(f'/api/accounts/{self.account.user_id}/')
        activity = TokenActivity.objects.get(token_key=self.token)
        self.assertLess(timezone.now() - activity.last_seen, timedelta(seconds=5))

    def test_logout_deletes_the_token(self):
        response = self.client.post('/api/logout/')
        self.assertEqual(response.status_code, 204)

        response = self.client.get(f'/api/accounts/{self.account.user_id}/')
        self.assertEqual(response.status_code, 401)


class SignupTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_signup_creates_user_and_account(self):
        response = self.client.post(
            '/api/signup/', {'username': 'newperson', 'password': 'hunter2'}, format='json',
        )

        self.assertEqual(response.status_code, 201)
        self.assertIn('token', response.data)
        self.assertTrue(User.objects.filter(username='newperson').exists())
        self.assertTrue(Account.objects.filter(user_id='newperson').exists())

    def test_signup_token_immediately_works(self):
        response = self.client.post(
            '/api/signup/', {'username': 'newperson', 'password': 'hunter2'}, format='json',
        )
        token = response.data['token']

        response = self.client.get(
            '/api/accounts/newperson/', HTTP_AUTHORIZATION=f'Token {token}',
        )

        self.assertEqual(response.status_code, 200)

    def test_signup_rejects_duplicate_username(self):
        User.objects.create_user('taken', password='hunter2')

        response = self.client.post(
            '/api/signup/', {'username': 'taken', 'password': 'hunter2'}, format='json',
        )

        self.assertEqual(response.status_code, 400)

    def test_signup_requires_username_and_password(self):
        response = self.client.post('/api/signup/', {'username': 'onlyusername'}, format='json')

        self.assertEqual(response.status_code, 400)
        self.assertFalse(User.objects.filter(username='onlyusername').exists())


class CreateChatTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.account = Account.objects.create(user_id='tester', name='Tester')
        self.client.force_authenticate(user=User.objects.create_user('tester'))

        # create_chat kicks off a background model download; stub it out so
        # tests never spawn a real download thread.
        ensure_ready_patcher = patch('api_data.model_downloads.ensure_ready')
        self.mock_ensure_ready = ensure_ready_patcher.start()
        self.addCleanup(ensure_ready_patcher.stop)

    def test_first_chat_gets_chat_1(self):
        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/', {}, format='json'
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data['chat_id'], 'chat-1')

    def test_each_account_has_its_own_chat_id_sequence(self):
        # This account already has chat-1, chat-2 from other tests in this
        # run's DB state; a *different* signed-in user must still start
        # fresh at chat-1, not continue this account's counter.
        Chat.objects.create(account=self.account, chat_id='chat-1')
        Chat.objects.create(account=self.account, chat_id='chat-2')
        self.account.next_chat_n = 3
        self.account.save(update_fields=['next_chat_n'])

        other_account = Account.objects.create(user_id='other', name='Other')
        other_client = APIClient()
        other_client.force_authenticate(user=User.objects.create_user('other'))

        response = other_client.post(
            f'/api/accounts/{other_account.user_id}/new_chat/', {}, format='json'
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data['chat_id'], 'chat-1')

    def test_chat_id_does_not_collide_after_deletion(self):
        # Simulates: chat-1, chat-2, chat-3 existed; chat-2 was deleted.
        # count() is now 2, but the next real chat-N in use is chat-3, so
        # the counter (already advanced to 4 by prior creates) must win.
        self.account.next_chat_n = 4
        self.account.save(update_fields=['next_chat_n'])
        Chat.objects.create(account=self.account, chat_id='chat-1')
        Chat.objects.create(account=self.account, chat_id='chat-3')

        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/', {}, format='json'
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data['chat_id'], 'chat-4')

    def test_max_chats_still_enforced(self):
        for i in range(1, 6):
            Chat.objects.create(account=self.account, chat_id=f'chat-{i}')
        self.account.next_chat_n = 6
        self.account.save(update_fields=['next_chat_n'])

        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/', {}, format='json'
        )

        self.assertEqual(response.status_code, 400)

    @patch('api_data.chat_cache.sync_chat')
    def test_syncs_new_chat_to_redis(self, mock_sync_chat):
        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/', {}, format='json'
        )

        self.assertEqual(response.status_code, 201)
        mock_sync_chat.assert_called_once()
        (synced_chat,) = mock_sync_chat.call_args.args
        self.assertEqual(synced_chat.chat_id, 'chat-1')

    def test_accepts_config_at_creation(self):
        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/',
            {
                'name': 'Weekend trip', 'use_conditioning': False,
                'conditioning_strength': 0.3, 'num_inference_steps': 4,
            },
            format='json',
        )

        self.assertEqual(response.status_code, 201)
        chat = Chat.objects.get(chat_id=response.data['chat_id'])
        self.assertEqual(chat.name, 'Weekend trip')
        self.assertFalse(chat.use_conditioning)
        self.assertEqual(chat.conditioning_strength, 0.3)
        self.assertEqual(chat.num_inference_steps, 4)

    def test_defaults_to_sd_turbo(self):
        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/', {}, format='json'
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data['model_key'], 'sd-turbo')

    def test_accepts_a_different_model_at_creation(self):
        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/',
            {'model_key': 'realvisxl-v5'}, format='json',
        )

        self.assertEqual(response.status_code, 201)
        chat = Chat.objects.get(chat_id=response.data['chat_id'])
        self.assertEqual(chat.model_key, 'realvisxl-v5')

    def test_warms_the_chosen_model_at_creation(self):
        self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/',
            {'model_key': 'realvisxl-v5'}, format='json',
        )

        self.mock_ensure_ready.assert_called_once_with('realvisxl-v5', background=True)

    def test_rejects_unknown_model_key(self):
        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/',
            {'model_key': 'not-a-real-model'}, format='json',
        )

        self.assertEqual(response.status_code, 400)

    def test_rejects_out_of_range_strength_at_creation(self):
        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/',
            {'conditioning_strength': 1.5}, format='json',
        )

        self.assertEqual(response.status_code, 400)

    def test_rejects_strength_steps_combo_yielding_zero_effective_steps(self):
        # int(steps * strength) is diffusers' effective denoising step count;
        # int(2 * 0.15) == 0, which crashes the pipeline on an empty latent.
        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/',
            {'conditioning_strength': 0.15, 'num_inference_steps': 2}, format='json',
        )

        self.assertEqual(response.status_code, 400)

    def test_allows_strength_steps_combo_yielding_one_effective_step(self):
        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/',
            {'conditioning_strength': 0.25, 'num_inference_steps': 4}, format='json',
        )

        self.assertEqual(response.status_code, 201)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ChatDetailTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.account = Account.objects.create(user_id='tester', name='Tester')
        self.chat = Chat.objects.create(account=self.account, chat_id='chat-1')
        self.client.force_authenticate(user=User.objects.create_user('tester'))

        # Every POST to chat_detail calls this to make sure the chat's model
        # is downloaded; stub it out so tests never trigger a real download.
        ensure_ready_patcher = patch('api_data.model_downloads.ensure_ready')
        self.mock_ensure_ready = ensure_ready_patcher.start()
        self.addCleanup(ensure_ready_patcher.stop)

    def test_config_cannot_be_changed_after_creation(self):
        response = self.client.patch(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/',
            {'name': 'Weekend trip'}, format='json',
        )

        # PATCH is no longer a supported method on this endpoint at all -
        # config is set once at creation (see CreateChatTests) and locked.
        self.assertEqual(response.status_code, 405)

    @patch('api_data.chat_cache.delete_chat')
    def test_delete_removes_chat_from_redis(self, mock_delete_chat):
        response = self.client.delete(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/'
        )

        self.assertEqual(response.status_code, 204)
        mock_delete_chat.assert_called_once_with(self.chat.chat_id)

    def test_delete_removes_chat_and_image_files(self):
        image = Image.new('RGB', (4, 4))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        log = ChatImages.objects.create(
            chat=self.chat, log_id=1, text='a cat',
            image=ContentFile(buffer.getvalue(), name='chat-1_1.png'),
        )
        image_path = log.image.path
        self.assertTrue(os.path.exists(image_path))

        response = self.client.delete(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/'
        )

        self.assertEqual(response.status_code, 204)
        self.assertFalse(Chat.objects.filter(pk=self.chat.pk).exists())
        self.assertFalse(os.path.exists(image_path))

    def test_delete_removes_comparison_image_file(self):
        image = Image.new('RGB', (4, 4))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        comparison_buffer = io.BytesIO()
        Image.new('RGB', (4, 4)).save(comparison_buffer, format='PNG')

        log = ChatImages.objects.create(
            chat=self.chat, log_id=1, text='a cat',
            image=ContentFile(buffer.getvalue(), name='chat-1_1.png'),
            comparison_image=ContentFile(
                comparison_buffer.getvalue(), name='chat-1_1_comparison.png'
            ),
        )
        comparison_path = log.comparison_image.path
        self.assertTrue(os.path.exists(comparison_path))

        response = self.client.delete(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/'
        )

        self.assertEqual(response.status_code, 204)
        self.assertFalse(os.path.exists(comparison_path))

    @patch('api_data.views.generate_image')
    def test_post_still_creates_log(self, mock_generate_image):
        mock_generate_image.return_value = Image.new('RGB', (4, 4))

        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/',
            {'text': 'a dog'}, format='json',
        )

        self.assertEqual(response.status_code, 201)
        self.assertTrue(
            ChatImages.objects.filter(chat=self.chat, text='a dog').exists()
        )

    @patch('api_data.views.generate_image')
    def test_post_waits_for_chat_model_before_generating(self, mock_generate_image):
        mock_generate_image.return_value = Image.new('RGB', (4, 4))

        self.client.post(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/',
            {'text': 'a dog'}, format='json',
        )

        self.mock_ensure_ready.assert_called_once_with(self.chat.model_key)

    @patch('api_data.views.generate_image')
    def test_post_rejects_once_log_limit_reached(self, mock_generate_image):
        mock_generate_image.return_value = Image.new('RGB', (4, 4))
        for i in range(1, 6):
            ChatImages.objects.create(chat=self.chat, log_id=i, text=f'prompt {i}')

        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/',
            {'text': 'one too many'}, format='json',
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.chat.chat_log.count(), 5)

    @patch('api_data.chat_cache.sync_chat')
    @patch('api_data.views.generate_image')
    def test_post_syncs_to_redis(self, mock_generate_image, mock_sync_chat):
        mock_generate_image.return_value = Image.new('RGB', (4, 4))

        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/',
            {'text': 'a dog'}, format='json',
        )

        self.assertEqual(response.status_code, 201)
        mock_sync_chat.assert_called_once()
        (synced_chat,) = mock_sync_chat.call_args.args
        self.assertEqual(synced_chat.chat_id, self.chat.chat_id)

    @patch('api_data.views.generate_image')
    def test_post_uses_no_reference_for_first_log(self, mock_generate_image):
        mock_generate_image.return_value = Image.new('RGB', (4, 4))

        self.client.post(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/',
            {'text': 'a dog'}, format='json',
        )

        mock_generate_image.assert_called_once_with(
            'a dog', num_inference_steps=self.chat.num_inference_steps,
            model_key=self.chat.model_key,
        )

    @patch('api_data.views.generate_image')
    def test_post_leaves_comparison_image_empty_for_first_log(self, mock_generate_image):
        mock_generate_image.return_value = Image.new('RGB', (4, 4))

        self.client.post(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/',
            {'text': 'a dog'}, format='json',
        )

        log = ChatImages.objects.get(chat=self.chat, log_id=1)
        self.assertFalse(log.comparison_image)

    @patch('api_data.views.generate_image')
    def test_post_chains_prompts_and_uses_latest_image_as_reference(self, mock_generate_image):
        primary_result = Image.new('RGB', (4, 4), color='green')
        comparison_result = Image.new('RGB', (4, 4), color='yellow')
        mock_generate_image.side_effect = [primary_result, comparison_result]

        first_image = Image.new('RGB', (4, 4), color='red')
        buffer1 = io.BytesIO()
        first_image.save(buffer1, format='PNG')
        ChatImages.objects.create(
            chat=self.chat, log_id=1, text='a fox',
            image=ContentFile(buffer1.getvalue(), name='chat-1_1.png'),
        )

        second_image = Image.new('RGB', (4, 4), color='blue')
        buffer2 = io.BytesIO()
        second_image.save(buffer2, format='PNG')
        log2 = ChatImages.objects.create(
            chat=self.chat, log_id=2, text='now make it snow',
            image=ContentFile(buffer2.getvalue(), name='chat-1_2.png'),
        )

        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/',
            {'text': 'add northern lights'}, format='json',
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(mock_generate_image.call_count, 2)

        primary_call, comparison_call = mock_generate_image.call_args_list
        primary_args, primary_kwargs = primary_call
        comparison_args, comparison_kwargs = comparison_call

        self.assertEqual(primary_args[0], 'a fox, now make it snow, add northern lights')
        self.assertEqual(
            primary_kwargs['reference_image'].convert('RGB').tobytes(),
            Image.open(log2.image.path).convert('RGB').tobytes(),
        )

        self.assertEqual(comparison_args[0], 'a fox, now make it snow, add northern lights')
        self.assertNotIn('reference_image', comparison_kwargs)

        self.assertIsNotNone(primary_kwargs.get('seed'))
        self.assertEqual(primary_kwargs['seed'], comparison_kwargs['seed'])

        new_log = ChatImages.objects.get(chat=self.chat, log_id=3)
        self.assertTrue(new_log.image)
        self.assertTrue(new_log.comparison_image)

    @patch('api_data.views.generate_image')
    def test_post_uses_chat_conditioning_settings(self, mock_generate_image):
        self.chat.conditioning_strength = 0.4
        self.chat.num_inference_steps = 2
        self.chat.save(update_fields=['conditioning_strength', 'num_inference_steps'])

        mock_generate_image.return_value = Image.new('RGB', (4, 4))
        first_image = Image.new('RGB', (4, 4))
        buffer = io.BytesIO()
        first_image.save(buffer, format='PNG')
        ChatImages.objects.create(
            chat=self.chat, log_id=1, text='a fox',
            image=ContentFile(buffer.getvalue(), name='chat-1_1.png'),
        )

        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/',
            {'text': 'add snow'}, format='json',
        )

        self.assertEqual(response.status_code, 201)
        primary_kwargs = mock_generate_image.call_args_list[0].kwargs
        self.assertEqual(primary_kwargs['strength'], 0.4)
        self.assertEqual(primary_kwargs['num_inference_steps'], 2)

    @patch('api_data.views.generate_image')
    def test_post_uses_chat_model_key(self, mock_generate_image):
        self.chat.model_key = 'realvisxl-v5'
        self.chat.save(update_fields=['model_key'])

        mock_generate_image.return_value = Image.new('RGB', (4, 4))
        first_image = Image.new('RGB', (4, 4))
        buffer = io.BytesIO()
        first_image.save(buffer, format='PNG')
        ChatImages.objects.create(
            chat=self.chat, log_id=1, text='a fox',
            image=ContentFile(buffer.getvalue(), name='chat-1_1.png'),
        )

        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/',
            {'text': 'add snow'}, format='json',
        )

        self.assertEqual(response.status_code, 201)
        primary_kwargs, comparison_kwargs = (
            call.kwargs for call in mock_generate_image.call_args_list
        )
        self.assertEqual(primary_kwargs['model_key'], 'realvisxl-v5')
        self.assertEqual(comparison_kwargs['model_key'], 'realvisxl-v5')

    @patch('api_data.views.generate_image')
    def test_post_skips_reference_when_conditioning_disabled(self, mock_generate_image):
        self.chat.use_conditioning = False
        self.chat.save(update_fields=['use_conditioning'])

        mock_generate_image.return_value = Image.new('RGB', (4, 4))
        first_image = Image.new('RGB', (4, 4))
        buffer = io.BytesIO()
        first_image.save(buffer, format='PNG')
        ChatImages.objects.create(
            chat=self.chat, log_id=1, text='a fox',
            image=ContentFile(buffer.getvalue(), name='chat-1_1.png'),
        )

        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/chat/{self.chat.chat_id}/',
            {'text': 'add snow'}, format='json',
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(mock_generate_image.call_count, 1)
        args, kwargs = mock_generate_image.call_args_list[0]
        self.assertEqual(args[0], 'a fox, add snow')
        self.assertNotIn('reference_image', kwargs)
        new_log = ChatImages.objects.get(chat=self.chat, log_id=2)
        self.assertFalse(new_log.comparison_image)


class ModelEndpointTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        Account.objects.create(user_id='tester', name='Tester')
        self.client.force_authenticate(user=User.objects.create_user('tester'))

    def test_list_models_includes_all_four_choices(self):
        response = self.client.get('/api/models/')

        self.assertEqual(response.status_code, 200)
        keys = {m['key'] for m in response.data}
        self.assertEqual(
            keys, {'sd-turbo', 'sdxl-turbo', 'flux2-klein', 'realvisxl-v5'},
        )
        for entry in response.data:
            self.assertIn('label', entry)
            self.assertIn('status', entry)
            self.assertIn('downloaded_bytes', entry)
            self.assertIn('total_bytes', entry)

    @patch('api_data.views.model_downloads.ensure_ready')
    def test_warm_model_triggers_background_download(self, mock_ensure_ready):
        response = self.client.post('/api/models/sdxl-turbo/warm/')

        self.assertEqual(response.status_code, 204)
        mock_ensure_ready.assert_called_once_with('sdxl-turbo', background=True)

    def test_warm_model_rejects_unknown_key(self):
        response = self.client.post('/api/models/not-a-real-model/warm/')

        self.assertEqual(response.status_code, 404)


class ModelDownloadsTests(TestCase):
    """Exercises model_downloads.py's own state machine directly, mocking
    only the actual pipeline load (never hits the network)."""

    def setUp(self):
        from api_data import model_downloads
        self.model_downloads = model_downloads
        # Each test gets a clean slate regardless of execution order/what
        # other tests (or the app's own startup hook) already touched.
        with model_downloads._lock:
            for key in model_downloads._status:
                model_downloads._status[key] = {
                    'status': 'not_downloaded', 'downloaded_bytes': 0, 'total_bytes': 0,
                }
                model_downloads._done_events[key].clear()

    @patch('api_data.model_downloads.image_model.get_text2img_pipeline')
    def test_ensure_ready_blocking_marks_model_ready(self, mock_get_pipeline):
        self.model_downloads.ensure_ready('sd-turbo')

        self.assertEqual(self.model_downloads.get_statuses()['sd-turbo']['status'], 'ready')
        mock_get_pipeline.assert_called_once_with('stabilityai/sd-turbo')

    @patch('api_data.model_downloads.image_model.get_text2img_pipeline')
    def test_ensure_ready_background_returns_immediately(self, mock_get_pipeline):
        mock_get_pipeline.side_effect = lambda repo_id: time.sleep(0.05)

        self.model_downloads.ensure_ready('sd-turbo', background=True)

        # Returned before the (slow) pipeline load finished.
        self.assertIn(
            self.model_downloads.get_statuses()['sd-turbo']['status'], ('queued', 'downloading'),
        )
        self.model_downloads._done_events['sd-turbo'].wait(timeout=2)
        self.assertEqual(self.model_downloads.get_statuses()['sd-turbo']['status'], 'ready')

    @patch('api_data.model_downloads.image_model.get_text2img_pipeline')
    def test_ensure_ready_is_a_noop_once_ready(self, mock_get_pipeline):
        self.model_downloads.ensure_ready('sd-turbo')
        self.model_downloads.ensure_ready('sd-turbo')

        mock_get_pipeline.assert_called_once()

    @patch('api_data.model_downloads.image_model.get_text2img_pipeline')
    def test_ensure_ready_records_error_on_failure(self, mock_get_pipeline):
        # The download itself always runs on the worker thread, even for a
        # blocking (background=False) caller - it can only observe the
        # failure via `_status`, not as a raised exception here.
        mock_get_pipeline.side_effect = RuntimeError('network is down')

        self.model_downloads.ensure_ready('sd-turbo')

        self.assertEqual(self.model_downloads.get_statuses()['sd-turbo']['status'], 'error')

    def test_unknown_model_key_raises(self):
        with self.assertRaises(KeyError):
            self.model_downloads.ensure_ready('not-a-real-model')
