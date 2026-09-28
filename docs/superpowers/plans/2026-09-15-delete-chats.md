# Delete Chats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users delete an entire chat (and all its log entries/images) from the sidebar, freeing up a slot under the 5-chat cap, while fixing the chat-ID collision bug that deletion would otherwise expose.

**Architecture:** A monotonic `next_chat_n` counter on `Account` replaces the current `count()+1` chat-ID scheme (which collides once chats can be deleted). The existing `create_log` view is renamed to `chat_detail` and extended to also handle `DELETE` on the same URL, removing the chat's log entries' image files from disk before cascading the DB delete. The sidebar's per-chat `<button>` gains a sibling delete control, matching the existing per-log-entry `×` pattern already in `chat.html`.

**Tech Stack:** Django 6.1, Django REST Framework, SQLite, vanilla JS/HTML (no build step), Pillow for test images.

## Global Constraints

- No confirmation dialogs anywhere in this feature (matches existing per-log delete, which is also immediate).
- `MAX_CHATS = 5` value and its enforcement in `create_chat` are unchanged.
- No new URL patterns — the chat-delete endpoint reuses the existing `accounts/<user_id>/chat/<chat_id>/` path, dispatched by HTTP method.
- `delete_log` (single log-entry delete) is untouched.
- Deleting a chat must remove its images from disk, not just the DB rows (`ChatImages.image.delete(save=False)` per entry, mirroring `delete_log`'s existing pattern) — Django's `on_delete=CASCADE` only removes DB rows, never files.
- Tests must not require the real `sd-turbo` model — any test that exercises `create_log`'s POST path must `unittest.mock.patch('api_data.views.generate_image')`.
- Tests that write image files must override `MEDIA_ROOT` to a temp directory and clean it up, so the real `reffusion_data/media/` is never touched by the test suite.

---

## File Structure

- **Create** `reffusion_data/api_data/chat_id_utils.py` — pure function computing the next available `chat-N` counter value from a list of existing `chat_id` strings. Extracted out of the data migration so it's unit-testable without any DB/migration machinery.
- **Create** `reffusion_data/api_data/migrations/0007_account_next_chat_n.py` — schema migration, adds `Account.next_chat_n`.
- **Create** `reffusion_data/api_data/migrations/0008_backfill_next_chat_n.py` — data migration, backfills `next_chat_n` for existing accounts using `chat_id_utils.next_chat_n_for`.
- **Modify** `reffusion_data/api_data/models.py` — add `next_chat_n` field to `Account`.
- **Modify** `reffusion_data/api_data/views.py` — `create_chat` uses/increments `next_chat_n`; `create_log` renamed to `chat_detail`, gains a `DELETE` branch.
- **Modify** `reffusion_data/api_data/urls.py` — point the existing chat-detail URL at `chat_detail` instead of `create_log`.
- **Modify** `reffusion_data/api_data/templates/api_data/chat.html` — sidebar entries restructured to add a per-chat delete control.
- **Modify** `reffusion_data/api_data/tests.py` — add all tests below (currently an empty stub).

---

### Task 1: Fix chat-ID collisions with a monotonic counter

**Files:**
- Create: `reffusion_data/api_data/chat_id_utils.py`
- Create: `reffusion_data/api_data/migrations/0007_account_next_chat_n.py`
- Create: `reffusion_data/api_data/migrations/0008_backfill_next_chat_n.py`
- Modify: `reffusion_data/api_data/models.py:8-13` (the `Account` class)
- Modify: `reffusion_data/api_data/views.py:45-60` (the `create_chat` view)
- Test: `reffusion_data/api_data/tests.py`

**Interfaces:**
- Produces: `chat_id_utils.next_chat_n_for(chat_ids: list[str]) -> int` — given existing chat IDs, returns the next unused `chat-N` counter value (`1` if no `chat-N`-pattern IDs exist).
- Produces: `Account.next_chat_n` (`PositiveIntegerField`, default `1`) — used by `create_chat` and by later tasks' test setup.

- [ ] **Step 1: Write the failing tests for `next_chat_n_for`**

Create `reffusion_data/api_data/tests.py` with:

```python
from django.test import TestCase

from .chat_id_utils import next_chat_n_for


class NextChatNForTests(TestCase):
    def test_no_existing_chats_starts_at_one(self):
        self.assertEqual(next_chat_n_for([]), 1)

    def test_ignores_non_matching_chat_ids(self):
        self.assertEqual(next_chat_n_for(['main']), 1)

    def test_returns_max_plus_one(self):
        self.assertEqual(next_chat_n_for(['chat-2', 'chat-3', 'chat-5', 'main']), 6)

    def test_handles_gap_from_deletion(self):
        self.assertEqual(next_chat_n_for(['chat-1', 'chat-3']), 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (from `reffusion_data/`): `../env/bin/python manage.py test api_data.tests.NextChatNForTests -v 2`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'api_data.chat_id_utils'`

- [ ] **Step 3: Implement `next_chat_n_for`**

Create `reffusion_data/api_data/chat_id_utils.py`:

```python
import re

CHAT_ID_PATTERN = re.compile(r'^chat-(\d+)$')


def next_chat_n_for(chat_ids):
    max_n = 0
    for chat_id in chat_ids:
        match = CHAT_ID_PATTERN.match(chat_id)
        if match:
            max_n = max(max_n, int(match.group(1)))
    return max_n + 1
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../env/bin/python manage.py test api_data.tests.NextChatNForTests -v 2`
Expected: `OK` (4 tests)

- [ ] **Step 5: Write the failing tests for `create_chat`'s new ID scheme**

Append to `reffusion_data/api_data/tests.py`:

```python
from rest_framework.test import APIClient

from .models import Account, Chat


class CreateChatTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.account = Account.objects.create(user_id='tester', name='Tester')

    def test_first_chat_gets_chat_1(self):
        response = self.client.post(
            f'/api/accounts/{self.account.user_id}/new_chat/', {}, format='json'
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
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `../env/bin/python manage.py test api_data.tests.CreateChatTests -v 2`
Expected: FAIL — no `next_chat_n` field yet (`AttributeError` or migration/field error), and `test_first_chat_gets_chat_1` fails because `create_chat` still uses `count()+1`.

- [ ] **Step 7: Add the field, migrations, and update `create_chat`**

In `reffusion_data/api_data/models.py`, add the field to `Account` (models.py:8-13):

```python
class Account(models.Model):
    name = models.CharField() # Extract from user account.
    user_id = models.CharField(max_length=32, unique=True)
    next_chat_n = models.PositiveIntegerField(default=1)

    def __str__(self):
        return self.user_id
```

Create `reffusion_data/api_data/migrations/0007_account_next_chat_n.py`:

```python
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api_data', '0006_seed_demo_account'),
    ]

    operations = [
        migrations.AddField(
            model_name='account',
            name='next_chat_n',
            field=models.PositiveIntegerField(default=1),
        ),
    ]
```

Create `reffusion_data/api_data/migrations/0008_backfill_next_chat_n.py`:

```python
from django.db import migrations

from api_data.chat_id_utils import next_chat_n_for


def backfill_next_chat_n(apps, schema_editor):
    Account = apps.get_model('api_data', 'Account')

    for account in Account.objects.all():
        chat_ids = list(account.chat.values_list('chat_id', flat=True))
        account.next_chat_n = next_chat_n_for(chat_ids)
        account.save(update_fields=['next_chat_n'])


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('api_data', '0007_account_next_chat_n'),
    ]

    operations = [
        migrations.RunPython(backfill_next_chat_n, noop_reverse),
    ]
```

In `reffusion_data/api_data/views.py`, update `create_chat` (views.py:45-60):

```python
@api_view(['POST'])
def create_chat(request, user_id):
    account = Account.objects.get(user_id=user_id)

    if account.chat.count() >= MAX_CHATS:
        return Response(
            {'detail': f'{MAX_CHATS} chats reached.'}, status.HTTP_400_BAD_REQUEST
        )

    serial = ChatSerial(data=request.data)

    if serial.is_valid():
        chat_id = f'chat-{account.next_chat_n}'
        serial.save(account=account, chat_id=chat_id)
        account.next_chat_n += 1
        account.save(update_fields=['next_chat_n'])
        return Response(serial.data, status.HTTP_201_CREATED)
    return Response(serial.errors, status.HTTP_400_BAD_REQUEST)
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `../env/bin/python manage.py test api_data.tests.NextChatNForTests api_data.tests.CreateChatTests -v 2`
Expected: `OK` (7 tests)

- [ ] **Step 9: Apply the migrations to the real dev database**

Run (from `reffusion_data/`): `../env/bin/python manage.py migrate`
Expected: `Applying api_data.0007_account_next_chat_n... OK` and `Applying api_data.0008_backfill_next_chat_n... OK`. Verify the demo account backfilled correctly:

Run: `../env/bin/python manage.py shell -c "from api_data.models import Account; print(Account.objects.get(user_id='demo').next_chat_n)"`
Expected: `6` (demo account's existing chats are `chat-2, chat-3, chat-4, chat-5, main` — max is 5, so next is 6)

- [ ] **Step 10: Commit**

```bash
git add reffusion_data/api_data/chat_id_utils.py reffusion_data/api_data/migrations/0007_account_next_chat_n.py reffusion_data/api_data/migrations/0008_backfill_next_chat_n.py reffusion_data/api_data/models.py reffusion_data/api_data/views.py reffusion_data/api_data/tests.py
git commit -m "fix: replace count()-based chat IDs with a monotonic counter"
```

(If this repo has no git history yet, run `git init` first and skip this step's `git add`/`git commit` split — just note in your session that history starts here.)

---

### Task 2: Add whole-chat DELETE to the chat-detail endpoint

**Files:**
- Modify: `reffusion_data/api_data/views.py:63-88` (rename `create_log` to `chat_detail`, add `DELETE` branch)
- Modify: `reffusion_data/api_data/urls.py:9` (point at `chat_detail`)
- Test: `reffusion_data/api_data/tests.py`

**Interfaces:**
- Consumes: `Account`, `Chat`, `ChatImages` models (unchanged); `generate_image(prompt: str) -> PIL.Image` from `api_data/image_model.py` (unchanged, must be mocked in tests).
- Produces: `chat_detail(request, user_id, chat_id)` view, bound to `DELETE /api/accounts/<user_id>/chat/<chat_id>/` (new) and `POST /api/accounts/<user_id>/chat/<chat_id>/` (same behavior as the old `create_log`).

- [ ] **Step 1: Write the failing tests**

Append to `reffusion_data/api_data/tests.py`:

```python
import io
import os
import tempfile
from unittest.mock import patch

from django.conf import settings
from django.core.files.base import ContentFile
from django.test import override_settings
from PIL import Image

from .models import ChatImages


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ChatDetailTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.account = Account.objects.create(user_id='tester', name='Tester')
        self.chat = Chat.objects.create(account=self.account, chat_id='chat-1')

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `../env/bin/python manage.py test api_data.tests.ChatDetailTests -v 2`
Expected: FAIL — `test_delete_removes_chat_and_image_files` gets a `405 Method Not Allowed` (DELETE isn't wired up yet); `test_post_still_creates_log` should already pass (it exercises unchanged behavior) — confirms it's a valid regression check going in.

- [ ] **Step 3: Rename `create_log` to `chat_detail` and add the DELETE branch**

In `reffusion_data/api_data/views.py`, replace the `create_log` function (views.py:63-88) with:

```python
@api_view(['POST', 'DELETE'])
def chat_detail(request, user_id, chat_id):
    account = Account.objects.get(user_id=user_id)
    chat = account.chat.get(chat_id=chat_id)

    if request.method == 'DELETE':
        for log in chat.chat_log.all():
            if log.image:
                log.image.delete(save=False)
        chat.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    serial = ChatImagesSerial(data=request.data)

    if serial.is_valid():
        last_log_id = chat.chat_log.order_by('log_id').last()

        if not last_log_id:
            l_id = 1
        else:
            l_id = last_log_id.log_id + 1

        prompt = serial.validated_data['text']
        image = generate_image(prompt)

        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_file = ContentFile(buffer.getvalue(), name=f'{chat.chat_id}_{l_id}.png')

        serial.save(chat=chat, log_id=l_id, image=image_file)

        return Response(serial.data, status.HTTP_201_CREATED)
    return Response(serial.errors, status.HTTP_400_BAD_REQUEST)
```

In `reffusion_data/api_data/urls.py`, update the chat-detail line (urls.py:9):

```python
    path('accounts/<str:user_id>/chat/<str:chat_id>/', chat_detail, name='chat_detail'),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../env/bin/python manage.py test api_data.tests -v 2`
Expected: `OK` (11 tests total across all three test classes so far)

- [ ] **Step 5: Manually verify against the running dev server**

If the dev server isn't already running: `cd reffusion_data && ../env/bin/python manage.py runserver` (background it).

```bash
curl -s -X DELETE http://localhost:8000/api/accounts/demo/chat/chat-2/ -w "\n%{http_code}\n"
curl -s http://localhost:8000/api/accounts/demo/ | python3 -c "import json,sys; print([c['chat_id'] for c in json.load(sys.stdin)['chat']])"
```

Expected: first command prints `204`; second command's list no longer contains `chat-2`, and `reffusion_data/media/gens/chat-2_*.png` files are gone from disk (`ls reffusion_data/media/gens/`).

- [ ] **Step 6: Commit**

```bash
git add reffusion_data/api_data/views.py reffusion_data/api_data/urls.py reffusion_data/api_data/tests.py
git commit -m "feat: add DELETE support to chat_detail for removing a whole chat"
```

---

### Task 3: Sidebar delete control

**Files:**
- Modify: `reffusion_data/api_data/templates/api_data/chat.html`

**Interfaces:**
- Consumes: `DELETE /api/accounts/<user_id>/chat/<chat_id>/` (from Task 2); existing `loadAccount()` (already resets `activeChatId` to the first remaining chat, or `null`, whenever the active chat is missing from the fetched account — no changes needed there).

There's no JS test runner in this project, so this task is verified manually in the browser (existing convention — `chat.html` has no other automated coverage either).

- [ ] **Step 1: Add `.chat-item` / `.chat-delete` styles**

In `reffusion_data/api_data/templates/api_data/chat.html`, in the `<style>` block, replace this line:

```css
    .chat-btn { display: block; width: 100%; padding: 8px; text-align: left; border: 1px solid #ddd; border-radius: 6px; background: white; cursor: pointer; box-sizing: border-box; }
```

with:

```css
    .chat-item { display: flex; align-items: stretch; gap: 4px; }
    .chat-btn { flex: 1; min-width: 0; padding: 8px; text-align: left; border: 1px solid #ddd; border-radius: 6px; background: white; cursor: pointer; box-sizing: border-box; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .chat-delete { flex: 0 0 auto; padding: 2px 8px; line-height: 1; border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer; }
```

- [ ] **Step 2: Restructure `renderSidebar()` and add `deleteChat()`**

Replace the `renderSidebar` function:

```javascript
    function renderSidebar() {
      chatListEl.innerHTML = '';
      account.chat.forEach((chat) => {
        const btn = document.createElement('button');
        btn.className = 'chat-btn' + (chat.chat_id === activeChatId ? ' active' : '');
        btn.textContent = chat.chat_id;
        btn.addEventListener('click', () => {
          activeChatId = chat.chat_id;
          render();
        });
        chatListEl.appendChild(btn);
      });
      newChatBtn.disabled = account.chat.length >= MAX_CHATS;
    }
```

with:

```javascript
    function renderSidebar() {
      chatListEl.innerHTML = '';
      account.chat.forEach((chat) => {
        const item = document.createElement('div');
        item.className = 'chat-item';

        const btn = document.createElement('button');
        btn.className = 'chat-btn' + (chat.chat_id === activeChatId ? ' active' : '');
        btn.textContent = chat.chat_id;
        btn.addEventListener('click', () => {
          activeChatId = chat.chat_id;
          render();
        });

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'chat-delete';
        deleteBtn.type = 'button';
        deleteBtn.title = 'Delete chat';
        deleteBtn.textContent = '×';
        deleteBtn.addEventListener('click', () => deleteChat(chat.chat_id));

        item.appendChild(btn);
        item.appendChild(deleteBtn);
        chatListEl.appendChild(item);
      });
      newChatBtn.disabled = account.chat.length >= MAX_CHATS;
    }

    async function deleteChat(chatId) {
      await fetch(`/api/accounts/${USER_ID}/chat/${chatId}/`, {
        method: 'DELETE',
      });
      await loadAccount();
    }
```

- [ ] **Step 3: Manually verify in the browser**

With the dev server running (`cd reffusion_data && ../env/bin/python manage.py runserver`), open `http://localhost:8000/`:

1. Confirm each sidebar entry now shows a chat-name button plus a small `×` next to it.
2. Click `×` on a non-active chat — confirm it disappears from the sidebar and the currently-open chat's log is untouched.
3. Click `×` on the currently-active chat — confirm the view auto-switches to another remaining chat's log.
4. Delete every chat down to zero — confirm the sidebar goes empty, `+ New Chat` is still enabled, and no JS console errors appear (check via `read_console_messages` or the browser devtools console).
5. Click `+ New Chat` after deleting some — confirm the new chat's ID doesn't collide with a still-existing one (this exercises Task 1's fix live).

- [ ] **Step 4: Commit**

```bash
git add reffusion_data/api_data/templates/api_data/chat.html
git commit -m "feat: add per-chat delete control to the sidebar"
```

---

## Self-Review Notes

- **Spec coverage:** Data model fix (Task 1), backend DELETE endpoint reusing the existing URL (Task 2), sidebar delete control with no confirmation and correct active/empty-state handling via existing `loadAccount()` (Task 3) — all sections of `docs/superpowers/specs/2026-09-15-delete-chats-design.md` are covered. "Out of scope" items (confirmation prompts, `delete_log` changes, `MAX_CHATS` changes) are correctly untouched by every task.
- **Placeholder scan:** No TBDs; every step has runnable code or an exact command with expected output.
- **Type/name consistency:** `next_chat_n_for` used identically in the migration and both test files; `chat_detail` name is consistent across `views.py`, `urls.py`, and its own tests; `deleteChat(chatId)` matches its call site in `renderSidebar`.
