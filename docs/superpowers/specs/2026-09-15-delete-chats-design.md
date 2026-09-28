# Design: Delete Chats (Sidebar)

## Purpose

Users are capped at `MAX_CHATS = 5` chats per account, and there is currently no
way to remove one. This adds a delete control per chat in the sidebar so users
can free up slots. Deleting a chat removes all of its log entries (prompts +
generated images, both DB rows and image files on disk).

This is distinct from the existing per-entry `×` delete (`delete_log`), which
removes a single prompt/image within a chat. This feature deletes the whole
chat.

## Bug found while scoping this

`create_chat` currently assigns new chat IDs as `chat-{account.chat.count() + 1}`.
Once chats can be deleted, this collides: e.g. with `chat-1, chat-2, chat-3`,
deleting `chat-2` drops the count to 2, so the next created chat computes
`chat-3` again — already taken, violating the `unique_per_account` constraint
on `(account, chat_id)` and crashing the request. This is fixed as part of
this change (see Data Model below).

Verified against the live demo account, which is already in a collision-prone
state: existing chats are `chat-2, chat-3, chat-4, chat-5, main` (5 total,
`chat-1` missing from earlier testing).

## Data model

Add a monotonic counter to `Account` so chat IDs are never recomputed from the
current count:

```python
class Account(models.Model):
    ...
    next_chat_n = models.PositiveIntegerField(default=1)
```

`create_chat` becomes:

```python
chat_id = f'chat-{account.next_chat_n}'
serial.save(account=account, chat_id=chat_id)
account.next_chat_n += 1
account.save()
```

**Migration**: a schema migration adds the field (`default=1`), followed by a
data migration that backfills `next_chat_n` correctly for any existing
accounts — for each account, scan its chats' `chat_id`s matching the
`chat-<N>` pattern, set `next_chat_n = max(N) + 1` (or `1` if none match).
This makes the demo account come out of migration at `next_chat_n = 6`
(continuing past `chat-5`), not colliding with any existing chat.

## Backend endpoint

Reuse the existing URL `accounts/<user_id>/chat/<chat_id>/`. It currently maps
only `POST` to `create_log`. Rename that view to `chat_detail` and allow both
methods (`@api_view(['POST', 'DELETE'])`), branching on `request.method`:

- **POST** (unchanged): existing `create_log` body, creates a new log entry
  with a generated image.
- **DELETE** (new): looks up the chat, deletes every log entry's image file
  off disk (`log.image.delete(save=False)` for each, mirroring `delete_log`),
  then `chat.delete()` (cascades the `ChatImages` rows via the existing FK
  `on_delete=CASCADE`). Returns `204 No Content`.

No new URL pattern is added. `delete_log` (single-entry delete) is untouched.

## Frontend (`chat.html`)

Each sidebar entry currently renders as a single `<button class="chat-btn">`
containing just the chat's name — clicking it selects the chat. A delete
control can't nest inside that button (invalid HTML: button-in-button), so
each entry is restructured to match the existing `.log` / `.log-delete`
pattern already used for log entries:

- Wrap each entry in a `div.chat-item` containing:
  - the existing clickable name element (still selects the chat on click)
  - a sibling `button.chat-delete` ("×"), styled like `.log-delete`
- Clicking the `×` calls `DELETE /api/accounts/demo/chat/<chat_id>/`, then
  `loadAccount()`.

No confirmation dialog — matches the existing per-log delete, which is also
immediate.

**Active-chat handling requires no new logic.** `loadAccount()` already resets
`activeChatId` to the first remaining chat, or `null` if none remain, whenever
the currently active chat is missing from the freshly fetched account. Since
deleting the active chat removes it from that list, the existing reselect
logic already produces "auto-switch to another chat" and correctly handles
the zero-chats case (empty state, only "+ New Chat" enabled).

## Edge cases

- **Deleting the last chat**: allowed. Sidebar becomes empty, `logs` area
  shows nothing, only "+ New Chat" is usable. No special-casing needed.
- **Deleting a non-active chat**: sidebar list refreshes; active chat
  selection is untouched (already in the fetched list, so no reselect fires).
- **`MAX_CHATS` cap**: unaffected — `create_chat`'s existing
  `account.chat.count() >= MAX_CHATS` check is independent of the new
  `next_chat_n` counter.

## Out of scope

- No confirmation prompts anywhere in this feature.
- No changes to `delete_log` or per-entry deletion.
- No change to `MAX_CHATS` value or enforcement logic.
