# Design: Per-Chat Log Limit + Remove Per-Image Delete

## Purpose

Two changes, bundled because the second directly affects how the first
behaves once a chat is capped:

1. Cap each chat at `MAX_CHAT_HIST = 3` log entries (prompt + generated
   image pairs). The constant already exists in `models.py` and
   `ChatImages.clean()` references it, but nothing enforces it today —
   identical situation to the `MAX_CHATS` bug already fixed for chat
   creation this session.
2. Remove the ability to delete a single log entry/image. The only deletion
   affordance left in the app is the whole-chat delete in the sidebar
   (shipped earlier this session). This means once a chat hits the 3-log
   cap, it stays capped for good — the only way to free it up is deleting
   the entire chat and starting a new one.

## Backend

**Log limit** — in `chat_detail`'s POST branch
(`reffusion_data/api_data/views.py`), before creating a new log entry:

```python
if chat.chat_log.count() >= MAX_CHAT_HIST:
    return Response(
        {'detail': f'{MAX_CHAT_HIST} logs reached.'}, status.HTTP_400_BAD_REQUEST
    )
```

Same shape as `create_chat`'s existing `MAX_CHATS` check.

**Remove per-log delete** — delete the `delete_log` view function entirely
from `views.py`, and remove its URL (`accounts/<user_id>/chat/<chat_id>/<log_id>/`)
from `urls.py`. It currently has no test coverage, so there's nothing to
remove on that front.

## Frontend

In `chat.html`:
- Add `const MAX_CHAT_HIST = 3;` alongside the existing `const MAX_CHATS = 5;`.
- In `renderLogs()`, disable the prompt `<input>` and `Generate` button when
  the active chat's `chat_log.length >= MAX_CHAT_HIST`. Mirrors how
  `+ New Chat` already disables at the 5-chat cap — no new status text or
  error messaging. Since individual logs can no longer be deleted, there is
  no path that re-enables the form for that same chat — it stays disabled
  until the user deletes the whole chat (or picks a different one).
- Remove the per-log `×` delete button, its `.log-delete` CSS rule, and the
  `deleteLog()` function — the log entry's markup becomes just the prompt
  text and image, no delete control.

## Out of scope

- No changes to `MAX_CHATS` (chat-count cap) or its enforcement.
- No changes to the whole-chat delete feature (sidebar `×`), which remains
  the only deletion path in the app.
- No new error messaging beyond the existing disabled-state pattern.
