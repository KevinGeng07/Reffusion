import re

CHAT_ID_PATTERN = re.compile(r'^chat-(\d+)$')


def next_chat_n_for(chat_ids):
    max_n = 0
    for chat_id in chat_ids:
        match = CHAT_ID_PATTERN.match(chat_id)
        if match:
            max_n = max(max_n, int(match.group(1)))
    return max_n + 1
