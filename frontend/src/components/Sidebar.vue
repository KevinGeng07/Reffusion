<script setup lang="ts">
import { getStoredAuth } from "../api";
import { chatDisplayName } from "../chatLabel";
import type { Chat, ChatSettings } from "../types";
import NewChatPopup from "./NewChatPopup.vue";

const props = defineProps<{
  chats: Chat[];
  activeChatId: string | null;
  atChatLimit: boolean;
}>();

const emit = defineEmits<{
  select: [chatId: string];
  create: [settings: ChatSettings];
  remove: [chatId: string];
  logout: [];
}>();

const username = getStoredAuth()?.username ?? "";

function remove(event: MouseEvent, chatId: string) {
  event.stopPropagation();
  emit("remove", chatId);
}
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar__head">
      <span class="sidebar__title">Reffusion</span>
      <NewChatPopup :disabled="props.atChatLimit" @create="(settings) => emit('create', settings)" />
    </div>

    <nav class="chat-list">
      <div
        v-for="chat in props.chats"
        :key="chat.chat_id"
        class="chat-item"
        :class="{ 'chat-item--active': chat.chat_id === props.activeChatId }"
        role="button"
        tabindex="0"
        @click="emit('select', chat.chat_id)"
        @keydown.enter="emit('select', chat.chat_id)"
      >
        <span class="chat-item__label">{{ chatDisplayName(chat) }}</span>
        <span class="chat-item__remove" role="button" @click="remove($event, chat.chat_id)">
          &times;
        </span>
      </div>
    </nav>

    <div class="sidebar__footer">
      <span class="sidebar__username">{{ username }}</span>
      <button class="sidebar__logout" type="button" @click="emit('logout')">Log out</button>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 20px;
  width: 220px;
  flex-shrink: 0;
  padding: 24px 16px;
  border-right: 1px solid var(--line);
  background: var(--panel);
}

.sidebar__head {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.sidebar__title {
  font-family: var(--font-display);
  font-size: 20px;
  letter-spacing: 0.01em;
  color: var(--ink);
}

.chat-list {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
  overflow-y: auto;
}

.sidebar__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
}

.sidebar__username {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
  color: var(--ink-faint);
}

.sidebar__logout {
  flex-shrink: 0;
  font-size: 12px;
  color: var(--ink-muted);
  background: transparent;
  border: none;
  cursor: pointer;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.sidebar__logout:hover {
  color: var(--accent);
}

.chat-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 9px 10px;
  border: none;
  border-radius: 8px;
  background: transparent;
  color: var(--ink-muted);
  font-size: 14px;
  text-align: left;
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
}

.chat-item:hover {
  background: var(--panel-hover);
  color: var(--ink);
}

.chat-item--active {
  background: var(--panel-hover);
  color: var(--ink);
  font-weight: 600;
}

.chat-item__label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chat-item__remove {
  flex-shrink: 0;
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  color: var(--ink-faint);
  font-size: 14px;
  line-height: 1;
}

.chat-item__remove:hover {
  background: var(--line);
  color: var(--ink);
}
</style>
