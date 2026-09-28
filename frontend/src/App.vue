<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { createChat, deleteChat, fetchAccount, getStoredAuth, logout, postPrompt } from "./api";
import { chatDisplayName } from "./chatLabel";
import type { Account, ChatSettings as ChatSettingsType } from "./types";
import Sidebar from "./components/Sidebar.vue";
import ChatLog from "./components/ChatLog.vue";
import PromptBar from "./components/PromptBar.vue";
import ChatSettings from "./components/ChatSettings.vue";
import Login from "./components/Login.vue";

const MAX_CHATS = 5;
const MAX_CHAT_HIST = 5;

// Matches api_data/auth.py's SESSION_TIMEOUT - this is a client-side
// safeguard that proactively shows the login screen without waiting for a
// request to fail first; the backend enforces the real timeout regardless.
const INACTIVITY_LIMIT_MS = 5 * 60 * 1000;
const ACTIVITY_EVENTS = ["mousemove", "mousedown", "keydown", "scroll", "touchstart"] as const;

type Status = "loading" | "ready" | "error";

const loggedIn = ref(Boolean(getStoredAuth()));
const account = ref<Account | null>(null);
const status = ref<Status>("loading");
const error = ref("");
const activeChatId = ref<string | null>(null);
const generating = ref(false);

const activeChat = computed(() =>
  account.value?.chat.find((chat) => chat.chat_id === activeChatId.value) ?? null
);

const atChatLimit = computed(() => (account.value?.chat.length ?? 0) >= MAX_CHATS);
const atLogLimit = computed(() => (activeChat.value?.chat_log.length ?? 0) >= MAX_CHAT_HIST);

async function load() {
  status.value = "loading";
  error.value = "";
  try {
    account.value = await fetchAccount();
    const stillExists = account.value.chat.some((c) => c.chat_id === activeChatId.value);
    if (!activeChatId.value || !stillExists) {
      activeChatId.value = account.value.chat[0]?.chat_id ?? null;
    }
    status.value = "ready";
  } catch (err) {
    if (!getStoredAuth()) {
      // The token was rejected (401) and cleared mid-request; drop back to login.
      stopInactivityTracking();
      loggedIn.value = false;
      return;
    }
    error.value = err instanceof Error ? err.message : "Something went wrong.";
    status.value = "error";
  }
}

let inactivityTimer: ReturnType<typeof setTimeout> | null = null;

function resetInactivityTimer() {
  if (inactivityTimer) clearTimeout(inactivityTimer);
  inactivityTimer = setTimeout(handleLogout, INACTIVITY_LIMIT_MS);
}

function startInactivityTracking() {
  ACTIVITY_EVENTS.forEach((evt) => window.addEventListener(evt, resetInactivityTimer));
  resetInactivityTimer();
}

function stopInactivityTracking() {
  ACTIVITY_EVENTS.forEach((evt) => window.removeEventListener(evt, resetInactivityTimer));
  if (inactivityTimer) {
    clearTimeout(inactivityTimer);
    inactivityTimer = null;
  }
}

function handleLoginSuccess() {
  loggedIn.value = true;
  startInactivityTracking();
  load();
}

async function handleLogout() {
  stopInactivityTracking();
  await logout();
  loggedIn.value = false;
  account.value = null;
  activeChatId.value = null;
}

async function handleCreateChat(settings: ChatSettingsType) {
  try {
    const chat = await createChat(settings);
    await load();
    activeChatId.value = chat.chat_id;
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not create chat.";
  }
}

async function handleDeleteChat(chatId: string) {
  try {
    await deleteChat(chatId);
    if (activeChatId.value === chatId) {
      activeChatId.value = null;
    }
    await load();
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not delete chat.";
  }
}

async function handleSubmitPrompt(text: string) {
  if (!activeChatId.value) return;
  generating.value = true;
  error.value = "";
  try {
    await postPrompt(activeChatId.value, text);
    await load();
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not generate that image.";
  } finally {
    generating.value = false;
  }
}

onMounted(() => {
  if (loggedIn.value) {
    startInactivityTracking();
    load();
  }
});

onBeforeUnmount(stopInactivityTracking);
</script>

<template>
  <Login v-if="!loggedIn" @success="handleLoginSuccess" />

  <div v-else class="app">
    <Sidebar
      v-if="account"
      :chats="account.chat"
      :active-chat-id="activeChatId"
      :at-chat-limit="atChatLimit"
      @select="(id) => (activeChatId = id)"
      @create="handleCreateChat"
      @remove="handleDeleteChat"
      @logout="handleLogout"
    />

    <main class="main">
      <p v-if="status === 'loading'" class="note">Loading…</p>

      <div v-else-if="status === 'error'" class="note note--error">
        <p>Can't reach the server. {{ error }}</p>
        <button class="retry" type="button" @click="load">Try again</button>
      </div>

      <template v-else-if="account">
        <p v-if="error" class="note note--error">{{ error }}</p>

        <div v-if="activeChat" class="chat-panel">
          <div class="chat-panel__header">
            <h1 class="chat-panel__title">{{ chatDisplayName(activeChat) }}</h1>
            <ChatSettings :settings="activeChat" />
          </div>
          <ChatLog :logs="activeChat.chat_log" />
          <PromptBar
            :disabled="atLogLimit"
            :busy="generating"
            :limit-reached="atLogLimit"
            @submit="handleSubmitPrompt"
          />
        </div>
        <p v-else class="note">Start a new chat to begin.</p>
      </template>
    </main>
  </div>
</template>

<style scoped>
.app {
  display: flex;
  height: 100vh;
}

.main {
  flex: 1;
  overflow-y: auto;
  display: flex;
  justify-content: center;
  padding: 32px 24px;
}

.chat-panel {
  width: 100%;
  max-width: 640px;
  display: flex;
  flex-direction: column;
}

.chat-panel__header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
}

.chat-panel__title {
  margin: 0;
  font-family: var(--font-display);
  font-size: 22px;
  font-weight: 500;
  color: var(--ink);
}

.note {
  color: var(--ink-muted);
  font-size: 14px;
}

.note--error {
  color: var(--error);
}

.retry {
  margin-top: 10px;
  padding: 8px 14px;
  font-size: 13px;
  font-family: inherit;
  color: var(--ink);
  background: transparent;
  border: 1px solid var(--line);
  border-radius: 8px;
  cursor: pointer;
}
</style>
