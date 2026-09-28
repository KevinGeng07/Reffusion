<script setup lang="ts">
import type { ChatImage } from "../types";
import ChatMessage from "./ChatMessage.vue";

const props = defineProps<{
  logs: ChatImage[];
}>();
</script>

<template>
  <ul v-if="props.logs.length" class="chat-log">
    <ChatMessage v-for="log in props.logs" :key="log.log_id" :log="log" />
  </ul>
  <p v-else class="chat-log__empty">Describe an image below to get started.</p>
</template>

<style scoped>
.chat-log {
  list-style: none;
  margin: 0;
  padding: 0;
}

.chat-log__empty {
  color: var(--ink-faint);
  font-size: 14px;
}
</style>
