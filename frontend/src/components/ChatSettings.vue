<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { modelLabel } from "../models";
import type { ChatSettings } from "../types";

const props = defineProps<{
  settings: ChatSettings;
}>();

const open = ref(false);
const rootRef = ref<HTMLElement | null>(null);

function toggle() {
  open.value = !open.value;
}

function handleOutsideClick(event: MouseEvent) {
  if (open.value && rootRef.value && !rootRef.value.contains(event.target as Node)) {
    open.value = false;
  }
}

function handleEscape(event: KeyboardEvent) {
  if (event.key === "Escape") open.value = false;
}

onMounted(() => {
  document.addEventListener("mousedown", handleOutsideClick);
  document.addEventListener("keydown", handleEscape);
});

onBeforeUnmount(() => {
  document.removeEventListener("mousedown", handleOutsideClick);
  document.removeEventListener("keydown", handleEscape);
});
</script>

<template>
  <div ref="rootRef" class="chat-settings">
    <button
      class="chat-settings__trigger"
      type="button"
      title="Chat settings"
      aria-label="Chat settings"
      @click="toggle"
    >
      ⚙
    </button>

    <!-- Set once when the chat was created; view-only from here on. -->
    <div v-if="open" class="chat-settings__popup">
      <div class="chat-settings__field chat-settings__field--name">
        <span>Name</span>
        <span class="chat-settings__value">{{ props.settings.name || "Untitled" }}</span>
      </div>

      <div class="chat-settings__field">
        <span>Model</span>
        <span class="chat-settings__value">{{ modelLabel(props.settings.model_key) }}</span>
      </div>

      <label class="chat-settings__toggle">
        <input type="checkbox" :checked="props.settings.use_conditioning" disabled />
        Non-Conditioned Reference
      </label>

      <div class="chat-settings__field">
        <span>Strength</span>
        <span class="chat-settings__value">{{ props.settings.conditioning_strength }}</span>
      </div>

      <div class="chat-settings__field">
        <span>Steps</span>
        <span class="chat-settings__value">{{ props.settings.num_inference_steps }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-settings {
  position: relative;
  display: inline-flex;
}

.chat-settings__trigger {
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  color: var(--ink-muted);
  background: transparent;
  border: 1px solid var(--line);
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.15s ease, color 0.15s ease;
}

.chat-settings__trigger:hover {
  border-color: var(--accent);
  color: var(--accent);
}

.chat-settings__popup {
  position: absolute;
  top: calc(100% + 8px);
  left: 0;
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 200px;
  padding: 14px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  font-size: 13px;
  color: var(--ink-muted);
}

.chat-settings__toggle {
  display: flex;
  align-items: center;
  gap: 8px;
}

.chat-settings__toggle input {
  accent-color: var(--accent);
}

.chat-settings__field {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.chat-settings__value {
  color: var(--ink);
  font-weight: 500;
}

.chat-settings__field--name {
  padding-bottom: 8px;
  border-bottom: 1px solid var(--line);
}

.chat-settings__field--name .chat-settings__value {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 120px;
}
</style>
