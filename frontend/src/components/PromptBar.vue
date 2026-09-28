<script setup lang="ts">
import { ref } from "vue";

const props = defineProps<{
  disabled: boolean;
  busy: boolean;
  limitReached: boolean;
}>();

const emit = defineEmits<{
  submit: [text: string];
}>();

const text = ref("");

function handleSubmit() {
  const value = text.value.trim();
  if (!value || props.disabled || props.busy) return;
  emit("submit", value);
  text.value = "";
}
</script>

<template>
  <form class="prompt-bar" @submit.prevent="handleSubmit">
    <input
      v-model="text"
      class="prompt-bar__input"
      type="text"
      maxlength="250"
      placeholder="Describe an image..."
      :disabled="props.disabled || props.busy"
      required
    />
    <button
      class="prompt-bar__submit"
      type="submit"
      :disabled="props.disabled || props.busy"
      :title="props.limitReached ? '5 message limit reached' : 'Generate next sample'"
    >
      {{ props.busy ? "Working…" : "Generate" }}
    </button>
    <p v-if="props.limitReached" class="prompt-bar__hint">
      This chat has reached its message limit.
    </p>
  </form>
</template>

<style scoped>
.prompt-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  padding: 16px 0 4px;
}

.prompt-bar__input {
  flex: 1;
  min-width: 200px;
  padding: 11px 14px;
  font-size: 14px;
  font-family: inherit;
  color: var(--ink);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  outline: none;
  transition: border-color 0.15s ease;
}

.prompt-bar__input:focus {
  border-color: var(--accent);
}

.prompt-bar__input:disabled {
  opacity: 0.5;
}

.prompt-bar__submit {
  padding: 11px 18px;
  font-size: 14px;
  font-family: inherit;
  color: var(--paper);
  background: var(--accent);
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: opacity 0.15s ease;
}

.prompt-bar__submit:hover:not(:disabled) {
  opacity: 0.88;
}

.prompt-bar__submit:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.prompt-bar__hint {
  width: 100%;
  margin: 0;
  font-size: 12px;
  color: var(--ink-faint);
}
</style>
