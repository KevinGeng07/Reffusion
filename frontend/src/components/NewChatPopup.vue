<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { fetchModelStatuses, warmModel } from "../api";
import { MODEL_CHOICES } from "../models";
import type { ModelStatus } from "../models";
import type { ChatSettings } from "../types";

const props = defineProps<{
  disabled: boolean;
}>();

const emit = defineEmits<{
  create: [settings: ChatSettings];
}>();

const open = ref(false);
const rootRef = ref<HTMLElement | null>(null);

const name = ref("");
const useConditioning = ref(true);
const strength = ref(0.8);
const steps = ref(4);
const modelKey = ref(MODEL_CHOICES[0].key);

const modelStatuses = ref<ModelStatus[]>([]);
let pollHandle: ReturnType<typeof setInterval> | null = null;

const selectedStatus = computed(
  () => modelStatuses.value.find((m) => m.key === modelKey.value) ?? null
);

const downloadPercent = computed(() => {
  const s = selectedStatus.value;
  if (!s || !s.total_bytes) return 0;
  return Math.min(100, Math.round((s.downloaded_bytes / s.total_bytes) * 100));
});

const downloadLabel = computed(() => {
  const s = selectedStatus.value;
  if (!s) return "";
  if (s.status === "queued") return "Queued…";
  if (!s.total_bytes) return "Downloading…";
  const mb = (n: number) => (n / 1_000_000).toFixed(0);
  return `Downloading… ${mb(s.downloaded_bytes)} / ${mb(s.total_bytes)} MB`;
});

async function refreshStatuses() {
  try {
    modelStatuses.value = await fetchModelStatuses();
  } catch {
    // Transient network hiccup while polling; just try again next tick.
  }
}

function startPolling() {
  if (pollHandle) return;
  refreshStatuses();
  pollHandle = setInterval(refreshStatuses, 1000);
}

function stopPolling() {
  if (pollHandle) {
    clearInterval(pollHandle);
    pollHandle = null;
  }
}

function handleModelChange() {
  warmModel(modelKey.value).catch(() => {});
  refreshStatuses();
}

function reset() {
  name.value = "";
  useConditioning.value = true;
  strength.value = 0.8;
  steps.value = 4;
  modelKey.value = MODEL_CHOICES[0].key;
}

function toggle() {
  if (props.disabled) return;
  open.value = !open.value;
  if (open.value) {
    startPolling();
  } else {
    stopPolling();
  }
}

function handleOutsideClick(event: MouseEvent) {
  if (open.value && rootRef.value && !rootRef.value.contains(event.target as Node)) {
    open.value = false;
    stopPolling();
  }
}

function handleEscape(event: KeyboardEvent) {
  if (event.key === "Escape") {
    open.value = false;
    stopPolling();
  }
}

onMounted(() => {
  document.addEventListener("mousedown", handleOutsideClick);
  document.addEventListener("keydown", handleEscape);
});

onBeforeUnmount(() => {
  document.removeEventListener("mousedown", handleOutsideClick);
  document.removeEventListener("keydown", handleEscape);
  stopPolling();
});

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

// Mirrors the backend: int(steps * strength) is the number of denoising
// steps diffusers actually runs; at 0 it crashes instead of no-op-ing.
const effectiveSteps = computed(() => Math.floor(steps.value * strength.value));
const stepsError = computed(() =>
  useConditioning.value && effectiveSteps.value < 1
    ? `Strength × Steps must be at least 1 (currently ${effectiveSteps.value}).`
    : ""
);

function handleCreate() {
  if (stepsError.value) return;

  emit("create", {
    name: name.value.trim(),
    use_conditioning: useConditioning.value,
    conditioning_strength: clamp(strength.value, 0, 1),
    num_inference_steps: Math.round(clamp(steps.value, 1, 10)),
    model_key: modelKey.value,
  });
  open.value = false;
  stopPolling();
  reset();
}
</script>

<template>
  <div ref="rootRef" class="new-chat">
    <button
      class="new-chat__trigger"
      type="button"
      :disabled="props.disabled"
      :title="props.disabled ? '5 chat limit reached' : undefined"
      @click="toggle"
    >
      + New chat
    </button>

    <form v-if="open" class="new-chat__popup" @submit.prevent="handleCreate">
      <label class="new-chat__field new-chat__field--name">
        <span>Name</span>
        <input v-model="name" type="text" maxlength="50" placeholder="Untitled" />
      </label>

      <label class="new-chat__field">
        <span>Model</span>
        <select v-model="modelKey" class="new-chat__select" @change="handleModelChange">
          <option v-for="model in MODEL_CHOICES" :key="model.key" :value="model.key">
            {{ model.label }}
          </option>
        </select>
      </label>

      <div
        v-if="selectedStatus && !['ready', 'not_downloaded'].includes(selectedStatus.status)"
        class="new-chat__download"
      >
        <div v-if="selectedStatus.status === 'error'" class="new-chat__error">
          Download failed - try selecting the model again.
        </div>
        <template v-else>
          <div class="new-chat__download-bar">
            <div
              class="new-chat__download-fill"
              :style="{ width: downloadPercent + '%' }"
            ></div>
          </div>
          <span class="new-chat__download-label">{{ downloadLabel }}</span>
        </template>
      </div>

      <label class="new-chat__toggle">
        <input v-model="useConditioning" type="checkbox" />
        Non-Conditioned Reference
      </label>

      <label class="new-chat__field">
        <span>Strength</span>
        <input
          v-model.number="strength"
          type="number"
          min="0"
          max="1"
          step="0.05"
          :disabled="!useConditioning"
        />
      </label>

      <label class="new-chat__field">
        <span>Steps</span>
        <input
          v-model.number="steps"
          type="number"
          min="1"
          max="10"
          step="1"
          :disabled="!useConditioning"
        />
      </label>

      <button class="new-chat__submit" type="submit" :disabled="Boolean(stepsError)">
        Create chat
      </button>
      <p v-if="stepsError" class="new-chat__error">{{ stepsError }}</p>
    </form>
  </div>
</template>

<style scoped>
.new-chat {
  position: relative;
  align-self: flex-start;
}

.new-chat__trigger {
  font-size: 13px;
  color: var(--ink);
  background: transparent;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 14px;
  cursor: pointer;
  transition: border-color 0.15s ease, color 0.15s ease;
}

.new-chat__trigger:hover:not(:disabled) {
  border-color: var(--accent);
  color: var(--accent);
}

.new-chat__trigger:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.new-chat__popup {
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

.new-chat__toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.new-chat__toggle input {
  accent-color: var(--accent);
}

.new-chat__field {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.new-chat__field input[type="number"] {
  width: 56px;
  padding: 4px 6px;
  font: inherit;
  color: var(--ink);
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 6px;
}

.new-chat__field input[type="number"]:disabled {
  opacity: 0.4;
}

.new-chat__select {
  padding: 4px 6px;
  font: inherit;
  font-size: 12px;
  color: var(--ink);
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 6px;
  max-width: 110px;
}

.new-chat__field--name {
  flex-direction: column;
  align-items: stretch;
  gap: 4px;
}

.new-chat__field--name input[type="text"] {
  width: 100%;
  padding: 5px 8px;
  font: inherit;
  color: var(--ink);
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 6px;
  box-sizing: border-box;
}

.new-chat__submit {
  margin-top: 2px;
  padding: 8px;
  font: inherit;
  font-weight: 500;
  color: var(--paper);
  background: var(--accent);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: opacity 0.15s ease;
}

.new-chat__submit:hover:not(:disabled) {
  opacity: 0.88;
}

.new-chat__submit:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.new-chat__error {
  margin: 0;
  font-size: 12px;
  color: var(--error);
}

.new-chat__download {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.new-chat__download-bar {
  height: 6px;
  border-radius: 999px;
  background: var(--line);
  overflow: hidden;
}

.new-chat__download-fill {
  height: 100%;
  background: var(--accent);
  transition: width 0.3s ease;
}

.new-chat__download-label {
  font-size: 11px;
  color: var(--ink-faint);
}
</style>
