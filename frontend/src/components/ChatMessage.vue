<script setup lang="ts">
import type { ChatImage } from "../types";
import ImagePair from "./ImagePair.vue";

const props = defineProps<{
  log: ChatImage;
}>();
</script>

<template>
  <li class="message">
    <p class="message__prompt">{{ props.log.text }}</p>
    <ImagePair
      v-if="props.log.image"
      :image="props.log.image"
      :comparison-image="props.log.comparison_image"
      :alt="props.log.text"
    />
  </li>
</template>

<style scoped>
.message {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 18px 0;
  border-bottom: 1px solid var(--line);
}

.message:last-child {
  border-bottom: none;
}

.message__prompt {
  margin: 0;
  font-size: 15px;
  color: var(--ink);
  max-width: 60ch;
}
</style>
