<script setup lang="ts">
import { mediaUrl } from "../api";

const props = defineProps<{
  image: string;
  comparisonImage: string | null;
  alt: string;
}>();
</script>

<template>
  <div v-if="props.comparisonImage" class="image-pair">
    <figure class="image-pair__frame image-pair__frame--side">
      <img :src="mediaUrl(props.comparisonImage)" :alt="`${props.alt} (no reference)`" />
      <figcaption>No reference</figcaption>
    </figure>
    <figure class="image-pair__frame image-pair__frame--main">
      <img :src="mediaUrl(props.image)" :alt="props.alt" />
      <figcaption>Result</figcaption>
    </figure>
  </div>
  <figure v-else class="image-pair__frame image-pair__frame--solo">
    <img :src="mediaUrl(props.image)" :alt="props.alt" />
  </figure>
</template>

<style scoped>
.image-pair {
  display: flex;
  align-items: flex-end;
  gap: 10px;
}

.image-pair__frame {
  margin: 0;
  min-width: 0;
}

.image-pair__frame--side {
  flex: 1;
}

.image-pair__frame--main {
  flex: 2;
}

.image-pair__frame--solo {
  max-width: 60%;
}

.image-pair__frame img {
  width: 100%;
  display: block;
  border-radius: 10px;
  border: 1px solid var(--line);
}

.image-pair__frame figcaption {
  margin-top: 6px;
  font-size: 11px;
  color: var(--ink-faint);
  text-align: center;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
</style>
