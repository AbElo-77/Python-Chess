<template>
  <div
    :class="['square', isLight ? 'light' : 'dark', { selected } ]"
    @click="onClick"
    role="button"
    :aria-label="label"
  >
    <slot />
  </div>
</template>

<script setup>
import { defineProps, defineEmits, computed } from 'vue'

const props = defineProps({
  rank: { type: Number, required: true },
  file: { type: Number, required: true },
  selected: { type: Boolean, default: false },
})

const emit = defineEmits(['select'])

const isLight = computed(() => ((props.rank + props.file) % 2) === 0)
const label = computed(() => `${props.file},${props.rank}`)

function onClick() {
  emit('select', { rank: props.rank, file: props.file })
}
</script>

<style scoped>
.square {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  user-select: none;
}
.light { background: #f0d9b5; }
.dark { background: #b58863; }
.selected { outline: 3px solid rgba(255, 215, 0, 0.8); }
</style>
