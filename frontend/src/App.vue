<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
      <nav>
        <RouterLink to="/game">Play Game</RouterLink>
        <RouterLink to="/fen-playhouse">Fen Playhouse</RouterLink>
      </nav>

  <RouterView />
</template>

<style scoped>
nav {
  width: 100%;
  position: fixed; 
  top: 0;
  left: 0;
  font-size: 1.25rem;
  text-align: center;
  justify-content: center;
  margin-top: 2rem;
}

nav a.router-link-exact-active {
  color: var(--color-text);
}

nav a.router-link-exact-active:hover {
  background-color: transparent;
}

nav a {
  display: inline-block;
  padding: 0 1rem;
  border-left: 1px solid var(--color-border);
}

nav a:first-of-type {
  border: 0;
}
</style>
