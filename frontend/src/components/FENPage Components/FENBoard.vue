<template>
  <div>
    <p>Count: {{ count }}</p>
    <button @click="incrementCount">Increment</button>
  </div>
</template>

<script>
export default {
  
  name: 'FENBoard',

  setup() {
    return {
      count: 0
    };
  },

  methods: {
    incrementCount() {
      this.count++;
    }
  }
};
</script>

<style scoped>

div {
  border: 1px solid #ccc;
  padding: 10px;
  margin: 10px;
  text-align: center;
}

button {
  background-color: #4CAF50;
  color: white;
  padding: 8px 16px;
  border: none;
  cursor: pointer;
}
</style>