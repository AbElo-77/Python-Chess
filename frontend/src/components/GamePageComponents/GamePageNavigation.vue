<template>
    <div class="options">
        <header>User Configurations</header>
        <div class="model-choice">Choose Model Type</div>
        <form class="model-selection"> 
          <label @click="updateModel('cnn')" :class="[currentModel == 'cnn' ? 'selected' : '']">Convolutional NN</label>
          <label @click="updateModel('rnn')" :class="[currentModel == 'rnn' ? 'selected' : '']">Recurrent NN</label>
          <label @click="updateModel('gnn')" :class="[currentModel == 'gnn' ? 'selected' : '']">Graph NN</label>
          <label class="disabled">Random Forest Model</label>
          <label class="disabled">Reinforcement Learning</label>
        </form>
    </div>
</template>





<script>
export default {
  name: 'GamePageNavigation',

  components: {},

  data() {
    return {
      currentModel: 'cnn'

    }; 
  }, 
  methods: {
  updateModel(newModel) {
    this.currentModel = newModel;
    this.$emit('model-change', newModel);
    }
  }
}
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=Red+Rose:wght@300..700&display=swap');

header {
    font-size: 2rem; 
    font-family: 'Red Rose', sans-serif;
    color: rgb(15, 185, 0); 
}

.model-choice {
  font-size: 1.5rem; 
  font-family: 'Red Rose', sans-serif;
  color: white;
  margin-top: 2rem; 
  text-decoration: underline;
}

.model-selection {
  display: grid; 
  grid-template-columns: 30%, 70%;
  margin: 1rem 0rem 0rem 1rem;
}

input[type="radio"] {
  appearance: none; 
  cursor: pointer;
  outline: none; 
}

.model-selection > label {
  cursor: pointer; 
  font-family: 'Red Rose', sans-serif; 
  font-size: 1.25rem;
  color: white; 
}

.selected {
  color: rgb(216, 216, 115) !important;
}

.disabled {
  color: grey !important;
  cursor: default !important; 
}


</style>