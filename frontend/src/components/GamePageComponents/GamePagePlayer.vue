<template>
    <div class="player-information">
      <header>Player vs. Computer</header>
      <div class="pieces-list">
           <div><img v-for="(imgSrc, index) in capturedPieces" :key="index" :src="imgSrc"></div>
           <div><img v-for="(imgSrc, index) in computerPieces" :key="index" :src="imgSrc"></div>
      </div>
    </div>
</template>





<script>
export default {
  name: 'GamePagePlayer',

  props: {
    capturedPieces: {type: Array, required: true, default: ['/white_pawn.png']},
    computerPieces: {type: Array, required: true, default: ['/white_pawn.png']}
  },

  components: {},

  data() {
    return {}; 
  }
}
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=Red+Rose:wght@300..700&display=swap');

  .player-information {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  header {
      font-size: 2rem;
      font-family: 'Red Rose', sans-serif;
      color: rgb(15, 185, 0); 
      text-align: center;
  }

  .pieces-list {
    display: flex; 
    flex-direction: column;
    justify-content: space-between;
    border: 2px solid rgba(124, 235, 114, 0.5);
    background-color: rgb(32, 32, 32);
    padding: 1rem;
  }


  .pieces-list > div > img {
    width: 25px;
  }


</style>