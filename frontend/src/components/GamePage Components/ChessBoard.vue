<template>
  <div class="board">
    <div class="board-grid">
      <div v-for="(rank, rIdx) in ranks" :key="rIdx" class="board-row">
        <BoardSquare
          v-for="(file, fIdx) in files"
          :key="`${rIdx}-${fIdx}`"
          :rank="rank"
          :file="file"
          :selected="selected && selected.rank === rank && selected.file === file"
          @select="onSelect"
        >
        </BoardSquare>
      </div>`1`
    </div>
  </div>
</template>

<script>
export default {
  
  name: 'ChessBoard',
  components: {
    BoardSquare: () => import('./BoardSquare.vue')
  },

  props: {
    fen: { type: String, required: false }
  },

  data() {
    return {
      ranks: [7,6,5,4,3,2,1,0],
      files: [0,1,2,3,4,5,6,7],
      selected: null
    }
  },
  methods: {
    onSelect(square) {
      this.selected = square;
      this.$emit('square-selected', square);
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

.board {
    display: flex;
    justify-content: center;
    margin: 0 auto;
}

.board-grid {
  --square-size: 7vw;
  display: grid;
  grid-template-columns: repeat(8, var(--square-size));
  grid-template-rows: repeat(8, var(--square-size));
  gap: 0;
  justify-items: center;
  align-content: center;
}

.board-row {
    border: 0;
    line-height: 5vh;
}
</style>