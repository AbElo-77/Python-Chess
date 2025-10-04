<template>
  <div class="board">
    <div class="controls">
      <button @click="requestMove">Request move (backend)</button>
    </div>

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
          <span v-if="pieces[rank] && pieces[rank][file]">{{ pieces[rank][file] }}</span>
        </BoardSquare>
      </div>
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
      selected: null,
      pieces: Array.from({ length: 8 }, () => Array(8).fill(null)),
    };
  },

  mounted() {
    if (this.fen) {
      this.pieces = this.parseFen(this.fen);
    } else {
      this.fetchBoard();
    }
  },

  methods: {
    onSelect(square) {
      this.selected = square;
      this.$emit('square-selected', square);
    },

    parseFen(fen) {
      const rows = fen.split(' ')[0].split('/');
      const out = Array.from({ length: 8 }, () => Array(8).fill(null));
      for (let r = 0; r < 8; r++) {
        const row = rows[r];
        let file = 0;
        for (const ch of row) {
          if (/[1-8]/.test(ch)) {
            file += parseInt(ch, 10);
          } else {
            const glyph = this.pieceToGlyph(ch);
            out[7 - r][file] = glyph;
            file += 1;
          }
        }
      }
      return out;
    },

    pieceToGlyph(ch) {
      const map = {
        p: '♟', r: '♜', n: '♞', b: '♝', q: '♛', k: '♚',
        P: '♙', R: '♖', N: '♘', B: '♗', Q: '♕', K: '♔'
      };
      return map[ch] || null;
    },

    async fetchBoard() {
      try {
        const res = await fetch('/board');
        const data = await res.json();
        this.pieces = this.parseFen(data.board_fen);
      } catch (e) {
        console.error('failed to fetch board', e);
      }
    },

    async requestMove() {
      try {
        const fen = this.serializePiecesToFen();
        const res = await fetch('/move_cnn', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ current_fen: fen })
        });
        const data = await res.json();
        if (data.success) {
          this.pieces = this.parseFen(data.board_fen);
        } else {
          console.warn('move rejected', data.error);
        }
      } catch (e) {
        console.error('requestMove failed', e);
      }
    },

    serializePiecesToFen() {
      const glyphToLetter = {
        '♟': 'p','♜':'r','♞':'n','♝':'b','♛':'q','♚':'k',
        '♙':'P','♖':'R','♘':'N','♗':'B','♕':'Q','♔':'K'
      };
      const rows = [];
      for (let r = 7; r >= 0; r--) {
        let row = '';
        let empty = 0;
        for (let f = 0; f < 8; f++) {
          const g = this.pieces[r][f];
          if (!g) {
            empty += 1;
          } else {
            if (empty > 0) { row += String(empty); empty = 0; }
            row += glyphToLetter[g] || 'p';
          }
        }
        if (empty > 0) row += String(empty);
        rows.push(row);
      }
      return rows.join('/') + ' w - - 0 1';
    }
  }
};
</script>

<style scoped>

.board {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 8px;
    box-sizing: border-box;
    width: 100vw;
}

.board-grid {
  --square-size: 7vw;
  display: grid;
  grid-template-columns: repeat(8, var(--square-size));
  grid-template-rows: repeat(8, var(--square-size));
  gap: 0;
  justify-content: center;
  align-content: center;
  border: 1px solid white;
  padding: 1rem;
}

.board-row {
    border: 0;
    line-height: 5vh;
}
</style>