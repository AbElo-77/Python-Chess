<template>
  <div class="fen-page">
    <div class="board-container">
      <div class="board-grid">
        <div v-for="(rank, rIdx) in ranks" :key="rIdx" class="board-row">
          <div
            v-for="(file, fIdx) in files"
            :key="`${fIdx}-${rIdx}`"
            :class="['square', isLight(file, rank) ? 'light' : 'dark', selected && selected.rank === rank && selected.file === file ? 'selected' : '']"
            @click="onSquareClick({ rank, file })"
            @mousedown.prevent="onMouseDown({ file, rank })"
            @mouseup.prevent="onMouseUp({ file, rank })"
          >
            <img
              v-if="pieces[file][rank]"
              :src="getPieceImage(pieces[file][rank])"
              class="piece-img"
            />
          </div>
        </div>
      </div>
    </div>
    <div class="fen-controls">
      <h2>FEN Editor</h2>
      <textarea v-model="fenInput" placeholder="Enter or edit FEN here"></textarea>
      <div class="fen-buttons">
        <button @click="loadFen">Load FEN</button>
        <button @click="resetBoard">Reset Board</button>
      </div>
    </div>
  </div>
</template>

<script>
import { Chess } from 'chess.js';

export default {
  name: 'FenBoard',

  data() {
    return {
      chess: new Chess(),
      fen: 'startpos',
      fenInput: '',
      ranks: [0,1,2,3,4,5,6,7],
      files: [0,1,2,3,4,5,6,7],
      pieces: Array.from({ length: 8 }, () => Array(8).fill(null)),
      selected: null,
    };
  },

  mounted() {
    this.loadBoardFromFen(this.chess.fen());
  },

  methods: {
    parseFen(fen) {
      const rows = fen.split(' ')[0].split('/');
      const out = Array.from({ length: 8 }, () => Array(8).fill(null));
      for (let r = 0; r < 8; r++) {
        const row = rows[r];
        let file = 0;
        for (const ch of row) {
          if (/[1-8]/.test(ch)) {
            file += parseInt(ch);
          } else {
            out[file][7 - r] = ch;
            file++;
          }
        }
      }
      return out;
    },
    onMouseUp(square) {
      if (this.dragStart) {
        const from = this.toAlgebraic(this.dragStart);
        const to = this.toAlgebraic(square);

        this.$emit('move', { from, to });
      }
    },

    onKeyDown(e) {
      if (!this.selected) return;
      const { rank, file } = this.selected;
      let nr = rank;
      let nf = file;
      if (e.key === 'ArrowUp') nf = Math.min(7, file - 1);
      if (e.key === 'ArrowDown') nf = Math.max(0, file + 1);
      if (e.key === 'ArrowLeft') nr = Math.max(0, rank + 1);
      if (e.key === 'ArrowRight') nr = Math.min(7, rank - 1);
      if (nr !== rank || nf !== file) {
        this.selected = { rank: nr, file: nf };
        this.$emit('square-selected', this.toAlgebraic(this.selected));
      }
      if (e.key === 'Enter') {
        this.$emit('enter-pressed', this.toAlgebraic(this.selected));
      }
    },

    loadBoardFromFen(fen) {
      try {
        this.chess.load(fen);
        this.pieces = this.parseFen(fen);
        this.fen = this.chess.fen();
        this.fenInput = this.chess.fen();
      } catch (err) {
        alert('Invalid FEN: ' + err.message);
      }
    },

    loadFen() {
      this.loadBoardFromFen(this.fenInput.trim());
    },

    resetBoard() {
      this.chess.reset();
      this.loadBoardFromFen(this.chess.fen());
    },

    toAlgebraic({ rank, file }) {
      const files = ['a','b','c','d','e','f','g','h'];
      return `${files[file]}${rank + 1}`;
    },

    isLight(file, rank) {
      return (file + rank) % 2 === 0;
    },

    getPieceImage(letter) {
      const map = {
        p: '/black_pawn.png', r: '/black_rook.png', n: '/black_knight.png',
        b: '/black_bishop.png', q: '/black_queen.png', k: '/black_king.png',
        P: '/white_pawn.png', R: '/white_rook.png', N: '/white_knight.png',
        B: '/white_bishop.png', Q: '/white_queen.png', K: '/white_king.png',
      };
      return map[letter] || '';
    },

    onSquareClick(square) {
      if (!this.selected) {
        const piece = this.pieces[square.file][square.rank];
        if (piece) {
          this.selected = square;
        }
      } else {
        const from = this.toAlgebraic(this.selected);
        const to = this.toAlgebraic(square);
        const move = this.chess.move({ from, to, promotion: 'q' });

        if (move) {
          this.loadBoardFromFen(this.chess.fen());
        } else {
          console.warn('Illegal move:', from, '→', to);
        }
        this.selected = null;
      }
    },
  }
};
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=Red+Rose:wght@300..700&display=swap');

.fen-page {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 2rem;
  margin-top: 2rem;
  color: white;
  width: 100vw;
}

.fen-controls {
  width: 25%;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  font-family: 'Red Rose';
  color:  rgb(0, 68, 17);
}

textarea {
  width: 100%;
  height: 6rem;
  padding: 0.5rem;
  font-family: monospace;
  background-color: rgb(0, 68, 17);
}

.fen-buttons {
  display: flex;
  gap: 1rem;
}

.board-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.board-grid {
  --square-size: 5vw;
  display: grid;
  grid-template-columns: repeat(8, var(--square-size));
  grid-template-rows: repeat(8, var(--square-size));
  border: 2px solid #fff;
  padding: 1rem;
}

.square {
  width: var(--square-size);
  height: var(--square-size);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.light { background: #f0d9b5; }
.dark { background: #b58863; }
.selected { outline: 3px solid gold; }

.piece-img {
  max-width: 90%;
  max-height: 90%;
}
</style>
