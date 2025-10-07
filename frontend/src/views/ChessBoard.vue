<template>
  <div class="game-page">
    <div class="player-information">
      <GamePagePlayer
      :capturedPieces="capturedPieces"
      :computerPieces="computerPieces"
      />
    </div>
    <div class="board">
      <div class="board-grid">
        <div v-for="(rank, rIdx) in ranks" :key="rIdx" class="board-row">
          <div
            v-for="(file, fIdx) in files"
            :key="`${fIdx}-${rIdx}`"
            :id="squareId(file, rank)"
            :class="['square', isLight(file, rank) ? 'light' : 'dark', selected && selected.rank === rank && selected.file === file ? 'selected' : '']"
            @click="onSelect({ rank, file })"
            @mousedown.prevent="onMouseDown({ file, rank })"
            @mouseup.prevent="onMouseUp({ file, rank })"
          >
            <img v-if="pieces[file] && pieces[file][rank]" :src="getPieceImage(pieces[file][rank])" :alt="pieces[file][rank]" class="piece-img" />
          </div>
        </div>
      </div>
    </div>
    <div class="options-menu">
      <GamePageNavigation
      @model-change="modelChange"
    ></GamePageNavigation>
    </div>
  </div>
</template>

<script>
import GamePageNavigation from '@/components/GamePageComponents/GamePageNavigation.vue';
import GamePagePlayer from '@/components/GamePageComponents/GamePagePlayer.vue';

export default {
  name: 'ChessBoard',

  components: {GamePageNavigation, GamePagePlayer},

  props: {
    fen: { type: String, required: false }
  },

  data() {
    return {
      ranks: [0, 1, 2, 3, 4, 5, 6, 7],
      files: [0, 1, 2, 3, 4, 5, 6, 7],
      selected: null,
      pieces: Array.from({ length: 8 }, () => Array(8).fill(null)),
      dragStart: null,
      debugUseStartingFen: true,
      localFen: this.fen || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
      currentModel: 'cnn',
      capturedPieces: [],
      computerPieces: [],
      };
  },

  watch: {
  fen(newFen) {
    if (newFen && newFen !== this.localFen) {
      this.localFen = newFen;
      this.pieces = this.parseFen(newFen);
      }
    }
  }, 

  mounted() {
    if (this.localFen) {
      this.pieces = this.parseFen(this.localFen);
    } else {
      this.fetchBoard();
    }
    window.addEventListener('keydown', this.onKeyDown);
  },

  beforeUnmount() {
    window.removeEventListener('keydown', this.onKeyDown);
  },

  methods: {

    modelChange(model) {
      this.currentModel = model;

    }, 

    onSelect(square) {
      this.selected = square;
      this.$emit('square-selected', this.toAlgebraic(square));
    },

    onMouseDown(square) {
      this.dragStart = square;
      this.selected = square;
    },

    onMouseUp(square) {
      if (this.dragStart) {
        const from = this.toAlgebraic(this.dragStart);
        const to = this.toAlgebraic(square);

        const uci = `${from}${to}`;
        const backup = this.clonePieces();
        const applied = this.applyLocalMove(from, to);
        this.dragStart = null;

        fetch('http://127.0.0.1:5000/make_move_user', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ move: uci, current_fen: this.localFen })
        })
        .then(r => r.json())
        .then(data => {
          if (data.success) {
            this.pieces = this.parseFen(data.board_fen);
            this.localFen = data.board_fen;
            if (data.capture) { this.capturedPieces.push(this.getPieceImage(data.capture)) };
            this.capturedPieces = this.capturedPieces.sort(); 
            this.$emit('update:fen', data.board_fen); 

            this.requestMove();
          } else {
            console.warn('server rejected move:', data.error);
            this.pieces = backup;
          }
        })
        .catch((err) => {
          console.error('failed to send user move', err);
          this.pieces = backup;
        });

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
            out[file][7 - r] = ch;
            file += 1;
          }
        }
      }
      return out;
    },

    squareId(rank, file) {
      const files = ['a','b','c','d','e','f','g','h'];
      return `square-${files[file]}${rank + 1}`;
    },

    isLight(rank, file) {
      return ((rank + file) % 2) !== 0;
    },

    toAlgebraic({ rank, file }) {
      const files = ['a','b','c','d','e','f','g','h'];
      return `${files[file]}${rank + 1}`;
    },

    algebraicToCoords(square) {
      const files = ['a','b','c','d','e','f','g','h'];
      const file = files.indexOf(square[0]);
      const rank = parseInt(square.slice(1), 10) - 1;
      return { rank, file };
    },

    fenSquareToId(square) {
      if (!square) return null;
      if (/^[a-h][1-8]$/.test(square)) {
        return `square-${square}`;
      }
      return null;
    },

    fenToSquareIdMap(fen) {
      const rows = fen.split(' ')[0].split('/');
      const map = {};
      for (let r = 0; r < 8; r++) {
        const row = rows[r];
        let file = 0;
        for (const ch of row) {
          if (/[1-8]/.test(ch)) {
            file += parseInt(ch, 10);
          } else {
            const filesArr = ['a','b','c','d','e','f','g','h'];
            const algebraic = `${filesArr[file]}${7 - r}`;
            const id = `square-${algebraic}`;
            map[id] = ch;
            file += 1;
          }
        }
      }
      return map;
    },

    getPieceImage(fenLetter) {
      const map = {
        p: '/black_pawn.png', r: '/black_rook.png', n: '/black_knight.png', b: '/black_bishop.png', q: '/black_queen.png', k: '/black_king.png',
        P: '/white_pawn.png', R: '/white_rook.png', N: '/white_knight.png', B: '/white_bishop.png', Q: '/white_queen.png', K: '/white_king.png'
      };
      return map[fenLetter] || '';
    },

    async fetchBoard() {
      try {
        const res = await fetch('http://127.0.0.1:5000/board');
        const data = await res.json();
        this.pieces = this.parseFen(data.board_fen);
      } catch (e) {
        console.error('failed to fetch board', e);
      }
    },

    async requestMove() {
      try {
        const fen = this.localFen;
        const res = await fetch(`http://127.0.0.1:5000/move_${this.currentModel}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ current_fen: fen })
        });
        const data = await res.json();
        if (data.success) {
          this.pieces = this.parseFen(data.board_fen);
          this.localFen = data.board_fen;
          if (data.capture) { this.computerPieces.push(this.getPieceImage(data.capture))};
          this.computerPieces = this.computerPieces.sort(); 
          this.$emit('update:fen', data.board_fen); 
        } else {
          console.warn('move rejected', data.error);
        }
      } catch (e) {
        console.error('requestMove failed', e);
      }
    },

    clonePieces() {
      return this.pieces.map(row => row.slice());
    },

    applyLocalMove(fromAlgebraic, toAlgebraic) {

      const from = this.algebraicToCoords(fromAlgebraic);
      const to = this.algebraicToCoords(toAlgebraic);

      if (!from || !to) return false;
      const piece = this.pieces[from.rank][from.file];

      this.pieces[from.rank][from.file] = null;
      this.pieces[to.rank][to.file] = piece;
      return true;
    },

    async postFenAndUpdate(fen) {
      const res = await fetch('http://127.0.0.1:5000/move_cnn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ current_fen: fen })
      });
      const data = await res.json();
      if (data.success) {
        this.pieces = this.parseFen(data.board_fen);
        return data;
      }
      throw new Error(data.error || 'move rejected');
    },

    serializePiecesToFen() {
      const rows = [];
      for (let r = 7; r >= 0; r--) {
        let row = '';
        let empty = 0;
        for (let f = 0; f < 8; f++) {
          const p = this.pieces[f][r];
          if (!p) {
            empty += 1;
          } else {
            if (empty > 0) { row += String(empty); empty = 0; }
            row += p;
          }
        }
        if (empty > 0) row += String(empty);
        rows.push(row);
      }
      return rows.join('/') + ' w - - 0 1';
    }
  }
}
</script>

<style scoped>

.game-page {
  display: flex; 
  justify-content: center;
}

.player-information {
  display: flex; 
  justify-content: space-between;
  box-sizing: border-box;
  width: 25vw;
  padding: 0rem 0.5rem;
}

.board {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 8px;
    margin-top: 2rem;
    box-sizing: border-box;
}

.board-grid {
  --square-size: 6vw;
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

.square {
  width: var(--square-size);
  height: var(--square-size);
  display: flex;
  align-items: center;
  justify-content: center;
  user-select: none;
}
.light { background: #f0d9b5; }
.dark { background: #b58863; }
.selected { outline: 3px solid rgba(255, 215, 0, 0.8); }

.square > img {
  max-height: 90%;
}

.options-menu {
  display: flex; 
  justify-content: center;
  box-sizing: border-box;
  width: 22.5vw;
}
</style>