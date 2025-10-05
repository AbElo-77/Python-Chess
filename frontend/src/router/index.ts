import { createRouter, createWebHistory } from 'vue-router'
import ChessBoard from '@/views/ChessBoard.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/game',
      name: 'game',
      component: ChessBoard,
    },
    {
      path: '/fen-playhouse',
      name: 'FEN-playhouse',
      component: () => import('@/views/FENBoard.vue'),
    },
  ],
})

export default router
