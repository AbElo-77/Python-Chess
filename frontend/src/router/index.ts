import { createRouter, createWebHistory } from 'vue-router'
import ChessBoard from '@/components/GamePage Components/ChessBoard.vue'

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
      component: () => import('@/components/FENPage Components/FENBoard.vue'),
    },
  ],
})

export default router
