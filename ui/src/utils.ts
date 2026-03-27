import type { PlayerRow } from './types'

export function getStackNotation(players: PlayerRow[]): string {
  const hitters = players.filter(p => p.position !== 'P')
  const counts: Record<string, number> = {}
  for (const p of hitters) {
    counts[p.team] = (counts[p.team] ?? 0) + 1
  }
  const sorted = Object.values(counts).sort((a, b) => b - a)
  const significant = sorted.filter(c => c > 1)
  return significant.join('-')
}
