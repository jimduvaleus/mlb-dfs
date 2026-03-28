import type { PlayerRow } from './types'

export function getStackNotation(players: PlayerRow[]): string {
  const hitters = players.filter(p => p.position !== 'P')
  const counts: Record<string, number> = {}
  for (const p of hitters) {
    counts[p.team] = (counts[p.team] ?? 0) + 1
  }
  const sorted = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .filter(([, c]) => c > 1)
  if (sorted.length === 0) return ''
  const countPart = sorted.map(([, c]) => c).join('-')
  const teamPart = sorted.map(([t]) => t).join('/')
  return `${countPart} ${teamPart}`
}
