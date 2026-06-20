import type { PlayerRow } from './types'

const NAME_SUFFIXES = new Set(['jr', 'sr', 'ii', 'iii', 'iv'])

// Mirrors DK's own echo-back convention for duplicate-position roster slots
// (P,P and OF,OF,OF): alphabetical by last name, first name as tiebreak.
// Confirmed empirically by diffing an uploaded vs. downloaded entries CSV.
export function lastNameSortKey(name: string): [string, string] {
  const parts = name.trim().split(/\s+/)
  if (parts.length === 0 || parts[0] === '') return ['', '']
  let last = parts[parts.length - 1].replace(/\.$/, '').toLowerCase()
  if (NAME_SUFFIXES.has(last) && parts.length > 1) {
    last = parts[parts.length - 2].replace(/\.$/, '').toLowerCase()
  }
  const first = parts[0].replace(/\.$/, '').toLowerCase()
  return [last, first]
}

export function compareByLastName(nameA: string, nameB: string): number {
  const [lastA, firstA] = lastNameSortKey(nameA)
  const [lastB, firstB] = lastNameSortKey(nameB)
  return lastA !== lastB ? lastA.localeCompare(lastB) : firstA.localeCompare(firstB)
}

// Reorders items within each group of equal groupKey() (e.g. duplicate
// roster-slot labels like "P" or "OF") by compareByLastName(nameKey()),
// leaving items outside any size-1 group untouched and preserving each
// item's original index outside its group.
export function alphabetizeDuplicateGroups<T>(
  items: T[],
  groupKey: (item: T) => string,
  nameKey: (item: T) => string,
): T[] {
  const groups = new Map<string, number[]>()
  items.forEach((item, i) => {
    const idxs = groups.get(groupKey(item)) ?? []
    idxs.push(i)
    groups.set(groupKey(item), idxs)
  })
  const result = [...items]
  for (const idxs of groups.values()) {
    if (idxs.length < 2) continue
    const sortedIdxs = [...idxs].sort((a, b) => compareByLastName(nameKey(items[a]), nameKey(items[b])))
    idxs.forEach((slotIdx, k) => { result[slotIdx] = items[sortedIdxs[k]] })
  }
  return result
}

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
