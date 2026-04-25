import { useState, useRef, useEffect } from 'react'
import type { LineupResult, PlatformType, PlayerRow } from '../types'
import { getStackNotation } from '../utils'
import TeamBadge from './TeamBadge'

interface Props {
  lineups: LineupResult[]
  unconfirmedPlayerIds?: number[]
  onDeleteLineup?: (lineupIndex: number) => void
  replacingLineupIndex?: number | null
  platform?: PlatformType
}

function formatFdEntryInfo(entryFee?: string | null, contestName?: string | null): string {
  const fee = entryFee ? `$${entryFee}` : ''
  let name = contestName ?? ''
  // Strip everything up to and including "MLB " (e.g. "Sun MLB " → "")
  const mlbIdx = name.indexOf('MLB')
  if (mlbIdx >= 0) name = name.slice(mlbIdx + 3).trimStart()
  // Strip trailing parenthetical (e.g. " (150 Entries Max)")
  name = name.replace(/\s*\([^)]*\)\s*$/, '').trimEnd()
  return [fee, name].filter(Boolean).join(' ')
}

// Maps each slot label to the set of player positions that may fill it.
// DK slots are exact-match; FD adds compound labels (C/1B, UTIL).
const SLOT_ELIGIBILITY: Record<string, ReadonlySet<string>> = {
  'P':    new Set(['P']),
  'C':    new Set(['C']),
  '1B':   new Set(['1B']),
  '2B':   new Set(['2B']),
  '3B':   new Set(['3B']),
  'SS':   new Set(['SS']),
  'OF':   new Set(['OF']),
  'C/1B': new Set(['C', '1B']),
  'UTIL': new Set(['C', '1B', '2B', '3B', 'SS', 'OF']),
}

// Compute a canonical slot assignment for display, guaranteeing each DK/FD
// roster slot appears exactly once and in the correct order.
//
// We parse eligible positions from p.position (the slash-joined display string,
// e.g. "2B/SS") and run a bipartite-matching DFS with most-constrained-first
// ordering so single-position players always claim their natural slot first.
function sortAndAssignPositions(
  players: PlayerRow[],
  platform?: PlatformType,
): Array<{ player: PlayerRow; displayPos: string }> {
  const pitchers = players.filter(p => p.position === 'P')
  const batters  = players.filter(p => p.position !== 'P')

  const posOrder = platform === 'fanduel'
    ? ['C/1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'UTIL']
    : ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']

  // Most-constrained first → canonical, stable assignment
  const sortedBatters = [...batters].sort(
    (a, b) => a.position.split('/').length - b.position.split('/').length
  )

  const slotToPlayer: (PlayerRow | null)[] = new Array(posOrder.length).fill(null)

  function canFill(player: PlayerRow, slotIdx: number): boolean {
    const elig = player.position.split('/')
    const accepts = SLOT_ELIGIBILITY[posOrder[slotIdx]] ?? new Set([posOrder[slotIdx]])
    return elig.some(pos => accepts.has(pos))
  }

  function tryAssign(player: PlayerRow, visited: Set<number>): boolean {
    for (let j = 0; j < posOrder.length; j++) {
      if (!visited.has(j) && canFill(player, j)) {
        visited.add(j)
        const occ = slotToPlayer[j]
        if (occ === null || tryAssign(occ, visited)) {
          slotToPlayer[j] = player
          return true
        }
      }
    }
    return false
  }

  for (const batter of sortedBatters) {
    tryAssign(batter, new Set())
  }

  const result: Array<{ player: PlayerRow; displayPos: string }> = [
    ...pitchers.map(p => ({ player: p, displayPos: 'P' })),
  ]

  const matched = new Set<PlayerRow>()
  for (let j = 0; j < posOrder.length; j++) {
    const player = slotToPlayer[j]
    if (player !== null) {
      result.push({ player, displayPos: posOrder[j] })
      matched.add(player)
    }
  }

  // Safety valve: any unmatched batters go at the end (shouldn't occur for valid lineups)
  for (const batter of batters) {
    if (!matched.has(batter)) {
      result.push({ player: batter, displayPos: batter.position.split('/')[0] })
    }
  }

  return result
}

export function PortfolioTable({ lineups, unconfirmedPlayerIds, onDeleteLineup, replacingLineupIndex, platform }: Props) {
  const [filterPlayer, setFilterPlayer] = useState<PlayerRow | null>(null)
  const [search, setSearch] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const searchWrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (searchWrapRef.current && !searchWrapRef.current.contains(e.target as Node)) {
        setSearchOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  if (lineups.length === 0) return null

  const allPlayers = Array.from(
    new Map(lineups.flatMap(l => l.players).map(p => [p.player_id, p])).values()
  ).sort((a, b) => a.name.localeCompare(b.name))

  const searchLower = search.toLowerCase()
  const searchResults = allPlayers
    .filter(p => p.name.toLowerCase().includes(searchLower))
    .slice(0, 10)

  const visibleLineups = filterPlayer
    ? lineups.filter(l => l.players.some(p => p.player_id === filterPlayer.player_id))
    : lineups

  const unconfirmedSet = new Set(unconfirmedPlayerIds ?? [])

  const unconfirmedByPlayer = new Map<number, { name: string; count: number }>()
  for (const lineup of visibleLineups) {
    for (const p of lineup.players) {
      if (unconfirmedSet.has(p.player_id)) {
        const entry = unconfirmedByPlayer.get(p.player_id)
        if (entry) {
          entry.count++
        } else {
          unconfirmedByPlayer.set(p.player_id, { name: p.name, count: 1 })
        }
      }
    }
  }

  const totalUnconfirmed = Array.from(unconfirmedByPlayer.values()).reduce((sum, e) => sum + e.count, 0)

  const sortedUnconfirmedPlayers = Array.from(unconfirmedByPlayer.values()).sort((a, b) => b.count - a.count)
  const breakdown = unconfirmedByPlayer.size <= 5
    ? ' — ' + sortedUnconfirmedPlayers.map(e => `${e.count} ${e.name}`).join(', ')
    : ''

  return (
    <div className="portfolio-table-wrap">
      <h3>Portfolio — {filterPlayer ? `${visibleLineups.length} / ${lineups.length}` : lineups.length} Lineups</h3>
      <div className="portfolio-filter" ref={searchWrapRef}>
        {filterPlayer ? (
          <span className="portfolio-filter-chip">
            {filterPlayer.name}
            <button onClick={() => { setFilterPlayer(null); setSearch('') }}>×</button>
          </span>
        ) : (
          <>
            <input
              className="portfolio-filter-input"
              placeholder="Filter by player…"
              value={search}
              onChange={e => { setSearch(e.target.value); setSearchOpen(true) }}
              onFocus={() => setSearchOpen(true)}
            />
            {searchOpen && searchResults.length > 0 && (
              <div className="portfolio-filter-results">
                {searchResults.map(p => (
                  <button
                    key={p.player_id}
                    className="portfolio-filter-result-btn"
                    onMouseDown={e => {
                      e.preventDefault()
                      setFilterPlayer(p)
                      setSearch('')
                      setSearchOpen(false)
                    }}
                  >
                    <span>{p.name}</span>
                    <span className="portfolio-filter-result-meta">{p.position} · {p.team}</span>
                  </button>
                ))}
              </div>
            )}
          </>
        )}
      </div>
      <div className={`portfolio-unconfirmed-banner ${totalUnconfirmed === 0 ? 'portfolio-unconfirmed-banner--clear' : ''}`}>
        {totalUnconfirmed === 0
          ? '✓ All lineup slots confirmed'
          : `✕ ${totalUnconfirmed} unconfirmed lineup slot${totalUnconfirmed !== 1 ? 's' : ''} across portfolio${breakdown}`}
      </div>
      <div className="portfolio-cards">
        {visibleLineups.map(lineup => {
          const sorted = sortAndAssignPositions(lineup.players, platform)
          const stack = getStackNotation(lineup.players)
          const isReplacing = replacingLineupIndex === lineup.lineup_index
          return (
            <div key={lineup.lineup_index} className="lineup-card">
              <div className="lineup-card-header">
                <span className="lineup-card-num">#{lineup.lineup_index}</span>
                <span className="lineup-card-salary">${lineup.lineup_salary.toLocaleString()}</span>
                <div className="lineup-card-header-right">
                  {isReplacing ? (
                    <span className="lineup-card-generating">Generating…</span>
                  ) : (
                    <>
                      {stack && <span className="lineup-card-stack">{stack}</span>}
                      {onDeleteLineup && (
                        <button
                          className="lineup-card-delete"
                          onClick={() => onDeleteLineup(lineup.lineup_index)}
                          title="Delete and replace this lineup"
                          disabled={replacingLineupIndex != null}
                        >
                          🗑
                        </button>
                      )}
                    </>
                  )}
                </div>
              </div>
              {lineup.upload_tag && (
                <div className="lineup-card-entry-info">
                  {platform === 'fanduel'
                    ? formatFdEntryInfo(lineup.entry_fee, lineup.contest_name)
                    : `${lineup.upload_tag}${lineup.entry_fee ? ` ${lineup.entry_fee}` : ''}${lineup.contest_name ? ` ${lineup.contest_name}` : ''}`}
                </div>
              )}
              <div className="lineup-card-players">
                {sorted.map(({ player: p, displayPos }, i) => (
                  <div key={i} className="lineup-player">
                    <span className="lineup-player-pos">{displayPos}</span>
                    <span className="lineup-player-name">
                      {p.name}
                      {displayPos !== 'P' && (() => {
                        const slotNum = p.slot != null && p.slot >= 1 && p.slot <= 9 ? p.slot : null
                        if (p.slot_confirmed) {
                          return <span className="batting-slot-bubble batting-slot-bubble--confirmed" title="Confirmed lineup slot">{slotNum ?? '?'}</span>
                        }
                        return <span className="batting-slot-bubble batting-slot-bubble--projected" title="Projected lineup slot">{slotNum ?? '?'}</span>
                      })()}
                    </span>
                    <TeamBadge team={p.team} className="lineup-player-team" />
                    <span className="lineup-player-sal">${(p.salary / 1000).toFixed(1)}k</span>
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
