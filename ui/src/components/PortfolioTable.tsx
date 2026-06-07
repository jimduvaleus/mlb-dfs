import { useState, useRef, useEffect } from 'react'
import type { LineupResult, PlatformType, PlayerRow, PortfolioSweepEntry } from '../types'
import { fetchContestAnalysis } from '../api'
import { getStackNotation } from '../utils'
import TeamBadge from './TeamBadge'

interface Props {
  lineups: LineupResult[]
  optimalLineups?: LineupResult[]
  portfolioSweep?: PortfolioSweepEntry[]
  activeRisk?: number
  onActivateRisk?: (risk: number) => void
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

function entryInfoText(lineup: LineupResult, platform?: PlatformType): string | null {
  if (!lineup.upload_tag) return null
  return platform === 'fanduel'
    ? formatFdEntryInfo(lineup.entry_fee, lineup.contest_name)
    : `${lineup.upload_tag}${lineup.entry_fee ? ` ${lineup.entry_fee}` : ''}${lineup.contest_name ? ` ${lineup.contest_name}` : ''}`
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

function buildNormalizedFptsMap(fpts: Record<string, number>): Map<string, number> {
  const m = new Map<string, number>()
  for (const [name, val] of Object.entries(fpts)) {
    m.set(name.toLowerCase(), val)
  }
  return m
}

function lookupFpts(name: string, normalized: Map<string, number>): number {
  return normalized.get(name.toLowerCase()) ?? 0
}

function calcLineupFpts(lineup: LineupResult, normalized: Map<string, number>): number {
  return lineup.players.reduce((sum, p) => sum + lookupFpts(p.name, normalized), 0)
}

function calcSweepStats(lineups: LineupResult[], norm: Map<string, number>) {
  const scores = lineups.map(l => calcLineupFpts(l, norm))
  return {
    max: Math.max(...scores),
    avg: scores.reduce((a, b) => a + b, 0) / scores.length,
  }
}

function parseFeeCents(entryFee: string | null | undefined): number {
  if (!entryFee) return 0
  return Math.round(parseFloat(entryFee.replace(/[^0-9.]/g, '')) * 100)
}

function entrySortKey(lineup: LineupResult): [number, number, number] {
  const ratio = lineup.entry_sort_order ?? Infinity
  const fee = parseFeeCents(lineup.entry_fee)
  return [ratio, -fee, lineup.lineup_index]
}

function compareEntrySortKey(a: LineupResult, b: LineupResult): number {
  const [ra, fa, ia] = entrySortKey(a)
  const [rb, fb, ib] = entrySortKey(b)
  return ra !== rb ? ra - rb : fa !== fb ? fa - fb : ia - ib
}

function sortByEntryRatio(lineups: LineupResult[]): LineupResult[] {
  if (!lineups.some(l => l.upload_tag)) return lineups
  return [...lineups].sort(compareEntrySortKey)
}

function playerKey(players: PlayerRow[]): string {
  return [...players.map(p => p.player_id)].sort((a, b) => a - b).join(',')
}

export function PortfolioTable({ lineups, optimalLineups = [], portfolioSweep = [], activeRisk = 1, onActivateRisk, unconfirmedPlayerIds, onDeleteLineup, replacingLineupIndex, platform }: Props) {
  const [activeTab, setActiveTab] = useState<'portfolio' | 'optimal'>('portfolio')
  // viewingRisk: which risk the user is currently browsing (null = showing active)
  const [viewingRisk, setViewingRisk] = useState<number | null>(null)
  // Shared filter across all risk levels and both tabs
  const [filterPlayer, setFilterPlayer] = useState<PlayerRow | null>(null)
  const [search, setSearch] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const searchWrapRef = useRef<HTMLDivElement>(null)
  // Contest analysis state
  const [contestNormalized, setContestNormalized] = useState<Map<string, number>>(new Map())
  const [contestError, setContestError] = useState<string | null>(null)
  const [contestLoading, setContestLoading] = useState(false)
  const [sortByActual, setSortByActual] = useState(false)

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (searchWrapRef.current && !searchWrapRef.current.contains(e.target as Node)) {
        setSearchOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  async function handleAnalyzeContest() {
    setContestLoading(true)
    setContestError(null)
    try {
      const result = await fetchContestAnalysis()
      setContestNormalized(buildNormalizedFptsMap(result.player_fpts))
    } catch (e) {
      setContestError(e instanceof Error ? e.message : String(e))
    } finally {
      setContestLoading(false)
    }
  }

  if (lineups.length === 0) return null

  // Derive displayed lineups. viewingRisk=null shows the active portfolio (state.portfolio).
  const hasSweep = portfolioSweep.length > 0
  const displayedRisk = viewingRisk ?? activeRisk
  const sweepEntry = portfolioSweep.find(e => e.risk === displayedRisk)
  const isPrimary = displayedRisk === activeRisk
  // When viewing the active risk, prefer the main lineups prop (which carries entry meta from the
  // server) over the sweep entry's lineups (which may lack entry meta when loaded from disk).
  const activeLineups: LineupResult[] = (sweepEntry && !isPrimary) ? sweepEntry.lineups : lineups

  const allPlayers = Array.from(
    new Map(activeLineups.flatMap(l => l.players).map(p => [p.player_id, p])).values()
  ).sort((a, b) => a.name.localeCompare(b.name))

  const searchLower = search.toLowerCase()
  const searchResults = allPlayers
    .filter(p => p.name.toLowerCase().includes(searchLower))
    .slice(0, 10)

  const sortedActiveLineups = sortByEntryRatio(activeLineups)
  const filteredLineups = filterPlayer
    ? sortedActiveLineups.filter(l => l.players.some(p => p.player_id === filterPlayer.player_id))
    : sortedActiveLineups
  const visibleLineups = (contestNormalized.size > 0 && sortByActual)
    ? [...filteredLineups].sort((a, b) => calcLineupFpts(b, contestNormalized) - calcLineupFpts(a, contestNormalized))
    : filteredLineups
  const filterPlayerMissingFromRisk = filterPlayer !== null && filteredLineups.length === 0

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

  // portfolio key → lineup_index
  const portfolioKeyMap = new Map<string, number>(
    activeLineups.map(l => [playerKey(l.players), l.lineup_index])
  )

  // portfolio lineup_index → LineupResult (for looking up entry info on optimal cards)
  const portfolioLineupByIndex = new Map<number, LineupResult>(
    activeLineups.map(l => [l.lineup_index, l])
  )

  // optimal key → lineup_index (for showing Opt# on portfolio cards)
  const optimalIndexMap = new Map<string, number>(
    optimalLineups.map(ol => [playerKey(ol.players), ol.lineup_index])
  )

  // Only show optimal lineups that are included in the portfolio
  const optimalInPortfolio = optimalLineups.filter(ol => portfolioKeyMap.has(playerKey(ol.players)))

  const optimalAllPlayers = Array.from(
    new Map(optimalInPortfolio.flatMap(l => l.players).map(p => [p.player_id, p])).values()
  ).sort((a, b) => a.name.localeCompare(b.name))

  const optimalSearchResults = optimalAllPlayers
    .filter(p => p.name.toLowerCase().includes(searchLower))
    .slice(0, 10)

  const hasEntries = activeLineups.some(l => l.upload_tag)
  const sortedOptimalInPortfolio = hasEntries
    ? [...optimalInPortfolio].sort((a, b) => {
        const portA = portfolioLineupByIndex.get(portfolioKeyMap.get(playerKey(a.players))!)
        const portB = portfolioLineupByIndex.get(portfolioKeyMap.get(playerKey(b.players))!)
        const proxyA = portA ?? a
        const proxyB = portB ?? b
        return compareEntrySortKey(proxyA, proxyB)
      })
    : optimalInPortfolio
  const visibleOptimalLineups = filterPlayer
    ? sortedOptimalInPortfolio.filter(l => l.players.some(p => p.player_id === filterPlayer.player_id))
    : sortedOptimalInPortfolio
  const filterPlayerMissingFromOptimal = filterPlayer !== null && visibleOptimalLineups.length === 0

  const showOptimalTab = optimalLineups.length > 0

  const tabLabel = activeTab === 'optimal'
    ? `Optimal — ${filterPlayer ? `${visibleOptimalLineups.length} / ${optimalInPortfolio.length}` : optimalInPortfolio.length} Lineup${optimalInPortfolio.length !== 1 ? 's' : ''}`
    : `Portfolio — ${filterPlayer ? `${visibleLineups.length} / ${activeLineups.length}` : activeLineups.length} Lineup${activeLineups.length !== 1 ? 's' : ''}`

  return (
    <div className="portfolio-table-wrap">
      <div className="portfolio-tabs">
        {showOptimalTab && (
          <>
            <button
              className={`portfolio-tab${activeTab === 'portfolio' ? ' portfolio-tab--active' : ''}`}
              onClick={() => setActiveTab('portfolio')}
            >
              Portfolio ({lineups.length})
            </button>
            <button
              className={`portfolio-tab${activeTab === 'optimal' ? ' portfolio-tab--active' : ''}`}
              onClick={() => setActiveTab('optimal')}
            >
              Optimal ({optimalInPortfolio.length})
            </button>
          </>
        )}
        <span className="portfolio-tab-label">{tabLabel}</span>
      </div>
      {activeTab === 'optimal' ? (
        <>
        <div className="portfolio-filter-row">
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
                {searchOpen && optimalSearchResults.length > 0 && (
                  <div className="portfolio-filter-results">
                    {optimalSearchResults.map(p => (
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
          {filterPlayerMissingFromOptimal && (
            <span className="portfolio-filter-empty">No optimal lineups include {filterPlayer!.name}.</span>
          )}
          <div className={`portfolio-optimal-banner${optimalInPortfolio.length > 0 ? ' portfolio-optimal-banner--hit' : ''}`}>
            <span>{optimalInPortfolio.length} / {optimalLineups.length} optimal lineup{optimalLineups.length !== 1 ? 's' : ''} selected in portfolio</span>
          </div>
        </div>
        <div className="portfolio-cards">
          {visibleOptimalLineups.map(ol => {
            const key = playerKey(ol.players)
            const portfolioIndex = portfolioKeyMap.get(key)!
            const portLineup = portfolioLineupByIndex.get(portfolioIndex)
            const sorted = sortAndAssignPositions(ol.players, platform)
            const stack = getStackNotation(ol.players)
            const entryText = portLineup ? entryInfoText(portLineup, platform) : null
            return (
              <div key={ol.lineup_index} className="lineup-card lineup-card--in-portfolio">
                <div className="lineup-card-header">
                  <span className="lineup-card-num">#{ol.lineup_index}</span>
                  <span className="lineup-card-salary">${ol.lineup_salary.toLocaleString()}</span>
                  <span className="lineup-card-opt-ref">Portfolio #{portfolioIndex}</span>
                  <div className="lineup-card-header-right">
                    {stack && <span className="lineup-card-stack">{stack}</span>}
                  </div>
                </div>
                {entryText && (
                  <div className="lineup-card-entry-info">{entryText}</div>
                )}
                <div className="lineup-card-players">
                  {sorted.map(({ player: p, displayPos }, i) => (
                    <div key={i} className="lineup-player">
                      <span className="lineup-player-pos">{displayPos}</span>
                      <span className="lineup-player-name">{p.name}</span>
                      <TeamBadge team={p.team} className="lineup-player-team" />
                      <span className="lineup-player-sal">${(p.salary / 1000).toFixed(1)}k</span>
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
        </>
      ) : (
      <>
      {hasSweep && (
        <div className="portfolio-risk-selector">
          {portfolioSweep.map(entry => {
            const isViewing = displayedRisk === entry.risk
            const isActive = activeRisk === entry.risk
            return (
              <div key={entry.risk} className="portfolio-risk-btn-group">
                <button
                  className={`portfolio-risk-btn${isViewing ? ' portfolio-risk-btn--viewing' : ''}${isActive ? ' portfolio-risk-btn--active-risk' : ''}`}
                  onClick={() => {
                    setViewingRisk(entry.risk === displayedRisk ? null : entry.risk)
                  }}
                >
                  {isActive && <span className="portfolio-risk-star">★ </span>}Risk {entry.risk}
                  {contestNormalized.size > 0 && (() => {
                    const { max, avg } = calcSweepStats(entry.lineups, contestNormalized)
                    return <span className="portfolio-risk-btn-stats">{avg.toFixed(1)} avg · {max.toFixed(1)} max</span>
                  })()}
                </button>
                {isViewing && !isActive && onActivateRisk && (
                  <button
                    className="portfolio-risk-activate-btn"
                    onClick={() => { onActivateRisk(entry.risk); setViewingRisk(null) }}
                    title="Set as active portfolio (writes output files)"
                  >
                    Set Active
                  </button>
                )}
              </div>
            )
          })}
        </div>
      )}
      <div className="portfolio-filter-row">
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
        {filterPlayerMissingFromRisk && (
          <span className="portfolio-filter-empty">No lineups include {filterPlayer!.name} at this risk level.</span>
        )}
        <div className={`portfolio-unconfirmed-banner ${totalUnconfirmed === 0 ? 'portfolio-unconfirmed-banner--clear' : ''}`}>
          {totalUnconfirmed === 0
            ? '✓ All lineup slots confirmed'
            : `✕ ${totalUnconfirmed} unconfirmed lineup slot${totalUnconfirmed !== 1 ? 's' : ''} across portfolio${breakdown}`}
        </div>
        <div className="portfolio-contest-controls">
          <button
            className="portfolio-analyze-btn"
            onClick={handleAnalyzeContest}
            disabled={contestLoading}
          >
            {contestLoading ? 'Loading…' : contestNormalized.size > 0 ? 'Re-analyze' : 'Analyze Contest'}
          </button>
          {contestNormalized.size > 0 && (
            <button
              className={`portfolio-analyze-btn${sortByActual ? ' portfolio-analyze-btn--active' : ''}`}
              onClick={() => setSortByActual(s => !s)}
            >
              {sortByActual ? 'Original Order' : 'Sort by Score ↓'}
            </button>
          )}
          {contestError && (
            <span className="portfolio-contest-error" onClick={() => setContestError(null)} title="Click to dismiss">
              ✕ {contestError}
            </span>
          )}
        </div>
      </div>
      <div className="portfolio-cards">
        {visibleLineups.map(lineup => {
          const sorted = sortAndAssignPositions(lineup.players, platform)
          const stack = getStackNotation(lineup.players)
          const isReplacing = replacingLineupIndex === lineup.lineup_index
          const optIdx = optimalIndexMap.get(playerKey(lineup.players))
          return (
            <div key={lineup.lineup_index} className="lineup-card">
              <div className="lineup-card-header">
                <span className="lineup-card-num">#{lineup.lineup_index}</span>
                <span className="lineup-card-salary">${lineup.lineup_salary.toLocaleString()}</span>
                {lineup.mean_ev != null && (
                  <span className="lineup-card-ev">${lineup.mean_ev.toFixed(1)}</span>
                )}
                {optIdx != null && (
                  <span className="lineup-card-opt-ref">Opt #{optIdx}</span>
                )}
                <div className="lineup-card-header-right">
                  {isReplacing ? (
                    <span className="lineup-card-generating">Generating…</span>
                  ) : (
                    <>
                      {stack && <span className="lineup-card-stack">{stack}</span>}
                      {isPrimary && onDeleteLineup && (
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
                  {entryInfoText(lineup, platform)}
                </div>
              )}
              {contestNormalized.size > 0 && (
                <div className="lineup-card-actual-score">
                  {calcLineupFpts(lineup, contestNormalized).toFixed(2)} FPTS
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
                        if (!unconfirmedSet.has(p.player_id)) {
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
      </>
      )}
    </div>
  )
}
