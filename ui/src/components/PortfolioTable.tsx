import { useState, useRef, useEffect } from 'react'
import type { ContestNameCollision, LineupResult, PlatformType, PlayerRow, PortfolioSweepEntry } from '../types'
import { fetchContestAnalysis } from '../api'
import { getStackNotation, alphabetizeDuplicateGroups } from '../utils'
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
  evwBase?: number
  evwMax?: number
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
// The backend already computes the true per-player roster slot during
// optimization (the same assignment used for the upload CSV / late swap) and
// serializes it as `assigned_position`. We use that directly when present so
// the Portfolio panel can't diverge from the upload file. The bipartite-match
// recompute below (parsing eligible positions from p.position, e.g. "2B/SS")
// is only a fallback for older cached data that predates `assigned_position`.
function sortAndAssignPositions(
  players: PlayerRow[],
  platform?: PlatformType,
): Array<{ player: PlayerRow; displayPos: string }> {
  const pitchers = players.filter(p => p.position === 'P')
  const batters  = players.filter(p => p.position !== 'P')

  const posOrder = platform === 'fanduel'
    ? ['C/1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'UTIL']
    : ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']

  // assigned_position is computed backend-side in DK slot space (P/C/1B/2B/3B/SS/OF)
  // — it doesn't speak FanDuel's compound slot labels (C/1B, UTIL), which come from
  // a separate matching pass in fd_entries.py. Only trust it for DraftKings.
  if (platform !== 'fanduel' && batters.every(p => !!p.assigned_position)) {
    const byPos = new Map<string, PlayerRow[]>()
    for (const p of batters) {
      const key = p.assigned_position!
      const arr = byPos.get(key)
      if (arr) arr.push(p)
      else byPos.set(key, [p])
    }
    const result: Array<{ player: PlayerRow; displayPos: string }> = [
      ...pitchers.map(p => ({ player: p, displayPos: 'P' })),
    ]
    // Walk the canonical slot order so cards always render P, C, 1B, 2B, 3B,
    // SS, OF, OF, OF regardless of the players' original array order.
    for (const slot of posOrder) {
      const player = byPos.get(slot)?.shift()
      if (player) result.push({ player, displayPos: slot })
    }
    // Safety valve: any player whose assigned_position wasn't in posOrder
    // (shouldn't occur) still gets rendered instead of silently dropped.
    for (const leftover of byPos.values()) {
      for (const player of leftover) result.push({ player, displayPos: player.assigned_position! })
    }
    return alphabetizeDuplicateGroups(result, r => r.displayPos, r => r.player.name)
  }

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

  // DK echoes duplicate-position roster slots (P,P and OF,OF,OF) back
  // alphabetically by last name regardless of upload column order (confirmed
  // empirically by diffing an uploaded vs. downloaded entries CSV) — mirror
  // that convention here so cards match what DK itself will display.
  return alphabetizeDuplicateGroups(result, r => r.displayPos, r => r.player.name)
}

function buildNormalizedFptsMap(fpts: Record<string, number>): Map<string, number> {
  const m = new Map<string, number>()
  for (const [name, val] of Object.entries(fpts)) {
    m.set(name.toLowerCase(), val)
  }
  return m
}

// `overrides` resolves players caught in a contest name collision (e.g. two real
// players both named "Max Muncy") to their correct FPTS by player_id, since the
// plain name map can only hold one value per name. See buildFptsOverrides.
function lookupFpts(player: PlayerRow, normalized: Map<string, number>, overrides: Map<number, number>): number {
  const override = overrides.get(player.player_id)
  if (override != null) return override
  return normalized.get(player.name.toLowerCase()) ?? 0
}

function calcLineupFpts(lineup: LineupResult, normalized: Map<string, number>, overrides: Map<number, number>): number {
  return lineup.players.reduce((sum, p) => sum + lookupFpts(p, normalized, overrides), 0)
}

function calcSweepStats(lineups: LineupResult[], norm: Map<string, number>, overrides: Map<number, number>) {
  const scores = lineups.map(l => calcLineupFpts(l, norm, overrides))
  return {
    max: Math.max(...scores),
    avg: scores.reduce((a, b) => a + b, 0) / scores.length,
  }
}

// Builds a player_id → FPTS map for players caught in a contest name collision.
// The contest export has no team/ID for its FPTS sidebar table, so the server can only
// suggest which candidate maps to which value; `swapped` holds names the user has
// flipped away from that suggestion (2-candidate collisions only).
function buildFptsOverrides(collisions: ContestNameCollision[], swapped: Set<string>): Map<number, number> {
  const overrides = new Map<number, number>()
  for (const collision of collisions) {
    const candidates = swapped.has(collision.name) && collision.candidates.length === 2
      ? [collision.candidates[1], collision.candidates[0]]
      : collision.candidates
    collision.candidates.forEach((cand, i) => {
      const fpts = candidates[i].fpts
      if (fpts != null) overrides.set(cand.player_id, fpts)
    })
  }
  return overrides
}

function parseFeeCents(entryFee: string | null | undefined): number {
  if (!entryFee) return 0
  return Math.round(parseFloat(entryFee.replace(/[^0-9.]/g, '')) * 100)
}

// Portfolio-average projected score and summed ownership. Lineups missing
// either total are skipped for that average rather than counted as zero.
function calcAvgMeanOwn(lineups: LineupResult[]): { mean: number | null; own: number | null } {
  const pick = (f: (l: LineupResult) => number | null | undefined) => {
    const v = lineups.map(f).filter((x): x is number => x != null)
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
  }
  return { mean: pick(l => l.lineup_mean), own: pick(l => l.lineup_ownership) }
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

// Mirrors DeterminantPortfolioSelector.evw_for_risk in gpp_portfolio.py:
// EVw = evwBase at risk=1, evwMax at risk=5; linear in between.
function evwForRisk(risk: number, evwBase: number, evwMax: number): number {
  const t = (risk - 1) / 4
  return Math.min(Math.max(evwBase + t * (evwMax - evwBase), 0), 1)
}

export function PortfolioTable({ lineups, optimalLineups = [], portfolioSweep = [], activeRisk = 1, onActivateRisk, unconfirmedPlayerIds, onDeleteLineup, replacingLineupIndex, platform, evwBase = 0.10, evwMax = 0.40 }: Props) {
  const [activeTab, setActiveTab] = useState<'portfolio' | 'optimal'>('portfolio')
  // viewingRisk: which risk the user is currently browsing (null = showing active)
  const [viewingRisk, setViewingRisk] = useState<number | null>(null)
  // Shared filter across all risk levels and both tabs
  const [filterPlayer, setFilterPlayer] = useState<PlayerRow | null>(null)
  const [filterUnconfirmed, setFilterUnconfirmed] = useState(false)
  const [search, setSearch] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const searchWrapRef = useRef<HTMLDivElement>(null)
  // Contest analysis state
  const [contestNormalized, setContestNormalized] = useState<Map<string, number>>(new Map())
  const [contestCollisions, setContestCollisions] = useState<ContestNameCollision[]>([])
  const [swappedCollisions, setSwappedCollisions] = useState<Set<string>>(new Set())
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
      setContestCollisions(result.collisions)
      setSwappedCollisions(new Set())
    } catch (e) {
      setContestError(e instanceof Error ? e.message : String(e))
    } finally {
      setContestLoading(false)
    }
  }

  if (lineups.length === 0) return null

  const fptsOverrides = buildFptsOverrides(contestCollisions, swappedCollisions)

  // Derive displayed lineups. viewingRisk=null shows the active portfolio (state.portfolio).
  // A single-entry sweep (self_play: no risk/diversity sweep, one portfolio
  // only — see pipeline.py::_run_external's self_play branch) has nothing
  // to select between, so the risk-selector strip stays hidden rather than
  // rendering a pointless one-button row.
  const hasSweep = portfolioSweep.length > 1
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
  const playerFilteredLineups = filterPlayer
    ? sortedActiveLineups.filter(l => l.players.some(p => p.player_id === filterPlayer.player_id))
    : sortedActiveLineups
  const filterPlayerMissingFromRisk = filterPlayer !== null && playerFilteredLineups.length === 0

  const unconfirmedSet = new Set(unconfirmedPlayerIds ?? [])
  const lineupHasUnconfirmed = (l: LineupResult) => l.players.some(p => unconfirmedSet.has(p.player_id))
  // If every player gets confirmed while the filter is on, the filter stops
  // applying (and its toggle disappears), so all lineups show again.
  const showUnconfirmedOnly = filterUnconfirmed && playerFilteredLineups.some(lineupHasUnconfirmed)
  const filteredLineups = showUnconfirmedOnly
    ? playerFilteredLineups.filter(lineupHasUnconfirmed)
    : playerFilteredLineups
  const visibleLineups = (contestNormalized.size > 0 && sortByActual)
    ? [...filteredLineups].sort((a, b) => calcLineupFpts(b, contestNormalized, fptsOverrides) - calcLineupFpts(a, contestNormalized, fptsOverrides))
    : filteredLineups

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
  // Share of the shipped portfolio that came from a generated source rather
  // than the imported pool. Only rendered when something was generated -- on
  // every other allocator the flag is absent and the counter would read a
  // meaningless "0 of N".
  const nGenerated = activeLineups.filter(l => l.from_generated).length
  const genPct = activeLineups.length > 0
    ? Math.round((nGenerated / activeLineups.length) * 100) : 0
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
    : `Portfolio — ${(filterPlayer || showUnconfirmedOnly) ? `${visibleLineups.length} / ${activeLineups.length}` : activeLineups.length} Lineup${activeLineups.length !== 1 ? 's' : ''}`

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
        {nGenerated > 0 && activeTab === 'portfolio' && (
          <span
            className="portfolio-gen-summary"
            title="Lineups built by the marginal-reward frontier generator rather than imported from the SaberSim pool"
          >
            <span className="lineup-gen-badge">GEN</span>
            {nGenerated} of {activeLineups.length} ({genPct}%)
          </span>
        )}
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
                  <span className="portfolio-risk-btn-stats">EVw {evwForRisk(entry.risk, evwBase, evwMax).toFixed(3)}</span>
                  {(() => {
                    // Average projected score / ownership across the tier, so
                    // risk tiers are comparable on composition rather than on
                    // the selector's own EV currency.
                    const { mean, own } = calcAvgMeanOwn(entry.lineups)
                    return (mean != null || own != null) && (
                      <span className="portfolio-risk-btn-stats">
                        {[mean != null ? `${mean.toFixed(1)} mean` : null,
                          own != null ? `${own.toFixed(1)} own` : null]
                          .filter(Boolean).join(' · ')}
                      </span>
                    )
                  })()}
                  {contestNormalized.size > 0 && (() => {
                    const { max, avg } = calcSweepStats(entry.lineups, contestNormalized, fptsOverrides)
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
          {totalUnconfirmed > 0 && (
            <button
              className={`portfolio-unconfirmed-filter-btn${showUnconfirmedOnly ? ' portfolio-unconfirmed-filter-btn--active' : ''}`}
              onClick={() => setFilterUnconfirmed(f => !f)}
              title={showUnconfirmedOnly ? 'Show all lineups' : 'Show only lineups with unconfirmed players'}
            >
              {showUnconfirmedOnly ? 'Show all' : 'Filter'}
            </button>
          )}
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
      {contestCollisions.length > 0 && (
        <div className="portfolio-collision-banner">
          {contestCollisions.map(collision => {
            const swapped = swappedCollisions.has(collision.name)
            const ordered = swapped && collision.candidates.length === 2
              ? [collision.candidates[1], collision.candidates[0]]
              : collision.candidates
            return (
              <div key={collision.name} className="portfolio-collision-row">
                <span className="portfolio-collision-name">⚠ {collision.name} is ambiguous —</span>
                {collision.candidates.map((cand, i) => (
                  <span key={cand.player_id} className="portfolio-collision-candidate">
                    {cand.team} ${(cand.salary / 1000).toFixed(1)}k → {ordered[i].fpts ?? '?'} FPTS
                  </span>
                ))}
                {collision.candidates.length === 2 && (
                  <button
                    className="portfolio-collision-swap-btn"
                    onClick={() => setSwappedCollisions(prev => {
                      const next = new Set(prev)
                      if (next.has(collision.name)) next.delete(collision.name)
                      else next.add(collision.name)
                      return next
                    })}
                    title="The contest export doesn't say which player each value belongs to — swap if this guess looks wrong"
                  >
                    Swap
                  </button>
                )}
              </div>
            )
          })}
        </div>
      )}
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
                {/* Projected score and summed ownership rather than the
                    selector's EV — what the lineup is actually made of, so
                    the selector's choices can be read directly. */}
                {lineup.lineup_mean != null && (
                  <span className="lineup-card-ev" title="Sum of projected scores">
                    {`PRJ ${Math.round(lineup.lineup_mean)}`}
                  </span>
                )}
                {lineup.lineup_ownership != null && (
                  <span className="lineup-card-ev" title="Sum of projected ownership (percentage points)">
                    {`OWN ${Math.round(lineup.lineup_ownership)}`}
                  </span>
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
              {(lineup.upload_tag || lineup.from_generated) && (
                <div className="lineup-card-entry-info">
                  {entryInfoText(lineup, platform)}
                  {lineup.from_generated && (
                    <span
                      className="lineup-gen-badge"
                      title="Generated by the marginal-reward line-2 frontier, not imported from the SaberSim pool"
                    >
                      GEN
                    </span>
                  )}
                </div>
              )}
              {contestNormalized.size > 0 && (
                <div className="lineup-card-actual-score">
                  {calcLineupFpts(lineup, contestNormalized, fptsOverrides).toFixed(2)} FPTS
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
