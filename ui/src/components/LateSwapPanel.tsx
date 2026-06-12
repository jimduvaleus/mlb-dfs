import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { LateSwapCandidate, LateSwapEntry, LateSwapSlot, LateSwapState, PlatformType } from '../types'
import { fetchLateSwapCandidates, fetchLateSwapState, overrideLateSwap, resetLateSwap, runLateSwap } from '../api'
import TeamBadge from './TeamBadge'

interface Props {
  platform?: PlatformType
}

interface FilterPlayer {
  player_id: number
  name: string
  position: string
  team: string
}

function slotCurrentPlayerId(slot: LateSwapSlot): number | null {
  if (slot.swapped_in) return slot.swapped_in.player_id
  return slot.player?.player_id ?? null
}

function entrySalary(entry: LateSwapEntry): number {
  let total = 0
  for (const s of entry.slots) {
    const p = s.swapped_in ?? s.player
    if (p?.salary != null) total += p.salary
  }
  return total
}

export default function LateSwapPanel({ platform }: Props) {
  const [state, setState] = useState<LateSwapState | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)

  // Marks (client-side until Run Swap is pressed)
  const [entryMarks, setEntryMarks] = useState<Map<string, Set<number>>>(new Map())
  const [bulkPids, setBulkPids] = useState<Set<number>>(new Set())
  const [bulkTeams, setBulkTeams] = useState<Set<string>>(new Set())

  // Filter by player
  const [filterPlayer, setFilterPlayer] = useState<FilterPlayer | null>(null)
  const [search, setSearch] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const searchWrapRef = useRef<HTMLDivElement>(null)

  // Bulk player mark search
  const [bulkSearch, setBulkSearch] = useState('')
  const [bulkSearchOpen, setBulkSearchOpen] = useState(false)
  const bulkSearchWrapRef = useRef<HTMLDivElement>(null)

  // Candidate override dropdown
  const [openDropdown, setOpenDropdown] = useState<{ entryId: string; slotIndex: number } | null>(null)
  const [candidates, setCandidates] = useState<LateSwapCandidate[]>([])
  const [candidatesLoading, setCandidatesLoading] = useState(false)
  const [overrideError, setOverrideError] = useState<string | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const applyState = useCallback((s: LateSwapState) => {
    setState(s)
    // Re-derive marks from server state so a re-run preserves prior swaps.
    const marks = new Map<string, Set<number>>()
    for (const e of s.entries) {
      const pids = new Set<number>()
      for (const slot of e.slots) {
        if (slot.swapped_in && slot.player?.player_id != null) pids.add(slot.player.player_id)
      }
      if (pids.size > 0) marks.set(e.entry_id, pids)
    }
    setEntryMarks(marks)
    setBulkPids(new Set(s.bulk_marked_player_ids))
    setBulkTeams(new Set(s.bulk_marked_teams))
  }, [])

  useEffect(() => {
    setLoading(true)
    fetchLateSwapState()
      .then(applyState)
      .catch(err => setError(err instanceof Error ? err.message : 'Failed to load'))
      .finally(() => setLoading(false))
  }, [applyState])

  // Close popovers on outside click
  useEffect(() => {
    const onMouseDown = (e: MouseEvent) => {
      if (searchWrapRef.current && !searchWrapRef.current.contains(e.target as Node)) setSearchOpen(false)
      if (bulkSearchWrapRef.current && !bulkSearchWrapRef.current.contains(e.target as Node)) setBulkSearchOpen(false)
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) setOpenDropdown(null)
    }
    document.addEventListener('mousedown', onMouseDown)
    return () => document.removeEventListener('mousedown', onMouseDown)
  }, [])

  const allPlayers = useMemo<FilterPlayer[]>(() => {
    if (!state) return []
    const seen = new Map<number, FilterPlayer>()
    for (const e of state.entries) {
      for (const s of e.slots) {
        for (const p of [s.player, s.swapped_in]) {
          if (p?.player_id != null && !seen.has(p.player_id)) {
            seen.set(p.player_id, { player_id: p.player_id, name: p.name, position: p.position, team: p.team })
          }
        }
      }
    }
    return [...seen.values()]
  }, [state])

  const swappablePlayers = useMemo<FilterPlayer[]>(() => {
    if (!state) return []
    const seen = new Map<number, FilterPlayer>()
    for (const e of state.entries) {
      for (const s of e.slots) {
        const p = s.player
        if (!s.locked && p?.player_id != null && !seen.has(p.player_id)) {
          seen.set(p.player_id, { player_id: p.player_id, name: p.name, position: p.position, team: p.team })
        }
      }
    }
    return [...seen.values()]
  }, [state])

  const searchResults = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return []
    return allPlayers.filter(p => p.name.toLowerCase().includes(q)).slice(0, 10)
  }, [search, allPlayers])

  const bulkSearchResults = useMemo(() => {
    const q = bulkSearch.trim().toLowerCase()
    if (!q) return []
    return swappablePlayers.filter(p => p.name.toLowerCase().includes(q) && !bulkPids.has(p.player_id)).slice(0, 10)
  }, [bulkSearch, swappablePlayers, bulkPids])

  const bulkPidPlayers = useMemo(
    () => [...bulkPids].map(pid => allPlayers.find(p => p.player_id === pid) ?? { player_id: pid, name: `#${pid}`, position: '', team: '' }),
    [bulkPids, allPlayers],
  )

  const isMarked = useCallback((entry: LateSwapEntry, slot: LateSwapSlot): boolean => {
    if (slot.locked || !slot.player || slot.player.player_id == null) return false
    const pid = slot.player.player_id
    if (entryMarks.get(entry.entry_id)?.has(pid)) return true
    if (bulkPids.has(pid)) return true
    if (slot.player.team && bulkTeams.has(slot.player.team)) return true
    return false
  }, [entryMarks, bulkPids, bulkTeams])

  const isBulkMarked = useCallback((slot: LateSwapSlot): boolean => {
    const pid = slot.player?.player_id
    if (pid == null) return false
    return bulkPids.has(pid) || (!!slot.player?.team && bulkTeams.has(slot.player.team))
  }, [bulkPids, bulkTeams])

  const visibleEntries = useMemo(() => {
    if (!state) return []
    let entries = state.entries.filter(e => e.n_swappable > 0)
    if (filterPlayer) {
      entries = entries.filter(e =>
        e.slots.some(s => slotCurrentPlayerId(s) === filterPlayer.player_id || s.player?.player_id === filterPlayer.player_id))
    }
    return entries
  }, [state, filterPlayer])

  const markedCount = useMemo(() => {
    if (!state) return 0
    let n = 0
    for (const e of state.entries) {
      for (const s of e.slots) {
        if (isMarked(e, s)) n++
        else if (!s.locked && s.player === null) n++ // empty cell: implicitly swapped
      }
    }
    return n
  }, [state, isMarked])

  const totalSwappable = useMemo(
    () => state ? state.entries.reduce((n, e) => n + e.n_swappable, 0) : 0,
    [state],
  )

  const toggleMark = (entry: LateSwapEntry, slot: LateSwapSlot) => {
    const pid = slot.player?.player_id
    if (pid == null || slot.locked || isBulkMarked(slot)) return
    setEntryMarks(prev => {
      const next = new Map(prev)
      const set = new Set(next.get(entry.entry_id) ?? [])
      if (set.has(pid)) set.delete(pid)
      else set.add(pid)
      if (set.size > 0) next.set(entry.entry_id, set)
      else next.delete(entry.entry_id)
      return next
    })
  }

  const handleRun = async () => {
    setRunning(true)
    setError(null)
    try {
      const result = await runLateSwap({
        entry_marks: Object.fromEntries([...entryMarks].map(([k, v]) => [k, [...v]])),
        bulk_marked_player_ids: [...bulkPids],
        bulk_marked_teams: [...bulkTeams],
      })
      applyState(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Swap failed')
    } finally {
      setRunning(false)
    }
  }

  const handleReset = async () => {
    setRunning(true)
    setError(null)
    try {
      applyState(await resetLateSwap())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Reset failed')
    } finally {
      setRunning(false)
    }
  }

  const openCandidates = async (entryId: string, slotIndex: number) => {
    if (openDropdown?.entryId === entryId && openDropdown?.slotIndex === slotIndex) {
      setOpenDropdown(null)
      return
    }
    setOpenDropdown({ entryId, slotIndex })
    setCandidates([])
    setOverrideError(null)
    setCandidatesLoading(true)
    try {
      const resp = await fetchLateSwapCandidates(entryId, slotIndex)
      setCandidates(resp.candidates)
    } catch (err) {
      setOverrideError(err instanceof Error ? err.message : 'Failed to load candidates')
    } finally {
      setCandidatesLoading(false)
    }
  }

  const handleOverride = async (entryId: string, slotIndex: number, playerId: number) => {
    setOverrideError(null)
    try {
      const resp = await overrideLateSwap(entryId, slotIndex, playerId)
      setState(prev => prev ? {
        ...prev,
        written_files: resp.written_files,
        entries: prev.entries.map(e => e.entry_id === entryId ? resp.entry : e),
      } : prev)
      setOpenDropdown(null)
    } catch (err) {
      setOverrideError(err instanceof Error ? err.message : 'Override failed')
    }
  }

  if (loading) return <p className="muted">Loading entries…</p>

  if (platform === 'fanduel' || state?.status === 'unsupported_platform') {
    return <p className="muted">Late swap is only available for DraftKings slates.</p>
  }
  if (error && !state) return <p className="error">{error}</p>
  if (!state) return null
  if (state.status === 'no_slate') {
    return <p className="muted">No slate file configured — set the DK salary CSV path on the Config tab.</p>
  }
  if (state.status === 'no_entries') {
    return <p className="muted">No entry files found in outputs/ — run a portfolio (which writes upload_*.csv), or place a DK entries CSV there.</p>
  }

  const hasSwaps = state.entries.some(e => e.slots.some(s => s.swapped_in))
  const allLocked = totalSwappable === 0

  return (
    <div className="lateswap-wrap">
      <div className="lateswap-toolbar">
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
                      onMouseDown={e => { e.preventDefault(); setFilterPlayer(p); setSearch(''); setSearchOpen(false) }}
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

        <div className="portfolio-filter" ref={bulkSearchWrapRef}>
          <input
            className="portfolio-filter-input"
            placeholder="Mark player in all lineups…"
            value={bulkSearch}
            onChange={e => { setBulkSearch(e.target.value); setBulkSearchOpen(true) }}
            onFocus={() => setBulkSearchOpen(true)}
          />
          {bulkSearchOpen && bulkSearchResults.length > 0 && (
            <div className="portfolio-filter-results">
              {bulkSearchResults.map(p => (
                <button
                  key={p.player_id}
                  className="portfolio-filter-result-btn"
                  onMouseDown={e => {
                    e.preventDefault()
                    setBulkPids(prev => new Set(prev).add(p.player_id))
                    setBulkSearch('')
                    setBulkSearchOpen(false)
                  }}
                >
                  <span>{p.name}</span>
                  <span className="portfolio-filter-result-meta">{p.position} · {p.team}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <select
          className="lateswap-team-select"
          value=""
          onChange={e => {
            const team = e.target.value
            if (team) setBulkTeams(prev => new Set(prev).add(team))
          }}
        >
          <option value="">Mark team in all lineups…</option>
          {state.teams.filter(t => !bulkTeams.has(t)).map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>

        <span className="lateswap-summary">
          {visibleEntries.length} lineup{visibleEntries.length !== 1 ? 's' : ''} · {totalSwappable} swappable · {markedCount} marked
        </span>

        <div className="lateswap-toolbar-actions">
          {hasSwaps && (
            <button className="btn-secondary" onClick={handleReset} disabled={running}>
              Reset
            </button>
          )}
          <button
            className="btn-run"
            onClick={handleRun}
            disabled={running || markedCount === 0}
          >
            {running ? 'Swapping…' : 'Run Swap'}
          </button>
        </div>
      </div>

      {(bulkPidPlayers.length > 0 || bulkTeams.size > 0) && (
        <div className="lateswap-bulk-chips">
          {bulkPidPlayers.map(p => (
            <span key={p.player_id} className="portfolio-filter-chip lateswap-bulk-chip">
              ✗ {p.name}
              <button onClick={() => setBulkPids(prev => { const n = new Set(prev); n.delete(p.player_id); return n })}>×</button>
            </span>
          ))}
          {[...bulkTeams].map(t => (
            <span key={t} className="portfolio-filter-chip lateswap-bulk-chip">
              ✗ <TeamBadge team={t} />
              <button onClick={() => setBulkTeams(prev => { const n = new Set(prev); n.delete(t); return n })}>×</button>
            </span>
          ))}
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {state.written_files.length > 0 && (
        <div className="lateswap-banner">
          ✓ Swap files written: {state.written_files.map(f => f.split('/').pop()).join(', ')}
          {state.last_run_at && <span className="lateswap-banner-time"> (run {state.last_run_at.replace('T', ' ')})</span>}
        </div>
      )}

      {allLocked && (
        <p className="muted">All games have started — nothing can be swapped.</p>
      )}
      {!allLocked && visibleEntries.length === 0 && (
        <p className="muted">
          {filterPlayer
            ? `No swappable lineups include ${filterPlayer.name}.`
            : 'No entries with swappable players.'}
        </p>
      )}

      <div className="portfolio-cards">
        {visibleEntries.map(entry => {
          const entryWarnings = entry.warnings.filter(w => w.slot_index === null)
          const slotWarnings = new Set(entry.warnings.filter(w => w.slot_index !== null).map(w => w.slot_index))
          return (
            <div key={`${entry.source_file}:${entry.entry_id}`} className="lineup-card">
              <div className="lineup-card-header">
                <span className="lineup-card-num">#{entry.entry_id}</span>
                <span className="lineup-card-salary">${entrySalary(entry).toLocaleString()}</span>
                <div className="lineup-card-header-right">
                  <span className="lateswap-card-swappable">{entry.n_swappable} open</span>
                </div>
              </div>
              <div className="lineup-card-entry-info">
                {entry.source_file} · {entry.entry_fee} · {entry.contest_name}
              </div>
              {entryWarnings.length > 0 && (
                <div className="lateswap-card-warning">⚠ {entryWarnings.map(w => w.reason).join(', ')}</div>
              )}
              <div className="lineup-card-players">
                {entry.slots.map(slot => {
                  const marked = isMarked(entry, slot)
                  const bulkMarked = isBulkMarked(slot)
                  const swapped = slot.swapped_in
                  const isOpen = openDropdown?.entryId === entry.entry_id && openDropdown?.slotIndex === slot.slot_index
                  const rowClass = slot.locked
                    ? 'lineup-player lineup-player--locked'
                    : swapped
                      ? 'lineup-player lineup-player--swapped'
                      : marked
                        ? 'lineup-player lineup-player--marked'
                        : 'lineup-player'
                  return (
                    <div key={slot.slot_index} className={rowClass}>
                      <span className="lineup-player-pos">{slot.slot_position}</span>
                      {swapped ? (
                        <span className="lineup-player-name lateswap-swapped-cell">
                          <button
                            className="lateswap-swapped-name"
                            onClick={() => !swapped.locked && openCandidates(entry.entry_id, slot.slot_index)}
                            disabled={swapped.locked}
                            title={swapped.locked ? 'Game has started' : 'Choose a different replacement'}
                          >
                            {swapped.name}
                          </button>
                          {slot.swap_source === 'manual' && <span className="lateswap-manual-badge">manual</span>}
                          <span className="lateswap-was">was {slot.player ? slot.player.name : '(empty)'}</span>
                          {isOpen && (
                            <div className="swap-candidates-dropdown" ref={dropdownRef}>
                              {candidatesLoading && <div className="swap-candidates-empty">Loading…</div>}
                              {overrideError && <div className="swap-candidates-error">{overrideError}</div>}
                              {!candidatesLoading && !overrideError && candidates.length === 0 && (
                                <div className="swap-candidates-empty">No eligible replacements</div>
                              )}
                              {candidates.map(c => (
                                <button
                                  key={c.player_id}
                                  className="swap-candidate-btn"
                                  disabled={c.player_id === swapped.player_id}
                                  onClick={() => handleOverride(entry.entry_id, slot.slot_index, c.player_id)}
                                >
                                  <span className="swap-candidate-name">{c.name}</span>
                                  <span className="swap-candidate-meta">
                                    {c.position} · {c.team} · ${(c.salary / 1000).toFixed(1)}k{c.mean != null ? ` · ${c.mean.toFixed(1)} pts` : ''}
                                  </span>
                                </button>
                              ))}
                            </div>
                          )}
                        </span>
                      ) : (
                        <span className="lineup-player-name">
                          {slot.player ? slot.player.name : <em className="lateswap-empty-slot">(empty)</em>}
                          {slotWarnings.has(slot.slot_index) && (
                            <span className="lateswap-slot-warning" title="No valid replacement found — original kept">⚠</span>
                          )}
                        </span>
                      )}
                      {(swapped ?? slot.player) && (swapped ?? slot.player)!.team
                        ? <TeamBadge team={(swapped ?? slot.player)!.team} className="lineup-player-team" />
                        : <span />}
                      <span className="lineup-player-sal lateswap-sal-cell">
                        {(() => {
                          const p = swapped ?? slot.player
                          return p?.salary != null ? `$${(p.salary / 1000).toFixed(1)}k` : '—'
                        })()}
                        {slot.locked ? (
                          <span className="lateswap-lock" title={slot.missing_from_slate ? 'Not on the current slate' : 'Game has started'}>🔒</span>
                        ) : (
                          <button
                            className={`lineup-player-swap-toggle${marked ? ' lineup-player-swap-toggle--on' : ''}`}
                            onClick={() => toggleMark(entry, slot)}
                            disabled={bulkMarked || (slot.player == null)}
                            title={
                              slot.player == null
                                ? 'Empty slot — filled automatically on swap'
                                : bulkMarked
                                  ? 'Marked by a bulk player/team action — unmark via the toolbar chips'
                                  : marked ? 'Unmark' : 'Mark for swap'
                            }
                          >
                            ✗
                          </button>
                        )}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
