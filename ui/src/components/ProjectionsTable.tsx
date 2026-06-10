import React, { useState, useEffect, useCallback } from 'react'
import type { ProjectionPlayerRow, PlatformType, TwitterLineupRecord } from '../types'
import { fetchTeamOwnershipReductions, saveTeamOwnershipReductions, fetchPlayerProjectionOverrides, savePlayerProjectionOverrides } from '../api'
import TeamBadge from './TeamBadge'

interface Props {
  players: ProjectionPlayerRow[]
  platform?: PlatformType
  teamTotals?: Record<string, number>
  onOwnershipSettingsChanged?: () => void
  twitterLineups?: TwitterLineupRecord[]
  onLockToggle?: (team: string, locked: boolean) => void
  onRefresh?: (team: string) => void
}

export function ProjectionsTable({ players, teamTotals, onOwnershipSettingsChanged, twitterLineups = [], onLockToggle, onRefresh }: Props) {
  const [reductionSlateId, setReductionSlateId] = useState<string>('')
  const [teamReductions, setTeamReductions] = useState<Record<string, string>>({})
  const [reductionSaving, setReductionSaving] = useState(false)

  const [overrideSlateId, setOverrideSlateId] = useState<string>('')
  const [projOverrides, setProjOverrides] = useState<Record<number, number>>({})
  const [editingProj, setEditingProj] = useState<Record<number, string>>({})
  const [overrideSaving, setOverrideSaving] = useState(false)
  const [refreshingTeam, setRefreshingTeam] = useState<string | null>(null)

  useEffect(() => {
    fetchTeamOwnershipReductions().then(r => {
      setReductionSlateId(prev => {
        // Only overwrite local reductions when the slate genuinely changes.
        // Frequent player refreshes within the same slate must not clobber state.
        if (r.slate_id !== prev) {
          const s: Record<string, string> = {}
          for (const [t, v] of Object.entries(r.team_ownership_reductions)) {
            if (v > 0) s[t] = String(v)
          }
          setTeamReductions(s)
        }
        return r.slate_id
      })
    }).catch(() => {})

    fetchPlayerProjectionOverrides().then(r => {
      setOverrideSlateId(prev => {
        if (r.slate_id !== prev) {
          const overrides: Record<number, number> = {}
          for (const [k, v] of Object.entries(r.player_projection_overrides)) {
            overrides[Number(k)] = v
          }
          setProjOverrides(overrides)
        }
        return r.slate_id
      })
    }).catch(() => {})
  }, [players])

  const persistReductions = useCallback(async () => {
    if (reductionSaving) return
    setReductionSaving(true)
    const parsed: Record<string, number> = {}
    for (const [t, v] of Object.entries(teamReductions)) {
      const n = parseFloat(v)
      if (!isNaN(n) && n >= 1 && n <= 99) parsed[t] = n
    }
    try {
      const r = await saveTeamOwnershipReductions({ slate_id: reductionSlateId, team_ownership_reductions: parsed })
      setReductionSlateId(r.slate_id)
      onOwnershipSettingsChanged?.()
    } catch {}
    finally { setReductionSaving(false) }
  }, [reductionSlateId, teamReductions, reductionSaving, onOwnershipSettingsChanged])

  const persistOverride = useCallback(async (playerId: number) => {
    if (overrideSaving) return
    const draft = editingProj[playerId]
    if (draft === undefined) return
    const val = parseFloat(draft)
    setEditingProj(prev => { const n = { ...prev }; delete n[playerId]; return n })
    if (isNaN(val) || val <= 0) return
    setOverrideSaving(true)
    try {
      const next = { ...projOverrides, [playerId]: val }
      const r = await savePlayerProjectionOverrides({ slate_id: overrideSlateId, player_projection_overrides: next })
      setOverrideSlateId(r.slate_id)
      setProjOverrides(r.player_projection_overrides as Record<number, number>)
      onOwnershipSettingsChanged?.()
    } catch {}
    finally { setOverrideSaving(false) }
  }, [overrideSlateId, projOverrides, editingProj, overrideSaving, onOwnershipSettingsChanged])

  const resetOverride = useCallback(async (playerId: number) => {
    if (overrideSaving) return
    setOverrideSaving(true)
    try {
      const next = { ...projOverrides }
      delete next[playerId]
      const r = await savePlayerProjectionOverrides({ slate_id: overrideSlateId, player_projection_overrides: next })
      setOverrideSlateId(r.slate_id)
      setProjOverrides(r.player_projection_overrides as Record<number, number>)
      onOwnershipSettingsChanged?.()
    } catch {}
    finally { setOverrideSaving(false) }
  }, [overrideSlateId, projOverrides, overrideSaving, onOwnershipSettingsChanged])

  const handleRefreshTeam = useCallback(async (team: string) => {
    setRefreshingTeam(team)
    try {
      await onRefresh?.(team)
    } finally {
      setRefreshingTeam(null)
    }
  }, [onRefresh])

  if (players.length === 0) {
    return (
      <div className="projections-table-wrap">
        <p className="muted">No projections available. Fetch projections on the Config tab.</p>
      </div>
    )
  }

  const byTeam = new Map<string, ProjectionPlayerRow[]>()
  for (const p of players) {
    if (!byTeam.has(p.team)) byTeam.set(p.team, [])
    byTeam.get(p.team)!.push(p)
  }
  const teams = [...byTeam.keys()].sort()

  const lockedTeamCount = teams.filter(team => twitterLineups.find(l => l.team === team && l.locked)).length
  const unlockedTeamCount = teams.length - lockedTeamCount

  return (
    <div className="projections-table-wrap">
      <h3>Projections — {players.length} players, {teams.length} teams{unlockedTeamCount > 0 ? <span className="projections-unlocked-count"> · {unlockedTeamCount} unlocked</span> : null}</h3>
      <div className="portfolio-cards">
        {teams.map(team => {
          const teamPlayers = byTeam.get(team)!
          const pitchers = teamPlayers.filter(p => p.position === 'P')
          const batters  = teamPlayers
            .filter(p => p.position !== 'P')
            .sort((a, b) => (a.slot ?? 99) - (b.slot ?? 99))
          const hitterProj = batters.reduce((sum, p) => sum + p.mean, 0)
          const hasReduction = parseFloat(teamReductions[team] ?? '0') > 0

          const teamRecord = twitterLineups.find(l => l.team === team) ?? null
          const isLocked = teamRecord?.locked ?? false
          const isRefreshing = refreshingTeam === team

          // For locked lineups, build the batter rows from the confirmed slot order.
          // Slots with player_id=null are rendered as placeholders (player N/A).
          // Slot numbers missing from the record entirely (gap) also render as placeholders.
          const renderLockedBatterRows = (): React.ReactElement[] => {
            if (!teamRecord) return []
            const slotMap = new Map(teamRecord.slots.map(s => [s.slot, s]))
            const rows: React.ReactElement[] = []
            for (let slotNum = 1; slotNum <= 9; slotNum++) {
              const slotEntry = slotMap.get(slotNum)
              if (!slotEntry) {
                // Gap: this slot number is absent from the saved record
                rows.push(
                  <div key={`gap-${slotNum}`} className="lineup-player lineup-player--placeholder">
                    <span className="lineup-player-pos">—</span>
                    <span className="lineup-player-name">
                      <span className="batting-slot-bubble batting-slot-bubble--confirmed" title="Confirmed lineup slot">{slotNum}</span>
                    </span>
                    <TeamBadge team={team} className="lineup-player-team" />
                    <span className="lineup-player-sal lineup-player-sal--placeholder">—</span>
                    <span className="lineup-player-proj">0.0</span>
                    <span className="lineup-player-not-in-slate" title="Player confirmed in lineup but not in the DK salary file">N/A</span>
                  </div>
                )
                continue
              }
              if (slotEntry.player_id !== null) {
                const p = batters.find(b => b.player_id === slotEntry.player_id)
                if (p) {
                  const draftVal = editingProj[p.player_id]
                  const displayVal = draftVal !== undefined ? draftVal : p.mean.toFixed(1)
                  rows.push(
                    <div key={`slot-${slotEntry.slot}`} className="lineup-player">
                      <span className="lineup-player-pos">{p.position}</span>
                      <span className="lineup-player-name">
                        {p.name}
                        <span className="batting-slot-bubble batting-slot-bubble--confirmed" title="Confirmed lineup slot">{slotEntry.slot}</span>
                      </span>
                      <TeamBadge team={p.team} className="lineup-player-team" />
                      <span className="lineup-player-sal">${(p.salary / 1000).toFixed(1)}k</span>
                      <span className={`lineup-player-proj${p.is_overridden ? ' lineup-player-proj--overridden' : ''}`}>
                        <input
                          className="proj-override-input"
                          type="number"
                          step={0.1}
                          value={displayVal}
                          title={p.is_overridden ? 'Manually overridden — click ↺ to reset' : 'Click to override projection'}
                          onChange={e => setEditingProj(prev => ({ ...prev, [p.player_id]: e.target.value }))}
                          onFocus={() => setEditingProj(prev => ({ ...prev, [p.player_id]: p.mean.toFixed(1) }))}
                          onBlur={() => persistOverride(p.player_id)}
                          onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }}
                          disabled={overrideSaving}
                        />
                        {p.is_overridden && (
                          <button
                            className="proj-reset-btn"
                            title="Reset to original projection"
                            onClick={() => resetOverride(p.player_id)}
                            disabled={overrideSaving}
                          >↺</button>
                        )}
                      </span>
                      {p.ownership_pct != null && (
                        <span className="lineup-player-own" title="Projected ownership">{p.ownership_pct.toFixed(1)}%</span>
                      )}
                    </div>
                  )
                  continue
                }
              }
              // Distinguish: player_id=null → not in DK salary CSV; player_id set but no projection → unprojected
              const notInSlate = slotEntry.player_id === null
              rows.push(
                <div key={`placeholder-${slotEntry.slot}`} className="lineup-player lineup-player--placeholder">
                  <span className="lineup-player-pos">—</span>
                  <span className="lineup-player-name">
                    {slotEntry.name}
                    <span className="batting-slot-bubble batting-slot-bubble--confirmed" title="Confirmed lineup slot">{slotEntry.slot}</span>
                  </span>
                  <TeamBadge team={team} className="lineup-player-team" />
                  <span className="lineup-player-sal lineup-player-sal--placeholder">—</span>
                  <span className="lineup-player-proj">0.0</span>
                  {notInSlate
                    ? <span className="lineup-player-not-in-slate" title="Player confirmed in lineup but not in the DK salary file">N/A</span>
                    : <span className="lineup-player-not-in-slate lineup-player-no-proj" title="Player in DK slate but has no projection">no prj</span>
                  }
                </div>
              )
            }
            return rows
          }
          const batterRows: React.ReactElement[] = isLocked && teamRecord
            ? renderLockedBatterRows()
            : batters.map((p, i) => {
                const slotNum = p.slot != null && p.slot >= 1 && p.slot <= 9 ? p.slot : null
                const draftVal = editingProj[p.player_id]
                const displayVal = draftVal !== undefined ? draftVal : p.mean.toFixed(1)
                return (
                  <div key={i} className="lineup-player">
                    <span className="lineup-player-pos">{p.position}</span>
                    <span className="lineup-player-name">
                      {p.name}
                      {p.slot_confirmed
                        ? <span className="batting-slot-bubble batting-slot-bubble--confirmed" title="Confirmed lineup slot">{slotNum ?? '?'}</span>
                        : <span className="batting-slot-bubble batting-slot-bubble--projected" title="Projected lineup slot">{slotNum ?? '?'}</span>
                      }
                    </span>
                    <TeamBadge team={p.team} className="lineup-player-team" />
                    <span className="lineup-player-sal">${(p.salary / 1000).toFixed(1)}k</span>
                    <span className={`lineup-player-proj${p.is_overridden ? ' lineup-player-proj--overridden' : ''}`}>
                      <input
                        className="proj-override-input"
                        type="number"
                        step={0.1}
                        value={displayVal}
                        title={p.is_overridden ? 'Manually overridden — click ↺ to reset' : 'Click to override projection'}
                        onChange={e => setEditingProj(prev => ({ ...prev, [p.player_id]: e.target.value }))}
                        onFocus={() => setEditingProj(prev => ({ ...prev, [p.player_id]: p.mean.toFixed(1) }))}
                        onBlur={() => persistOverride(p.player_id)}
                        onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }}
                        disabled={overrideSaving}
                      />
                      {p.is_overridden && (
                        <button
                          className="proj-reset-btn"
                          title="Reset to original projection"
                          onClick={() => resetOverride(p.player_id)}
                          disabled={overrideSaving}
                        >↺</button>
                      )}
                    </span>
                    {p.ownership_pct != null && (
                      <span className="lineup-player-own" title="Projected ownership">{p.ownership_pct.toFixed(1)}%</span>
                    )}
                  </div>
                )
              })

          return (
            <div key={team} className={`lineup-card${hasReduction ? ' lineup-card--own-red' : ''}${isLocked ? ' lineup-card--locked' : ''}`}>
              <div className="lineup-card-header">
                <TeamBadge team={team} />
                <div className="lineup-card-header-right">
                  <span className="projections-team-total">{hitterProj.toFixed(1)} pts</span>
                  <div className="lineup-lock-controls">
                    <button
                      className={`btn-lock${isLocked ? ' btn-lock--locked' : ' btn-lock--unlocked'}`}
                      title={isLocked ? 'Lineup locked — click to unlock' : teamRecord ? 'Click to lock lineup' : 'No confirmed lineup to lock'}
                      onClick={() => teamRecord && onLockToggle?.(team, !isLocked)}
                      disabled={!teamRecord}
                      aria-label={isLocked ? `Unlock ${team} lineup` : `Lock ${team} lineup`}
                    >
                      {isLocked ? '🔒' : '🔓'}
                    </button>
                    <button
                      className={`btn-lineup-refresh${isLocked || isRefreshing ? ' btn-lineup-refresh--disabled' : ''}`}
                      title={isLocked ? 'Unlock lineup first to refresh' : isRefreshing ? 'Refreshing…' : 'Refresh lineup from RotoWire'}
                      onClick={() => !isLocked && !isRefreshing && handleRefreshTeam(team)}
                      disabled={isLocked || isRefreshing}
                      aria-label={`Refresh ${team} lineup`}
                    >
                      {isRefreshing ? '…' : '↻'}
                    </button>
                  </div>
                  <label className="own-red-label" title="Reduce projected field ownership for this team">
                    Own Red
                    <input
                      className="own-red-input ppd-input"
                      type="number"
                      min={1}
                      max={99}
                      step={1}
                      placeholder="—"
                      value={teamReductions[team] ?? ''}
                      onChange={e => setTeamReductions(prev => ({ ...prev, [team]: e.target.value }))}
                      onBlur={persistReductions}
                      onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }}
                      disabled={reductionSaving}
                    />
                    %
                  </label>
                  {teamTotals?.[team] != null && (
                    <span className="projections-implied-total" title="Implied team run total (betting market)">{teamTotals[team].toFixed(1)} R</span>
                  )}
                </div>
              </div>
              <div className="lineup-card-players projections-card-players">
                {pitchers.map((p, i) => {
                  const draftVal = editingProj[p.player_id]
                  const displayVal = draftVal !== undefined ? draftVal : p.mean.toFixed(1)
                  return (
                    <div key={`p-${i}`} className="lineup-player">
                      <span className="lineup-player-pos">{p.position}</span>
                      <span className="lineup-player-name">{p.name}</span>
                      <TeamBadge team={p.team} className="lineup-player-team" />
                      <span className="lineup-player-sal">${(p.salary / 1000).toFixed(1)}k</span>
                      <span className={`lineup-player-proj${p.is_overridden ? ' lineup-player-proj--overridden' : ''}`}>
                        <input
                          className="proj-override-input"
                          type="number"
                          step={0.1}
                          value={displayVal}
                          title={p.is_overridden ? 'Manually overridden — click ↺ to reset' : 'Click to override projection'}
                          onChange={e => setEditingProj(prev => ({ ...prev, [p.player_id]: e.target.value }))}
                          onFocus={() => setEditingProj(prev => ({ ...prev, [p.player_id]: p.mean.toFixed(1) }))}
                          onBlur={() => persistOverride(p.player_id)}
                          onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }}
                          disabled={overrideSaving}
                        />
                        {p.is_overridden && (
                          <button
                            className="proj-reset-btn"
                            title="Reset to original projection"
                            onClick={() => resetOverride(p.player_id)}
                            disabled={overrideSaving}
                          >↺</button>
                        )}
                      </span>
                      {p.ownership_pct != null && (
                        <span className="lineup-player-own" title="Projected ownership">{p.ownership_pct.toFixed(1)}%</span>
                      )}
                    </div>
                  )
                })}
                {batterRows}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
