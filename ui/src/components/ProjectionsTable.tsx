import { useState, useEffect, useCallback } from 'react'
import type { ProjectionPlayerRow, PlatformType } from '../types'
import { fetchTeamOwnershipReductions, saveTeamOwnershipReductions, fetchPlayerProjectionOverrides, savePlayerProjectionOverrides } from '../api'
import TeamBadge from './TeamBadge'

interface Props {
  players: ProjectionPlayerRow[]
  platform?: PlatformType
  teamTotals?: Record<string, number>
  onOwnershipSettingsChanged?: () => void
}

export function ProjectionsTable({ players, teamTotals, onOwnershipSettingsChanged }: Props) {
  const [reductionSlateId, setReductionSlateId] = useState<string>('')
  const [teamReductions, setTeamReductions] = useState<Record<string, string>>({})
  const [reductionSaving, setReductionSaving] = useState(false)

  const [overrideSlateId, setOverrideSlateId] = useState<string>('')
  const [projOverrides, setProjOverrides] = useState<Record<number, number>>({})
  const [editingProj, setEditingProj] = useState<Record<number, string>>({})
  const [overrideSaving, setOverrideSaving] = useState(false)

  useEffect(() => {
    fetchTeamOwnershipReductions().then(r => {
      setReductionSlateId(r.slate_id)
      const s: Record<string, string> = {}
      for (const [t, v] of Object.entries(r.team_ownership_reductions)) {
        if (v > 0) s[t] = String(v)
      }
      setTeamReductions(s)
    }).catch(() => {})

    fetchPlayerProjectionOverrides().then(r => {
      setOverrideSlateId(r.slate_id)
      const overrides: Record<number, number> = {}
      for (const [k, v] of Object.entries(r.player_projection_overrides)) {
        overrides[Number(k)] = v
      }
      setProjOverrides(overrides)
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

  return (
    <div className="projections-table-wrap">
      <h3>Projections — {players.length} players, {teams.length} teams</h3>
      <div className="portfolio-cards">
        {teams.map(team => {
          const teamPlayers = byTeam.get(team)!
          const pitchers = teamPlayers.filter(p => p.position === 'P')
          const batters  = teamPlayers
            .filter(p => p.position !== 'P')
            .sort((a, b) => (a.slot ?? 99) - (b.slot ?? 99))
          const hitterProj = batters.reduce((sum, p) => sum + p.mean, 0)
          const hasReduction = parseFloat(teamReductions[team] ?? '0') > 0

          return (
            <div key={team} className={`lineup-card${hasReduction ? ' lineup-card--own-red' : ''}`}>
              <div className="lineup-card-header">
                <TeamBadge team={team} />
                <div className="lineup-card-header-right">
                  <span className="projections-team-total">{hitterProj.toFixed(1)} pts</span>
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
                {[...pitchers, ...batters].map((p, i) => {
                  const isPitcher = p.position === 'P'
                  const slotNum = !isPitcher && p.slot != null && p.slot >= 1 && p.slot <= 9 ? p.slot : null
                  const draftVal = editingProj[p.player_id]
                  const displayVal = draftVal !== undefined ? draftVal : p.mean.toFixed(1)
                  return (
                    <div key={i} className="lineup-player">
                      <span className="lineup-player-pos">{p.position}</span>
                      <span className="lineup-player-name">
                        {p.name}
                        {!isPitcher && (
                          p.slot_confirmed
                            ? <span className="batting-slot-bubble batting-slot-bubble--confirmed" title="Confirmed lineup slot">{slotNum ?? '?'}</span>
                            : <span className="batting-slot-bubble batting-slot-bubble--projected" title="Projected lineup slot">{slotNum ?? '?'}</span>
                        )}
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
                })}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
