import React, { useEffect, useRef, useState } from 'react'
import type { CappedPlayer, ExclusionScope, FallbackTeam, LowTeamProjection, MergeInfo, MissingOptPlayer, ProjectionsStatus } from '../types'

const MARKET_DISPLAY: Record<string, string> = {
  singles: '1B', doubles: '2B', triples: '3B', home_runs: 'HR',
  stolen_bases: 'SB', walks: 'BB', runs: 'R', rbis: 'RBI',
}
import { fetchMergeInfoState, fetchProjectionsStatus, fetchSlatePlayers, savePlayerExclusions } from '../api'

interface Props {
  disabled?: boolean
  onFetched?: () => void
  mergeInfo: MergeInfo | null
  onMergeInfo: (info: MergeInfo | null) => void
  projFetchExcluded?: string[]
  onFetchingChange?: (fetching: boolean) => void
  refreshTrigger?: number
}

function formatET(unixSec: number): string {
  return new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short',
  }).format(new Date(unixSec * 1000))
}

export function ProjectionsPanel({ disabled, onFetched, mergeInfo, onMergeInfo, projFetchExcluded = [], onFetchingChange, refreshTrigger }: Props) {
  const [status, setStatus] = useState<ProjectionsStatus | null>(null)
  const [fetching, setFetching] = useState(false)
  const [log, setLog] = useState<string[]>([])
  const [done, setDone] = useState<{ success: boolean; code: number } | null>(null)
  const logRef = useRef<HTMLDivElement>(null)
  const esRef = useRef<EventSource | null>(null)

  const refreshStatus = () => {
    fetchProjectionsStatus().then(setStatus).catch(console.error)
  }

  useEffect(() => {
    refreshStatus()
    fetchMergeInfoState().then(state => {
      const players = (state.players ?? []) as MergeInfo['players']
      const cappedPlayers = (state.capped_players ?? []) as CappedPlayer[]
      const missingOptPlayers = (state.missing_opt_players ?? []) as MissingOptPlayer[]
      if (players.length || cappedPlayers.length || missingOptPlayers.length) {
        const fallbackTeams = (state.fallback_teams ?? []) as FallbackTeam[]
        onMergeInfo({
          secondarySource: (state.secondary_source as string) || 'RotoWire',
          count: players.length,
          players,
          cappedPlayers,
          missingOptPlayers,
          fallbackTeams,
        })
      }
    }).catch(() => {})
    return () => esRef.current?.close()
  }, [])

  useEffect(() => {
    if (refreshTrigger) refreshStatus()
  }, [refreshTrigger])

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [log])

  const isPartial = projFetchExcluded.length > 0

  const handleFetch = () => {
    if (fetching || disabled) return
    setFetching(true)
    onFetchingChange?.(true)
    setLog([])
    setDone(null)
    onMergeInfo(null)

    const params = isPartial
      ? `?exclude_games=${encodeURIComponent(projFetchExcluded.join(','))}`
      : ''
    const es = new EventSource(`/api/projections/fetch${params}`)
    esRef.current = es

    es.onmessage = (e) => {
      const event = JSON.parse(e.data)
      if (event.type === 'log') {
        setLog(prev => [...prev, event.line])
      } else if (event.type === 'merge_info') {
        const players = event.players as Array<{ name: string; team: string; reason?: string; player_id?: number; is_pitcher?: boolean }>
        const cappedPlayers = (event.capped_players ?? []) as CappedPlayer[]
        const lowTeamProjections = (event.low_team_projections ?? []) as LowTeamProjection[]
        const fallbackTeams = (event.fallback_teams ?? []) as FallbackTeam[]
        const missingOptPlayers = (event.missing_opt_players ?? []) as MissingOptPlayer[]
        onMergeInfo({ secondarySource: event.secondary_source, count: event.count, players, cappedPlayers, lowTeamProjections, fallbackTeams, missingOptPlayers })
        // Auto-exclude pitchers that fell back to a secondary source at 'candidates'
        // scope — their projections are lower quality but they still model the field.
        const pitcherIds = players.filter(p => p.is_pitcher && p.player_id).map(p => p.player_id as number)
        if (pitcherIds.length > 0) {
          fetchSlatePlayers().then(slate => {
            const player_scopes: Record<string, ExclusionScope> = {}
            for (const p of slate.players) {
              if (p.individual_scope !== 'none') player_scopes[String(p.player_id)] = p.individual_scope
            }
            let addedNew = false
            for (const id of pitcherIds) {
              const key = String(id)
              if (!player_scopes[key]) {
                player_scopes[key] = 'candidates'
                addedNew = true
              }
              // already excluded at some scope — leave as-is (no downgrade)
            }
            if (addedNew) {
              savePlayerExclusions({ slate_id: slate.slate_id, player_scopes }).catch(console.error)
            }
          }).catch(console.error)
        }
      } else if (event.type === 'done') {
        setFetching(false)
        onFetchingChange?.(false)
        setDone({ success: event.returncode === 0, code: event.returncode })
        es.close()
        esRef.current = null
        refreshStatus()
        if (event.returncode === 0) onFetched?.()
      }
    }

    es.onerror = () => {
      setFetching(false)
      onFetchingChange?.(false)
      es.close()
      esRef.current = null
    }
  }

  return (
    <div className="projections-panel">
      <h3>Projections</h3>

      {status === null ? (
        <p className="muted">Loading…</p>
      ) : status.exists ? (
        <div className="proj-status">
          {status.is_fresh === false ? (
            <span
              className="badge badge-warn"
              onClick={refreshStatus}
              style={{ cursor: 'pointer' }}
              title="Click to re-check"
            >
              stale
            </span>
          ) : (
            <span
              className="badge badge-ok"
              onClick={refreshStatus}
              style={{ cursor: 'pointer' }}
              title="Click to re-check"
            >
              up to date
            </span>
          )}
          <span className="proj-path">{status.path}</span>
          {status.fetch_timestamp_utc !== null ? (
            <span className="muted">Updated {formatET(status.fetch_timestamp_utc)}</span>
          ) : status.age_seconds !== null ? (
            <span className="muted">Updated {formatAge(status.age_seconds)}</span>
          ) : null}
          {status.row_count !== null && (
            <span className="muted">{status.row_count.toLocaleString()} players</span>
          )}
          {status.unconfirmed_count !== null && status.unconfirmed_count > 0 && (
            <span className="badge badge-warn">
              {status.unconfirmed_count} unconfirmed lineup slot{status.unconfirmed_count !== 1 ? 's' : ''}
            </span>
          )}
          {status.no_changes === true && (
            <span className="muted proj-no-changes">No projection changes since last fetch</span>
          )}
        </div>
      ) : (
        <div className="proj-status">
          <span className="badge badge-warn">not found</span>
          {status.path && <span className="proj-path">{status.path}</span>}
        </div>
      )}

      <button
        onClick={handleFetch}
        disabled={fetching || disabled}
        className="btn-secondary"
      >
        {fetching ? 'Fetching…' : isPartial ? 'Fetch Projections (partial)' : 'Fetch New Projections'}
      </button>
      {isPartial && !fetching && (
        <p className="proj-partial-note">
          {projFetchExcluded.length} game{projFetchExcluded.length !== 1 ? 's' : ''} excluded —
          will fetch the remaining games and merge into existing projections.
        </p>
      )}

      {log.length > 0 && (
        <div className="proj-log" ref={logRef}>
          {log.map((line, i) => (
            <div key={i} className="log-line">{line}</div>
          ))}
          {done && (
            <div className={done.success ? 'log-line success' : 'log-line error'}>
              {done.success ? '✓ Done' : `✗ Exited with code ${done.code}`}
            </div>
          )}
        </div>
      )}

      {mergeInfo && mergeInfo.fallbackTeams && mergeInfo.fallbackTeams.length > 0 && (
        <div className="merge-info-fallback-teams-callout">
          <strong>
            ⛔ No market odds for {mergeInfo.fallbackTeams.length} team{mergeInfo.fallbackTeams.length !== 1 ? 's' : ''} — entire lineup on {mergeInfo.secondarySource} ×0.8 fallback
          </strong>
          <div className="merge-info-fallback-teams-list">
            {mergeInfo.fallbackTeams.map((ft, i) => (
              <span key={i} className="merge-info-fallback-team-chip">
                <span className="chip-team">{ft.team}</span>
                {ft.game && <span className="chip-game">({ft.game})</span>}
                <span className="chip-count">{ft.count} batters</span>
              </span>
            ))}
          </div>
          <div style={{ marginTop: 6, fontSize: '0.8em', color: '#fc8181', opacity: 0.85 }}>
            Fetch projections for {mergeInfo.fallbackTeams.length > 1 ? 'these games' : 'this game'} to get market odds coverage.
          </div>
        </div>
      )}

      {mergeInfo && mergeInfo.cappedPlayers && mergeInfo.cappedPlayers.length > 0 && (
        <div className="merge-info-callout merge-info-caps-callout">
          <strong>⚠ Hard cap applied — {mergeInfo.cappedPlayers.length} player{mergeInfo.cappedPlayers.length !== 1 ? 's' : ''}</strong>
          <div className="merge-info-cap-note">
            One or more market E[X] estimates hit the per-market ceiling.
            Verify the odds manually — may reflect a genuine edge case (e.g. Coors HR, speedster vs slow battery).
          </div>
          <div className="merge-info-players">
            {mergeInfo.cappedPlayers.map((p, i) => (
              <span key={i} className="merge-info-team-group">
                <span className="merge-info-team-label">{p.team}</span>
                {' ('}
                <span title={`Capped markets: ${p.markets.map(m => MARKET_DISPLAY[m] ?? m).join(', ')}`}>
                  {p.name}
                  <span className="merge-info-reason"> ⓘ {p.markets.map(m => MARKET_DISPLAY[m] ?? m).join(', ')}</span>
                </span>
                {')'}
              </span>
            )).reduce<React.ReactNode[]>((acc, el, i) => i === 0 ? [el] : [...acc, ', ', el], [])}
          </div>
        </div>
      )}

      {mergeInfo && mergeInfo.lowTeamProjections && mergeInfo.lowTeamProjections.length > 0 && (
        <div className="merge-info-callout merge-info-low-team-callout">
          <strong>⚠ Low team projection — {mergeInfo.lowTeamProjections.length} team{mergeInfo.lowTeamProjections.length !== 1 ? 's' : ''} below 40 pts</strong>
          <div className="merge-info-players">
            {mergeInfo.lowTeamProjections.map((t, i) => (
              <span key={i} className="merge-info-team-group">
                <span className="merge-info-team-label">{t.team}</span>
                {' '}
                <span className="merge-info-reason">{t.total.toFixed(1)} pts</span>
              </span>
            )).reduce<React.ReactNode[]>((acc, el, i) => i === 0 ? [el] : [...acc, ', ', el], [])}
          </div>
        </div>
      )}

      {mergeInfo && mergeInfo.missingOptPlayers && mergeInfo.missingOptPlayers.length > 0 && (
        <div className="merge-info-missing-opt-callout">
          <strong>
            ⚠ {mergeInfo.missingOptPlayers.length} batter{mergeInfo.missingOptPlayers.length !== 1 ? 's' : ''} missing optional market(s) — full MO projection used
          </strong>
          <div className="merge-info-players">
            {Object.entries(
              mergeInfo.missingOptPlayers.reduce<Record<string, MissingOptPlayer[]>>((acc, p) => {
                const key = p.team || '—'
                ;(acc[key] ??= []).push(p)
                return acc
              }, {})
            )
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([team, ps]) => (
                <span key={team} className="merge-info-team-group">
                  <span className="merge-info-team-label">{team}</span>
                  {' ('}
                  {ps.map((p, i) => (
                    <span key={p.name}>
                      {i > 0 && ', '}
                      {p.name}
                      <span className="merge-info-reason" title={`Missing: ${p.markets.map(m => MARKET_DISPLAY[m] ?? m).join(', ')}`}> ⓘ</span>
                    </span>
                  ))}
                  {')'}
                </span>
              ))
              .reduce<React.ReactNode[]>((acc, el, i) => i === 0 ? [el] : [...acc, ', ', el], [])}
          </div>
        </div>
      )}

      {(() => {
        if (!mergeInfo || mergeInfo.count === 0) return null
        const batters  = mergeInfo.players.filter(p => !p.is_pitcher)
        const pitchers = mergeInfo.players.filter(p => p.is_pitcher)

        const groupByTeam = (players: typeof mergeInfo.players) => {
          const byTeam = players.reduce<Record<string, typeof mergeInfo.players>>((acc, p) => {
            const key = p.team || '—'
            ;(acc[key] ??= []).push(p)
            return acc
          }, {})
          return Object.entries(byTeam)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([team, ps]) => (
              <span key={team} className="merge-info-team-group">
                <span className="merge-info-team-label">{team}</span>
                {' ('}
                {ps.map((p, i) => (
                  <span key={p.name}>
                    {i > 0 && ', '}
                    {p.name}
                    {p.reason
                      ? <span className="merge-info-reason" title={p.reason}> ⓘ</span>
                      : null}
                  </span>
                ))}
                {')'}
              </span>
            ))
            .reduce<React.ReactNode[]>((acc, el, i) => i === 0 ? [el] : [...acc, ', ', el], [])
        }

        return (
          <>
            {batters.length > 0 && (
              <div className="merge-info-player-fallback-callout">
                <strong>
                  ⚠ {batters.length} batter{batters.length !== 1 ? 's' : ''} using {mergeInfo.secondarySource} ×0.9 fallback
                </strong>
                <div className="merge-info-players">{groupByTeam(batters)}</div>
              </div>
            )}
            {pitchers.length > 0 && (
              <div className="merge-info-pitcher-fallback-callout">
                <strong>
                  ⚠ {pitchers.length} pitcher{pitchers.length !== 1 ? 's' : ''} using {mergeInfo.secondarySource} fallback — added to candidate exclusions
                </strong>
                <div className="merge-info-players">{groupByTeam(pitchers)}</div>
              </div>
            )}
          </>
        )
      })()}
    </div>
  )
}

function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s ago`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h ago`
  return `${(seconds / 86400).toFixed(1)}d ago`
}
