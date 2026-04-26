import React, { useEffect, useRef, useState } from 'react'
import type { CappedPlayer, MergeInfo, ProjectionsStatus } from '../types'

const MARKET_DISPLAY: Record<string, string> = {
  singles: '1B', doubles: '2B', triples: '3B', home_runs: 'HR',
  stolen_bases: 'SB', walks: 'BB', runs: 'R', rbis: 'RBI',
}
import { fetchProjectionsStatus, fetchSlatePlayers, savePlayerExclusions } from '../api'

interface Props {
  disabled?: boolean
  onFetched?: () => void
  mergeInfo: MergeInfo | null
  onMergeInfo: (info: MergeInfo | null) => void
  projFetchExcluded?: string[]
  onFetchingChange?: (fetching: boolean) => void
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

export function ProjectionsPanel({ disabled, onFetched, mergeInfo, onMergeInfo, projFetchExcluded = [], onFetchingChange }: Props) {
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
    return () => esRef.current?.close()
  }, [])

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
        onMergeInfo({ secondarySource: event.secondary_source, count: event.count, players, cappedPlayers })
        // Auto-exclude pitchers that fell back to a secondary source — their
        // projections are lower quality regardless of which primary source was used.
        const pitcherIds = players.filter(p => p.is_pitcher && p.player_id).map(p => p.player_id as number)
        if (pitcherIds.length > 0) {
          fetchSlatePlayers().then(slate => {
            const alreadyExcluded = slate.players.filter(p => p.excluded).map(p => p.player_id)
            const newExcluded = [...new Set([...alreadyExcluded, ...pitcherIds])]
            if (newExcluded.length > alreadyExcluded.length) {
              savePlayerExclusions({ slate_id: slate.slate_id, excluded_player_ids: newExcluded })
                .catch(console.error)
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

      {mergeInfo && mergeInfo.count > 0 && (
        <div className="merge-info-callout">
          <strong>{mergeInfo.count} player{mergeInfo.count !== 1 ? 's' : ''} using {mergeInfo.secondarySource} fallback projection</strong>
          {mergeInfo.players.some(p => p.is_pitcher) && (
            <div className="merge-info-pitcher-warning">
              ⚠ Pitcher(s) using {mergeInfo.secondarySource} fallback projections have been automatically added to Player Exclusions.
            </div>
          )}
          <div className="merge-info-players">
            {(() => {
              // Group by team; within each team show name + optional icon
              const byTeam = mergeInfo.players.reduce<Record<string, typeof mergeInfo.players>>((acc, p) => {
                const key = p.team || '—'
                ;(acc[key] ??= []).push(p)
                return acc
              }, {})
              return Object.entries(byTeam)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([team, players]) => (
                  <span key={team} className="merge-info-team-group">
                    <span className="merge-info-team-label">{team}</span>
                    {' ('}
                    {players.map((p, i) => (
                      <span key={p.name}>
                        {i > 0 && ', '}
                        {p.name}
                        {p.is_pitcher
                          ? <span className="merge-info-pitcher-icon" title="Pitcher excluded — no market odds projection available"> ⚠</span>
                          : p.reason
                            ? <span className="merge-info-reason" title={p.reason}> ⓘ</span>
                            : null
                        }
                      </span>
                    ))}
                    {')'}
                  </span>
                ))
                .reduce<React.ReactNode[]>((acc, el, i) => i === 0 ? [el] : [...acc, ', ', el], [])
            })()}
          </div>
        </div>
      )}
    </div>
  )
}

function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s ago`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h ago`
  return `${(seconds / 86400).toFixed(1)}d ago`
}
