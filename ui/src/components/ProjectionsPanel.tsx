import { useEffect, useRef, useState } from 'react'
import type { ProjectionsStatus } from '../types'
import { fetchProjectionsStatus } from '../api'

interface Props {
  disabled?: boolean
  onFetched?: () => void
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

interface MergeInfo {
  secondarySource: string
  count: number
  players: string[]
}

export function ProjectionsPanel({ disabled, onFetched }: Props) {
  const [status, setStatus] = useState<ProjectionsStatus | null>(null)
  const [fetching, setFetching] = useState(false)
  const [log, setLog] = useState<string[]>([])
  const [done, setDone] = useState<{ success: boolean; code: number } | null>(null)
  const [mergeInfo, setMergeInfo] = useState<MergeInfo | null>(null)
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

  const handleFetch = () => {
    if (fetching || disabled) return
    setFetching(true)
    setLog([])
    setDone(null)
    setMergeInfo(null)

    const es = new EventSource('/api/projections/fetch')
    esRef.current = es

    es.onmessage = (e) => {
      const event = JSON.parse(e.data)
      if (event.type === 'log') {
        setLog(prev => [...prev, event.line])
      } else if (event.type === 'merge_info') {
        setMergeInfo({ secondarySource: event.secondary_source, count: event.count, players: event.players })
      } else if (event.type === 'done') {
        setFetching(false)
        setDone({ success: event.returncode === 0, code: event.returncode })
        es.close()
        esRef.current = null
        refreshStatus()
        if (event.returncode === 0) onFetched?.()
      }
    }

    es.onerror = () => {
      setFetching(false)
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
        {fetching ? 'Fetching…' : 'Fetch New Projections'}
      </button>

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

      {mergeInfo && (
        <div className="merge-info-callout">
          <strong>{mergeInfo.count} player{mergeInfo.count !== 1 ? 's' : ''} using {mergeInfo.secondarySource} fallback projection</strong>
          <div className="merge-info-players">
            {mergeInfo.players.join(', ')}
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
