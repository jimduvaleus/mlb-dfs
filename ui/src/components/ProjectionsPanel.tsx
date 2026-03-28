import { useEffect, useRef, useState } from 'react'
import type { ProjectionsStatus, SlateOption } from '../types'
import { fetchProjectionsStatus, fetchProjectionSlates } from '../api'

interface Props {
  disabled?: boolean
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

export function ProjectionsPanel({ disabled }: Props) {
  const [status, setStatus] = useState<ProjectionsStatus | null>(null)
  const [slates, setSlates] = useState<SlateOption[]>([])
  const [selectedSlateId, setSelectedSlateId] = useState<string | null>(null)
  const [fetching, setFetching] = useState(false)
  const [log, setLog] = useState<string[]>([])
  const [done, setDone] = useState<{ success: boolean; code: number } | null>(null)
  const logRef = useRef<HTMLDivElement>(null)
  const esRef = useRef<EventSource | null>(null)

  const loadSlates = () => {
    fetchProjectionSlates()
      .then(res => {
        setSlates(res.slates)
        // Only set the default if no slate is selected yet
        setSelectedSlateId(prev => {
          if (prev !== null) return prev
          const def = res.slates.find(s => s.is_default)
          return def ? def.slate_id : (res.slates[0]?.slate_id ?? null)
        })
      })
      .catch(console.error)
  }

  useEffect(() => {
    fetchProjectionsStatus().then(setStatus).catch(console.error)
    loadSlates()
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

    const url = selectedSlateId
      ? `/api/projections/fetch?slate_id=${encodeURIComponent(selectedSlateId)}`
      : '/api/projections/fetch'
    const es = new EventSource(url)
    esRef.current = es

    es.onmessage = (e) => {
      const event = JSON.parse(e.data)
      if (event.type === 'log') {
        setLog(prev => [...prev, event.line])
      } else if (event.type === 'done') {
        setFetching(false)
        setDone({ success: event.returncode === 0, code: event.returncode })
        es.close()
        esRef.current = null
        fetchProjectionsStatus().then(setStatus).catch(console.error)
        loadSlates()
      }
    }

    es.onerror = () => {
      setFetching(false)
      es.close()
      esRef.current = null
    }
  }

  const defaultSlate = slates.find(s => s.is_default)

  return (
    <div className="projections-panel">
      <h3>Projections</h3>

      {slates.length > 1 && (
        <div className="proj-slate-selector">
          <label htmlFor="slate-select">Slate</label>
          <select
            id="slate-select"
            value={selectedSlateId ?? ''}
            onChange={e => setSelectedSlateId(e.target.value)}
            disabled={fetching || disabled}
          >
            {slates.map(s => (
              <option key={s.slate_id} value={s.slate_id}>
                {s.name}{s.is_default ? ' (default)' : ''}
              </option>
            ))}
          </select>
        </div>
      )}

      {slates.length === 1 && (
        <div className="proj-slate-name muted">
          {defaultSlate?.name ?? slates[0].name}
        </div>
      )}

      {status === null ? (
        <p className="muted">Loading…</p>
      ) : status.exists ? (
        <div className="proj-status">
          <span className="badge badge-ok">up to date</span>
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
              {status.unconfirmed_count} unconfirmed lineup{status.unconfirmed_count !== 1 ? 's' : ''}
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
    </div>
  )
}

function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s ago`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h ago`
  return `${(seconds / 86400).toFixed(1)}d ago`
}
