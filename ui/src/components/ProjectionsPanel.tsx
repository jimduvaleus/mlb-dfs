import { useEffect, useRef, useState } from 'react'
import type { ProjectionsStatus } from '../types'
import { fetchProjectionsStatus } from '../api'

interface Props {
  disabled?: boolean
}

function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s ago`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h ago`
  return `${(seconds / 86400).toFixed(1)}d ago`
}

export function ProjectionsPanel({ disabled }: Props) {
  const [status, setStatus] = useState<ProjectionsStatus | null>(null)
  const [fetching, setFetching] = useState(false)
  const [log, setLog] = useState<string[]>([])
  const [done, setDone] = useState<{ success: boolean; code: number } | null>(null)
  const logRef = useRef<HTMLDivElement>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    fetchProjectionsStatus().then(setStatus).catch(console.error)
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

    const es = new EventSource('/api/projections/fetch')
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
        // Refresh status after fetch
        fetchProjectionsStatus().then(setStatus).catch(console.error)
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
          <span className="badge badge-ok">up to date</span>
          <span className="proj-path">{status.path}</span>
          {status.age_seconds !== null && (
            <span className="muted">Updated {formatAge(status.age_seconds)}</span>
          )}
          {status.row_count !== null && (
            <span className="muted">{status.row_count.toLocaleString()} players</span>
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
