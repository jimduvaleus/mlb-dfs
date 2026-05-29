import { useState, useEffect } from 'react'
import type { LineupResult, GppMvSelectProgressEvent, CompleteEvent } from '../types'
import { useSSE } from '../hooks/useSSE'

interface Props {
  isOpen: boolean
  initialRisk: number
  initialNIter: number
  initialNRestarts: number
  onClose: () => void
  onComplete: (portfolio: LineupResult[]) => void
}

export function ReselectDialog({
  isOpen,
  initialRisk,
  initialNIter,
  initialNRestarts,
  onClose,
  onComplete,
}: Props) {
  const [risk, setRisk] = useState(initialRisk)
  const [nIter, setNIter] = useState(initialNIter)
  const [nRestarts, setNRestarts] = useState(initialNRestarts)
  const [saveConfig, setSaveConfig] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const { events, status, start, reset } = useSSE('/api/portfolio/reselect')
  const running = status === 'connecting' || status === 'streaming'

  // Reset form values when dialog opens with fresh initial values
  useEffect(() => {
    if (isOpen) {
      setRisk(initialRisk)
      setNIter(initialNIter)
      setNRestarts(initialNRestarts)
      setError(null)
    }
  }, [isOpen, initialRisk, initialNIter, initialNRestarts])

  // Handle terminal SSE events
  useEffect(() => {
    if (events.length === 0) return
    const last = events[events.length - 1]
    if (last.stage === 'complete') {
      const ce = last as CompleteEvent
      onComplete(ce.portfolio)
      handleClose()
    } else if (last.stage === 'error') {
      setError(((last as unknown) as { stage: 'error'; message: string }).message ?? 'An error occurred.')
    }
  }, [events])

  const handleRun = () => {
    setError(null)
    start({
      risk: String(risk),
      n_iter: String(nIter),
      n_restarts: String(nRestarts),
      save_config: String(saveConfig),
    })
  }

  const handleClose = () => {
    reset()
    onClose()
  }

  // Derive SA progress from the latest gpp_mv_select_progress event
  const progressEvents = events.filter(
    (e): e is GppMvSelectProgressEvent => e.stage === 'gpp_mv_select_progress'
  )
  const latestProgress = progressEvents[progressEvents.length - 1] ?? null
  const progressFraction = latestProgress
    ? latestProgress.iteration / latestProgress.total_iterations
    : 0

  if (!isOpen) return null

  return (
    <div className="dialog-overlay" onClick={running ? undefined : handleClose}>
      <div className="dialog reselect-dialog" onClick={e => e.stopPropagation()}>
        <p className="dialog-title">Refine SA Portfolio</p>
        <p className="dialog-message">
          Adjust simulated annealing settings and re-run portfolio selection
          using the cached candidate pool — no simulation re-run needed.
        </p>

        <div className="reselect-fields">
          <label className="reselect-field">
            <div className="reselect-field-header">
              <span className="reselect-field-label">Risk tolerance</span>
              <span className="reselect-field-value">{risk.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={10}
              step={0.5}
              value={risk}
              disabled={running}
              onChange={e => setRisk(Number(e.target.value))}
              className="reselect-slider"
            />
            <div className="reselect-slider-labels">
              <span>0 — max diversity</span>
              <span>10 — max EV</span>
            </div>
          </label>

          <label className="reselect-field">
            <span className="reselect-field-label">SA iterations</span>
            <input
              type="number"
              min={1000}
              step={1000}
              value={nIter}
              disabled={running}
              onChange={e => setNIter(Math.max(1000, Number(e.target.value)))}
              className="reselect-number"
            />
          </label>

          <label className="reselect-field">
            <span className="reselect-field-label">SA restarts</span>
            <input
              type="number"
              min={1}
              max={20}
              step={1}
              value={nRestarts}
              disabled={running}
              onChange={e => setNRestarts(Math.min(20, Math.max(1, Number(e.target.value))))}
              className="reselect-number"
            />
          </label>

          <label className="reselect-field reselect-field--inline">
            <input
              type="checkbox"
              checked={saveConfig}
              disabled={running}
              onChange={e => setSaveConfig(e.target.checked)}
            />
            <span>Save settings to config</span>
          </label>
        </div>

        {(running || progressEvents.length > 0) && (
          <div className="reselect-progress">
            <div className="reselect-progress-bar-track">
              <div
                className="reselect-progress-bar-fill"
                style={{ width: `${Math.round(progressFraction * 100)}%` }}
              />
            </div>
            {latestProgress && (
              <div className="reselect-progress-label">
                Restart {latestProgress.restart + 1} of {nRestarts} —{' '}
                {Math.round(progressFraction * 100)}% —{' '}
                mean payout {latestProgress.portfolio_mean.toFixed(3)}
              </div>
            )}
            {!latestProgress && running && (
              <div className="reselect-progress-label">Starting…</div>
            )}
          </div>
        )}

        {error && <div className="reselect-error">{error}</div>}

        <div className="dialog-actions">
          {!running && (
            <button className="btn-run" onClick={handleRun}>
              Run
            </button>
          )}
          <button
            className="btn-secondary"
            onClick={handleClose}
            disabled={running}
          >
            {running ? 'Running…' : 'Cancel'}
          </button>
        </div>
      </div>
    </div>
  )
}
