import { useEffect, useReducer, useState } from 'react'
import type { AppConfig, LineupResult, RunStatus, CompleteEvent, StoppedEvent } from './types'
import { fetchConfig, fetchPortfolio, fetchUnconfirmedPlayerIds, replaceLineup, stopRun, writeUploadFiles } from './api'
import { useSSE } from './hooks/useSSE'
import { ConfigForm } from './components/ConfigForm'
import { ProjectionsPanel } from './components/ProjectionsPanel'
import { ProgressPanel } from './components/ProgressPanel'
import { PortfolioTable } from './components/PortfolioTable'
import { MetricsPanel } from './components/MetricsPanel'
import { SlatePanel } from './components/SlatePanel'
import { StopUploadDialog } from './components/StopUploadDialog'
import { DeleteConfirmModal } from './components/DeleteConfirmModal'
import './App.css'

type Tab = 'config' | 'slate' | 'run' | 'portfolio' | 'metrics'

interface State {
  config: AppConfig | null
  portfolio: LineupResult[]
  runStatus: RunStatus
  activeTab: Tab
  unconfirmedPlayerIds: number[]
}

type Action =
  | { type: 'set_config'; config: AppConfig }
  | { type: 'set_portfolio'; portfolio: LineupResult[] }
  | { type: 'set_run_status'; status: RunStatus }
  | { type: 'set_tab'; tab: Tab }
  | { type: 'set_unconfirmed'; ids: number[] }

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'set_config':
      return { ...state, config: action.config }
    case 'set_portfolio':
      return { ...state, portfolio: action.portfolio }
    case 'set_run_status':
      return { ...state, runStatus: action.status }
    case 'set_tab':
      return { ...state, activeTab: action.tab }
    case 'set_unconfirmed':
      return { ...state, unconfirmedPlayerIds: action.ids }
  }
}

const initial: State = {
  config: null,
  portfolio: [],
  runStatus: 'idle',
  activeTab: 'config',
  unconfirmedPlayerIds: [],
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initial)
  const [configError, setConfigError] = useState<string | null>(null)
  const [showUploadDialog, setShowUploadDialog] = useState(false)
  const [stoppedLineupCount, setStoppedLineupCount] = useState(0)
  const [stopPending, setStopPending] = useState(false)
  const [pendingDeleteIndex, setPendingDeleteIndex] = useState<number | null>(null)
  const [replacingIndex, setReplacingIndex] = useState<number | null>(null)
  const { events, status: sseStatus, start: startSSE, reset: resetSSE } = useSSE('/api/run/stream')

  const running = state.runStatus === 'running'

  const refreshUnconfirmed = () => {
    fetchUnconfirmedPlayerIds()
      .then(ids => dispatch({ type: 'set_unconfirmed', ids }))
      .catch(() => {})
  }

  // Load config, existing portfolio, and unconfirmed player IDs on mount
  useEffect(() => {
    fetchConfig()
      .then(cfg => dispatch({ type: 'set_config', config: cfg }))
      .catch(e => setConfigError(String(e)))
    fetchPortfolio()
      .then(portfolio => {
        if (portfolio.length > 0) {
          dispatch({ type: 'set_portfolio', portfolio })
          dispatch({ type: 'set_run_status', status: 'complete' })
        }
      })
      .catch(() => {})
    refreshUnconfirmed()
  }, [])

  // React to SSE events
  useEffect(() => {
    for (const event of events) {
      if (event.stage === 'complete') {
        const ce = event as CompleteEvent
        dispatch({ type: 'set_portfolio', portfolio: ce.portfolio })
        dispatch({ type: 'set_run_status', status: 'complete' })
        dispatch({ type: 'set_tab', tab: 'portfolio' })
      } else if (event.stage === 'stopped') {
        const se = event as StoppedEvent
        dispatch({ type: 'set_portfolio', portfolio: se.portfolio })
        dispatch({ type: 'set_run_status', status: 'stopped' })
        dispatch({ type: 'set_tab', tab: 'portfolio' })
        setStopPending(false)
        if (se.n_lineups > 0) {
          setStoppedLineupCount(se.n_lineups)
          setShowUploadDialog(true)
        }
      } else if (event.stage === 'error') {
        dispatch({ type: 'set_run_status', status: 'error' })
      }
    }
  }, [events])

  const handleRun = () => {
    if (running) return
    resetSSE()
    setShowUploadDialog(false)
    setStopPending(false)
    dispatch({ type: 'set_run_status', status: 'running' })
    dispatch({ type: 'set_tab', tab: 'run' })
    startSSE()
  }

  const handleStop = () => {
    if (!running || stopPending) return
    setStopPending(true)
    stopRun().catch(() => setStopPending(false))
  }

  const handleWriteUpload = () => {
    setShowUploadDialog(false)
    writeUploadFiles().catch(() => {})
  }

  const handleDeleteLineup = (lineupIndex: number) => {
    setPendingDeleteIndex(lineupIndex)
  }

  const handleConfirmDelete = async () => {
    if (pendingDeleteIndex === null) return
    const idx = pendingDeleteIndex
    setPendingDeleteIndex(null)
    setReplacingIndex(idx)
    dispatch({ type: 'set_run_status', status: 'replacing' })
    try {
      const updated = await replaceLineup(idx)
      dispatch({ type: 'set_portfolio', portfolio: updated })
    } catch {
      fetchPortfolio()
        .then(p => { if (p.length > 0) dispatch({ type: 'set_portfolio', portfolio: p }) })
        .catch(() => {})
    } finally {
      setReplacingIndex(null)
      dispatch({ type: 'set_run_status', status: 'complete' })
    }
  }

  const tabDisabled = (tab: Tab): boolean => {
    if (tab === 'portfolio' && state.portfolio.length === 0) return true
    if (tab === 'metrics' && state.portfolio.length === 0) return true
    return false
  }

  // Suppress unused import warning
  void sseStatus

  return (
    <div className="app">
      <header className="app-header">
        <h1>MLB DFS Optimizer</h1>
        <div className="header-actions">
          <span className={`status-badge status-${state.runStatus}`}>
            {state.runStatus}
          </span>
          {running && (
            <button
              className="btn-stop"
              onClick={handleStop}
              disabled={stopPending}
            >
              Stop
            </button>
          )}
          <button
            className="btn-run"
            onClick={handleRun}
            disabled={running || state.runStatus === 'replacing' || state.config === null}
          >
            {running ? 'Running…' : 'Run Portfolio'}
          </button>
        </div>
      </header>

      <nav className="tabs">
        {(['config', 'slate', 'run', 'portfolio', 'metrics'] as Tab[]).map(tab => (
          <button
            key={tab}
            className={`tab ${state.activeTab === tab ? 'active' : ''}`}
            onClick={() => dispatch({ type: 'set_tab', tab })}
            disabled={tabDisabled(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
            {tab === 'run' && running && <span className="tab-dot" />}
            {tab === 'portfolio' && state.portfolio.length > 0 && (
              <span className="tab-count">{state.portfolio.length}</span>
            )}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {state.activeTab === 'config' && (
          <div>
            {configError && <p className="error">{configError}</p>}
            {state.config ? (
              <>
                <ProjectionsPanel disabled={running} onFetched={refreshUnconfirmed} />
                <ConfigForm
                  config={state.config}
                  onSaved={cfg => dispatch({ type: 'set_config', config: cfg })}
                  disabled={running}
                />
              </>
            ) : (
              <p className="muted">Loading config…</p>
            )}
          </div>
        )}

        {state.activeTab === 'slate' && (
          <SlatePanel disabled={running} />
        )}

        {state.activeTab === 'run' && (
          <ProgressPanel events={events} running={running} />
        )}

        {state.activeTab === 'portfolio' && (
          <PortfolioTable
            lineups={state.portfolio}
            unconfirmedPlayerIds={state.unconfirmedPlayerIds}
            onDeleteLineup={state.runStatus === 'complete' ? handleDeleteLineup : undefined}
            replacingLineupIndex={replacingIndex}
          />
        )}

        {state.activeTab === 'metrics' && (
          <MetricsPanel lineups={state.portfolio} events={events} />
        )}
      </main>

      {pendingDeleteIndex !== null && (
        <DeleteConfirmModal
          lineupIndex={pendingDeleteIndex}
          onConfirm={handleConfirmDelete}
          onCancel={() => setPendingDeleteIndex(null)}
        />
      )}

      {showUploadDialog && (
        <StopUploadDialog
          lineupCount={stoppedLineupCount}
          onConfirm={handleWriteUpload}
          onDismiss={() => setShowUploadDialog(false)}
        />
      )}
    </div>
  )
}
