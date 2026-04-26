import { useEffect, useReducer, useRef, useState } from 'react'
import type { AppConfig, LineupResult, MergeInfo, RunStatus, CompleteEvent, StoppedEvent, TwitterLineupParseResponse, TwitterLineupRecord, TwitterLineupSaveRequest, TwitterNotification } from './types'
import { dismissNotification, dismissTwitterLineup, fetchConfig, fetchNotifications, fetchPortfolio, fetchTwitterLineups, fetchUnconfirmedPlayerIds, parseTwitterLineup, replaceLineup, saveTwitterLineup, stopRun, writeUploadFiles } from './api'
import { useSSE } from './hooks/useSSE'
import { ConfigForm } from './components/ConfigForm'
import { ProjectionsPanel } from './components/ProjectionsPanel'
import { ProgressPanel } from './components/ProgressPanel'
import { PortfolioTable } from './components/PortfolioTable'
import { MetricsPanel } from './components/MetricsPanel'
import { SlatePanel } from './components/SlatePanel'
import { StopUploadDialog } from './components/StopUploadDialog'
import { DeleteConfirmModal } from './components/DeleteConfirmModal'
import { LineupParserDialog } from './components/LineupParserDialog'
import './App.css'

type Tab = 'config' | 'slate' | 'run' | 'portfolio' | 'metrics'

interface State {
  config: AppConfig | null
  portfolio: LineupResult[]
  runStatus: RunStatus
  activeTab: Tab
  unconfirmedPlayerIds: number[]
  notifications: TwitterNotification[]
  twitterLineups: TwitterLineupRecord[]
}

type Action =
  | { type: 'set_config'; config: AppConfig }
  | { type: 'set_portfolio'; portfolio: LineupResult[] }
  | { type: 'set_run_status'; status: RunStatus }
  | { type: 'set_tab'; tab: Tab }
  | { type: 'set_unconfirmed'; ids: number[] }
  | { type: 'set_notifications'; notifications: TwitterNotification[] }
  | { type: 'set_twitter_lineups'; lineups: TwitterLineupRecord[] }

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
    case 'set_notifications':
      return { ...state, notifications: action.notifications }
    case 'set_twitter_lineups':
      return { ...state, twitterLineups: action.lineups }
  }
}

const initial: State = {
  config: null,
  portfolio: [],
  runStatus: 'idle',
  activeTab: 'config',
  unconfirmedPlayerIds: [],
  notifications: [],
  twitterLineups: [],
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initial)
  const [configError, setConfigError] = useState<string | null>(null)
  const [mergeInfo, setMergeInfo] = useState<MergeInfo | null>(null)
  // Games (as "AWAY@HOME" strings) to exclude from projection fetches.
  // Empty = fetch all games (default). Non-empty = partial fetch + merge.
  // Stored per-platform in localStorage; restored once config/platform is known.
  const [projFetchExcluded, setProjFetchExcluded] = useState<string[]>([])
  const projFetchPlatformRef = useRef<string>('')
  const [projFetching, setProjFetching] = useState(false)
  const [showUploadDialog, setShowUploadDialog] = useState(false)
  const [stoppedLineupCount, setStoppedLineupCount] = useState(0)
  const [stopPending, setStopPending] = useState(false)
  const [pendingDeleteIndex, setPendingDeleteIndex] = useState<number | null>(null)
  const [replacingIndex, setReplacingIndex] = useState<number | null>(null)
  const [parsingNotification, setParsingNotification] = useState<TwitterNotification | null>(null)
  const [parseResult, setParseResult] = useState<TwitterLineupParseResponse | null>(null)
  const [parseError, setParseError] = useState<string | null>(null)
  const { events, status: sseStatus, start: startSSE, reset: resetSSE } = useSSE('/api/run/stream')

  const running = state.runStatus === 'running'

  const refreshUnconfirmed = () => {
    fetchUnconfirmedPlayerIds()
      .then(ids => dispatch({ type: 'set_unconfirmed', ids }))
      .catch(() => {})
  }

  const refreshTwitterLineups = () => {
    fetchTwitterLineups()
      .then(lineups => dispatch({ type: 'set_twitter_lineups', lineups }))
      .catch(() => {})
  }

  // Load config, existing portfolio, and unconfirmed player IDs on mount
  useEffect(() => {
    fetchConfig()
      .then(cfg => {
        dispatch({ type: 'set_config', config: cfg })
        // Load the platform-specific portfolio for the current platform
        return fetchPortfolio(cfg.platform)
      })
      .then(portfolio => {
        if (portfolio.length > 0) {
          dispatch({ type: 'set_portfolio', portfolio })
          dispatch({ type: 'set_run_status', status: 'complete' })
        }
      })
      .catch(e => setConfigError(String(e)))
    refreshUnconfirmed()
    refreshTwitterLineups()
  }, [])

  // Restore per-platform proj-fetch exclusions when the platform or salary file changes.
  // Storage format: { draftkings: string[], fanduel: string[] }
  // Invalidates (resets) a platform's exclusions when its salary file path changes.
  useEffect(() => {
    if (!state.config) return
    const { platform, paths } = state.config
    projFetchPlatformRef.current = platform
    const currentSlate = platform === 'fanduel' ? paths.fd_slate : paths.dk_slate
    const slateKey = `projFetchSlatePath_${platform}`
    const storedPath = localStorage.getItem(slateKey)
    const all: Record<string, string[]> = (() => {
      try { return JSON.parse(localStorage.getItem('projFetchExcluded') ?? '{}') } catch { return {} }
    })()
    if (storedPath !== null && storedPath !== currentSlate) {
      all[platform] = []
      localStorage.setItem('projFetchExcluded', JSON.stringify(all))
      setProjFetchExcluded([])
    } else {
      setProjFetchExcluded(Array.isArray(all[platform]) ? all[platform] : [])
    }
    localStorage.setItem(slateKey, currentSlate)
  }, [state.config?.platform, state.config?.paths.dk_slate, state.config?.paths.fd_slate])

  // Persist proj-fetch exclusions to the current platform's slot
  useEffect(() => {
    if (!projFetchPlatformRef.current) return
    const all: Record<string, string[]> = (() => {
      try { return JSON.parse(localStorage.getItem('projFetchExcluded') ?? '{}') } catch { return {} }
    })()
    all[projFetchPlatformRef.current] = projFetchExcluded
    localStorage.setItem('projFetchExcluded', JSON.stringify(all))
  }, [projFetchExcluded])

  // Update browser tab title with unread notification count
  useEffect(() => {
    const count = state.notifications.length
    document.title = count > 0 ? `MLB Portfolio Tool (${count})` : 'MLB Portfolio Tool'
  }, [state.notifications.length])

  // Poll for X/Twitter notifications every 5 seconds
  useEffect(() => {
    const poll = () => {
      fetchNotifications()
        .then(notifications => dispatch({ type: 'set_notifications', notifications }))
        .catch(() => {})
    }
    poll()
    const id = setInterval(poll, 5000)
    return () => clearInterval(id)
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

  const handleParseNotification = async (notif: TwitterNotification) => {
    setParseError(null)
    setParsingNotification(notif)
    try {
      const result = await parseTwitterLineup(notif.id, notif.body)
      if (result.team === null) {
        setParseError('Team name not recognized in this notification')
        setParsingNotification(null)
        return
      }
      setParseResult(result)
    } catch {
      setParseError('Failed to parse lineup')
      setParsingNotification(null)
    }
  }

  const handleConfirmTwitterLineup = async (req: TwitterLineupSaveRequest) => {
    await saveTwitterLineup(req)
    setParsingNotification(null)
    setParseResult(null)
    refreshTwitterLineups()
    refreshUnconfirmed()
  }

  const handleDismissTwitterLineup = async (team: string) => {
    await dismissTwitterLineup(team)
    refreshTwitterLineups()
    refreshUnconfirmed()
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
      <div className="sticky-top">
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
            {tab === 'config' && projFetching && <span className="tab-dot" />}
            {tab === 'run' && running && <span className="tab-dot" />}
            {tab === 'portfolio' && state.portfolio.length > 0 && (
              <span className="tab-count">{state.portfolio.length}</span>
            )}
            {tab === 'slate' && state.notifications.length > 0 && (
              <span className="tab-count">{state.notifications.length}</span>
            )}
          </button>
        ))}
      </nav>
      </div>

      <main className="app-main">
        {/* Always mounted so the projection fetch EventSource survives tab switches */}
        <div style={{ display: state.activeTab === 'config' ? undefined : 'none' }}>
          {configError && <p className="error">{configError}</p>}
          {state.config ? (
            <>
              <ProjectionsPanel disabled={running} onFetched={refreshUnconfirmed} mergeInfo={mergeInfo} onMergeInfo={setMergeInfo} projFetchExcluded={projFetchExcluded} onFetchingChange={setProjFetching} />
              <ConfigForm
                config={state.config}
                onSaved={cfg => {
                  const prevPlatform = state.config?.platform
                  dispatch({ type: 'set_config', config: cfg })
                  if (cfg.platform !== prevPlatform) {
                    // Platform changed — load the portfolio for the new platform (may be empty)
                    fetchPortfolio(cfg.platform)
                      .then(portfolio => {
                        dispatch({ type: 'set_portfolio', portfolio })
                        dispatch({ type: 'set_run_status', status: portfolio.length > 0 ? 'complete' : 'idle' })
                      })
                      .catch(() => {
                        dispatch({ type: 'set_portfolio', portfolio: [] })
                        dispatch({ type: 'set_run_status', status: 'idle' })
                      })
                  }
                }}
                disabled={running}
              />
            </>
          ) : (
            <p className="muted">Loading config…</p>
          )}
        </div>

        {state.activeTab === 'slate' && (
          <SlatePanel
            disabled={running}
            projFetchExcluded={projFetchExcluded}
            onProjFetchFilterChange={setProjFetchExcluded}
            platform={state.config?.platform}
            notifications={state.notifications}
            onDismissNotification={(id) => {
              dismissNotification(id)
              dispatch({ type: 'set_notifications', notifications: state.notifications.filter(n => n.id !== id) })
            }}
            twitterLineups={state.twitterLineups}
            onParseNotification={handleParseNotification}
            onDismissTwitterLineup={handleDismissTwitterLineup}
          />
        )}
        {parseError && state.activeTab === 'slate' && (
          <div className="parse-error-toast" onClick={() => setParseError(null)}>{parseError}</div>
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
            platform={state.config?.platform}
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

      {parsingNotification && parseResult && (
        <LineupParserDialog
          parseResult={parseResult}
          onConfirm={handleConfirmTwitterLineup}
          onCancel={() => { setParsingNotification(null); setParseResult(null) }}
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
