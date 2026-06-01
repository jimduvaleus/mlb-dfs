import { useEffect, useReducer, useRef, useState } from 'react'
import type { AppConfig, CacheStatus, LineupResult, MergeInfo, PortfolioSweepEntry, ProjectionPlayerRow, RunStatus, CompleteEvent, StoppedEvent, TwitterLineupParseResponse, TwitterLineupRecord, TwitterLineupSaveRequest, TwitterNotification } from './types'
import { dismissNotification, dismissTwitterLineup, fetchCacheStatus, fetchConfig, fetchNotifications, fetchOptimalLineups, fetchPortfolio, fetchProjectionPlayers, fetchTeamTotals, fetchTwitterLineups, fetchUnconfirmedPlayerIds, parseTwitterLineup, replaceLineup, saveTwitterLineup, stopRun, writeUploadFiles } from './api'
import { useSSE } from './hooks/useSSE'
import { ConfigForm } from './components/ConfigForm'
import { ProjectionsPanel } from './components/ProjectionsPanel'
import { ProgressPanel } from './components/ProgressPanel'
import { PortfolioTable } from './components/PortfolioTable'
import { ProjectionsTable } from './components/ProjectionsTable'
import { MetricsPanel } from './components/MetricsPanel'
import { SlatePanel } from './components/SlatePanel'
import { StopUploadDialog } from './components/StopUploadDialog'
import { RunOptionsDialog } from './components/RunOptionsDialog'
import { DeleteConfirmModal } from './components/DeleteConfirmModal'
import { LineupParserDialog } from './components/LineupParserDialog'
import { ReselectDialog } from './components/ReselectDialog'
import './App.css'

type Tab = 'config' | 'projections' | 'slate' | 'run' | 'portfolio' | 'metrics'

interface State {
  config: AppConfig | null
  portfolio: LineupResult[]
  optimalLineups: LineupResult[]
  portfolioSweep: PortfolioSweepEntry[]
  activeRisk: number
  runStatus: RunStatus
  activeTab: Tab
  unconfirmedPlayerIds: number[]
  notifications: TwitterNotification[]
  twitterLineups: TwitterLineupRecord[]
}

type Action =
  | { type: 'set_config'; config: AppConfig }
  | { type: 'set_portfolio'; portfolio: LineupResult[] }
  | { type: 'set_optimal_lineups'; lineups: LineupResult[] }
  | { type: 'set_portfolio_sweep'; sweep: PortfolioSweepEntry[] }
  | { type: 'set_active_risk'; risk: number; lineups: LineupResult[] }
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
    case 'set_optimal_lineups':
      return { ...state, optimalLineups: action.lineups }
    case 'set_portfolio_sweep':
      return { ...state, portfolioSweep: action.sweep }
    case 'set_active_risk':
      return { ...state, activeRisk: action.risk, portfolio: action.lineups }
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
  optimalLineups: [],
  portfolioSweep: [],
  activeRisk: 1,
  runStatus: 'idle',
  activeTab: 'config',
  unconfirmedPlayerIds: [],
  notifications: [],
  twitterLineups: [],
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initial)
  const [configError, setConfigError] = useState<string | null>(null)
  const [projectionPlayers, setProjectionPlayers] = useState<ProjectionPlayerRow[]>([])
  const [teamTotals, setTeamTotals] = useState<Record<string, number>>({})
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
  const [showRunOptionsDialog, setShowRunOptionsDialog] = useState(false)
  const [showReselectDialog, setShowReselectDialog] = useState(false)
  const [pendingCacheStatus, setPendingCacheStatus] = useState<CacheStatus | null>(null)
  const [pendingDeleteIndex, setPendingDeleteIndex] = useState<number | null>(null)
  const [replacingIndex, setReplacingIndex] = useState<number | null>(null)
  const [replaceError, setReplaceError] = useState<string | null>(null)
  const [projStatusTrigger, setProjStatusTrigger] = useState(0)
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

  const refreshProjectionPlayers = () => {
    fetchProjectionPlayers()
      .then(setProjectionPlayers)
      .catch(() => {})
    fetchTeamTotals()
      .then(setTeamTotals)
      .catch(() => {})
  }

  const refreshTwitterLineups = () => {
    fetchTwitterLineups()
      .then(lineups => dispatch({ type: 'set_twitter_lineups', lineups }))
      .catch(() => {})
  }

  // Load config, existing portfolio, sweep portfolios, and unconfirmed player IDs on mount
  useEffect(() => {
    fetchConfig()
      .then(cfg => {
        dispatch({ type: 'set_config', config: cfg })
        return Promise.all([
          fetchPortfolio(cfg.platform),
          fetchOptimalLineups(cfg.platform),
          fetch('/api/portfolio/sweep').then(r => r.ok ? r.json() : { sweep: [], active_risk: 1 }),
        ])
      })
      .then(([portfolio, optimalLineups, sweepData]) => {
        const sweep: PortfolioSweepEntry[] = sweepData.sweep ?? []
        const activeRisk: number = sweepData.active_risk ?? 1
        if (sweep.length > 0) {
          dispatch({ type: 'set_portfolio_sweep', sweep })
          const activeEntry = sweep.find(e => e.risk === activeRisk) ?? sweep[0]
          if (activeEntry) {
            dispatch({ type: 'set_active_risk', risk: activeEntry.risk, lineups: activeEntry.lineups })
            dispatch({ type: 'set_run_status', status: 'complete' })
          }
        } else if (portfolio.length > 0) {
          dispatch({ type: 'set_portfolio', portfolio })
          dispatch({ type: 'set_run_status', status: 'complete' })
        }
        if (optimalLineups.length > 0) {
          dispatch({ type: 'set_optimal_lineups', lineups: optimalLineups })
        }
      })
      .catch(e => setConfigError(String(e)))
    refreshUnconfirmed()
    refreshTwitterLineups()
    refreshProjectionPlayers()
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
      setMergeInfo(null)
      setProjStatusTrigger(t => t + 1)
      // Slate path changed — clear optimal lineups (server will also reject stale fingerprint)
      dispatch({ type: 'set_optimal_lineups', lineups: [] })
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
        const sweep = ce.portfolio_sweep ?? []
        const defaultEntry = sweep.find(e => e.risk === 1) ?? sweep[0]
        const defaultLineups = defaultEntry ? defaultEntry.lineups : ce.portfolio
        dispatch({ type: 'set_portfolio', portfolio: defaultLineups })
        dispatch({ type: 'set_optimal_lineups', lineups: ce.optimal_lineups ?? [] })
        dispatch({ type: 'set_portfolio_sweep', sweep })
        if (defaultEntry) dispatch({ type: 'set_active_risk', risk: defaultEntry.risk, lineups: defaultLineups })
        dispatch({ type: 'set_run_status', status: 'complete' })
        dispatch({ type: 'set_tab', tab: 'portfolio' })
      } else if (event.stage === 'stopped') {
        const se = event as StoppedEvent
        dispatch({ type: 'set_portfolio', portfolio: se.portfolio })
        dispatch({ type: 'set_optimal_lineups', lineups: se.optimal_lineups ?? [] })
        dispatch({ type: 'set_run_status', status: 'stopped' })
        dispatch({ type: 'set_tab', tab: 'portfolio' })
        setStopPending(false)
        if (se.n_lineups > 0) {
          setStoppedLineupCount(se.n_lineups)
          setShowUploadDialog(true)
        }
      } else if (event.stage === 'load_slate') {
        // New run started — clear stale sweep data.
        dispatch({ type: 'set_portfolio_sweep', sweep: [] })
      } else if (event.stage === 'error') {
        dispatch({ type: 'set_run_status', status: 'error' })
      }
    }
  }, [events])

  const _doStartRun = (useCandidates: boolean, useField: boolean, seedOptimal: boolean = false) => {
    setShowRunOptionsDialog(false)
    setPendingCacheStatus(null)
    resetSSE()
    setShowUploadDialog(false)
    setStopPending(false)
    dispatch({ type: 'set_run_status', status: 'running' })
    dispatch({ type: 'set_optimal_lineups', lineups: [] })
    dispatch({ type: 'set_tab', tab: 'run' })
    const params: Record<string, string> = {}
    if (useCandidates) params.use_candidates = 'true'
    if (useField) params.use_field = 'true'
    if (seedOptimal) params.seed_optimal = 'true'
    startSSE(Object.keys(params).length ? params : undefined)
  }

  const handleRun = async () => {
    if (running) return
    if (state.config?.optimizer?.objective === 'leverage_surplus') {
      try {
        const status = await fetchCacheStatus()
        if (status.is_gpp) {
          setPendingCacheStatus(status)
          setShowRunOptionsDialog(true)
          return
        }
      } catch {
        // fall through to immediate start
      }
    }
    _doStartRun(false, false)
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
    refreshProjectionPlayers()
    setProjStatusTrigger(t => t + 1)
  }

  const handleDismissTwitterLineup = async (team: string) => {
    await dismissTwitterLineup(team)
    refreshTwitterLineups()
    refreshUnconfirmed()
    refreshProjectionPlayers()
    setProjStatusTrigger(t => t + 1)
  }

  const handleWriteUpload = () => {
    setShowUploadDialog(false)
    writeUploadFiles().catch(() => {})
  }

  const handleActivateRisk = async (risk: number) => {
    // Immediately update displayed portfolio from sweep data.
    const sweepEntry = state.portfolioSweep.find(e => e.risk === risk)
    if (sweepEntry) {
      dispatch({ type: 'set_active_risk', risk, lineups: sweepEntry.lineups })
    }
    // Async: tell server to re-write output files for this risk.
    try {
      const res = await fetch('/api/portfolio/activate_risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ risk }),
      })
      if (res.ok) {
        const data = await res.json()
        // Use server response (may include entry info not in sweep data).
        dispatch({ type: 'set_active_risk', risk, lineups: data.lineups })
      }
    } catch {
      // File write failure is non-fatal; displayed portfolio already updated.
    }
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
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to replace lineup'
      setReplaceError(msg)
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
          {state.runStatus === 'complete' && state.config?.optimizer.objective === 'leverage_surplus' && state.config.gpp.portfolio_method === 'mean_variance' && (
            <button
              className="btn-refine"
              onClick={() => setShowReselectDialog(true)}
              disabled={running}
              title="Re-run SA selection with different settings — no simulation re-run needed"
            >
              Refine SA
            </button>
          )}
          <button
            className="btn-run"
            onClick={handleRun}
            disabled={running || state.runStatus === 'replacing' || state.runStatus === 'reselecting' || state.config === null}
          >
            {running ? 'Running…' : 'Run Portfolio'}
          </button>
        </div>
      </header>

      <nav className="tabs">
        {(['config', 'slate', 'projections', 'run', 'portfolio', 'metrics'] as Tab[]).map(tab => (
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
            <div className="config-layout">
              <ProjectionsPanel disabled={running} onFetched={() => { refreshUnconfirmed(); refreshProjectionPlayers() }} mergeInfo={mergeInfo} onMergeInfo={setMergeInfo} projFetchExcluded={projFetchExcluded} onFetchingChange={setProjFetching} refreshTrigger={projStatusTrigger} />
              <ConfigForm
                config={state.config}
                onSaved={cfg => {
                  const prevPlatform = state.config?.platform
                  dispatch({ type: 'set_config', config: cfg })
                  if (cfg.platform !== prevPlatform) {
                    refreshProjectionPlayers()
                    // Platform changed — load the portfolio and optimal lineups for the new platform
                    Promise.all([fetchPortfolio(cfg.platform), fetchOptimalLineups(cfg.platform)])
                      .then(([portfolio, optimalLineups]) => {
                        dispatch({ type: 'set_portfolio', portfolio })
                        dispatch({ type: 'set_run_status', status: portfolio.length > 0 ? 'complete' : 'idle' })
                        dispatch({ type: 'set_optimal_lineups', lineups: optimalLineups })
                      })
                      .catch(() => {
                        dispatch({ type: 'set_portfolio', portfolio: [] })
                        dispatch({ type: 'set_optimal_lineups', lineups: [] })
                        dispatch({ type: 'set_run_status', status: 'idle' })
                      })
                  }
                }}
                disabled={running}
              />
            </div>
          ) : (
            <p className="muted">Loading config…</p>
          )}
        </div>

        {state.activeTab === 'projections' && (
          <ProjectionsTable players={projectionPlayers} platform={state.config?.platform} teamTotals={teamTotals} onOwnershipSettingsChanged={refreshProjectionPlayers} />
        )}

        {state.activeTab === 'slate' && (
          <SlatePanel
            disabled={running}
            projFetchExcluded={projFetchExcluded}
            onProjFetchFilterChange={(excluded) => { setProjFetchExcluded(excluded); refreshProjectionPlayers() }}
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
            optimalLineups={state.optimalLineups}
            portfolioSweep={state.portfolioSweep}
            activeRisk={state.activeRisk}
            onActivateRisk={handleActivateRisk}
            unconfirmedPlayerIds={state.unconfirmedPlayerIds}
            onDeleteLineup={state.runStatus === 'complete' ? handleDeleteLineup : undefined}
            replacingLineupIndex={replacingIndex}
            platform={state.config?.platform}
          />
        )}
        {replaceError && state.activeTab === 'portfolio' && (
          <div className="parse-error-toast" onClick={() => setReplaceError(null)}>{replaceError}</div>
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

      {showRunOptionsDialog && pendingCacheStatus && (
        <RunOptionsDialog
          cacheStatus={pendingCacheStatus}
          onStart={(useCandidates, useField, seedOptimal) => _doStartRun(useCandidates, useField, seedOptimal)}
          onDismiss={() => { setShowRunOptionsDialog(false); setPendingCacheStatus(null) }}
        />
      )}
      {showReselectDialog && (
        <ReselectDialog
          isOpen={showReselectDialog}
          initialRisk={state.config?.gpp.risk ?? 5.0}
          initialNIter={state.config?.gpp.portfolio_n_iter ?? 10000}
          initialNRestarts={state.config?.gpp.portfolio_n_restarts ?? 6}
          onClose={() => setShowReselectDialog(false)}
          onComplete={portfolio => {
            dispatch({ type: 'set_portfolio', portfolio })
            dispatch({ type: 'set_tab', tab: 'portfolio' })
            setShowReselectDialog(false)
            fetchConfig().then(cfg => dispatch({ type: 'set_config', config: cfg })).catch(() => {})
          }}
        />
      )}
    </div>
  )
}
