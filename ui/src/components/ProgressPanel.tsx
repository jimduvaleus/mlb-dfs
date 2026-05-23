import { useEffect, useRef, useState } from 'react'
import type { SSEEvent, SimulateEvent, OptimizeLineupEvent, GppGenerateProgressEvent, GppFieldProgressEvent, GppScoreProgressEvent, GppSelectProgressEvent } from '../types'

interface Props {
  events: SSEEvent[]
  running: boolean
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  const m = Math.floor(ms / 60000)
  const s = Math.round((ms % 60000) / 1000)
  return `${m}m ${s}s`
}

function formatMsWhole(ms: number): string {
  if (ms < 60000) return `${Math.round(ms / 1000)}s`
  const m = Math.floor(ms / 60000)
  const s = Math.round((ms % 60000) / 1000)
  return `${m}m ${s}s`
}

const STAGE_LABELS: Record<string, string> = {
  load_slate: 'Load slate',
  simulate: 'Simulate',
  ppd_applied: 'PPD applied',
  compute_target: 'Compute target',
  calibrate_beta: 'Calibrate beta',
  optimize_lineup: 'Optimize lineups',
  gpp_generate_start: 'Generate candidates',
  gpp_generate_done: 'Generate candidates',
  gpp_score_start: 'Score candidates',
  gpp_score_done: 'Score candidates',
  gpp_field_inject: 'Field lineups',
  gpp_holdout: 'Holdout evaluation',
  complete: 'Complete',
  stopped: 'Stopped',
  error: 'Error',
}

const CONFIG_STAGES = new Set(['simulate', 'compute_target', 'calibrate_beta'])
const GPP_PROGRESS_STAGES = new Set(['gpp_generate_progress', 'gpp_score_progress', 'gpp_select_progress'])

export function ProgressPanel({ events, running }: Props) {
  const [now, setNow] = useState(() => Date.now())
  const tickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastEventTimeRef = useRef<number | null>(null)

  // Clear tick timer when run stops
  useEffect(() => {
    if (!running && tickTimerRef.current) {
      clearTimeout(tickTimerRef.current)
      tickTimerRef.current = null
    }
  }, [running])

  // Cleanup on unmount
  useEffect(() => {
    return () => { if (tickTimerRef.current) clearTimeout(tickTimerRef.current) }
  }, [])

  // Update now on each live progress event; schedule 30s-boundary ticks until the next event
  useEffect(() => {
    const last = events[events.length - 1]
    const isLiveProgressEvent =
      last?.stage === 'optimize_lineup' ||
      last?.stage === 'gpp_generate_start' ||
      last?.stage === 'gpp_generate_progress' ||
      last?.stage === 'gpp_generate_done' ||
      last?.stage === 'gpp_score_start' ||
      last?.stage === 'gpp_field_progress' ||
      last?.stage === 'gpp_score_progress' ||
      last?.stage === 'gpp_score_done' ||
      last?.stage === 'gpp_field_inject' ||
      last?.stage === 'gpp_select_progress'
    if (!isLiveProgressEvent) return

    const ts = Date.now()
    lastEventTimeRef.current = ts
    setNow(ts)

    if (tickTimerRef.current) clearTimeout(tickTimerRef.current)

    function scheduleTick() {
      const eventTime = lastEventTimeRef.current!
      const sinceEvent = Date.now() - eventTime
      const nextTick = Math.ceil((sinceEvent + 1) / 30000) * 30000
      tickTimerRef.current = setTimeout(() => {
        setNow(Date.now())
        if (lastEventTimeRef.current === eventTime) scheduleTick()
      }, nextTick - sinceEvent)
    }

    scheduleTick()
  }, [events])

  if (events.length === 0 && !running) return null

  const first = events[0]
  const last = events[events.length - 1]
  const elapsed = first && last ? last.timestamp - first.timestamp : null

  const isGpp = events.some(e =>
    e.stage === 'gpp_generate_start' || e.stage === 'gpp_generate_done' ||
    e.stage === 'gpp_score_start' || e.stage === 'gpp_field_inject' ||
    e.stage === 'gpp_select_progress'
  )

  // --- Non-GPP lineup progress ---
  const latestLineup = [...events]
    .reverse()
    .find(e => e.stage === 'optimize_lineup') as OptimizeLineupEvent | undefined

  const total = latestLineup?.total ?? 0
  const current = latestLineup?.lineup_index ?? 0
  const pct = total > 0 ? Math.round((current / total) * 100) : 0

  const lineupEvents = events.filter(e => e.stage === 'optimize_lineup') as OptimizeLineupEvent[]

  // --- GPP progress ---
  const generateStartEvent = events.find(e => e.stage === 'gpp_generate_start') as unknown as { n_candidates: number } | undefined
  const generateProgressEvents = events.filter(e => e.stage === 'gpp_generate_progress') as unknown as GppGenerateProgressEvent[]
  const latestGenerateProgress = generateProgressEvents[generateProgressEvents.length - 1]
  const generateDone = events.some(e => e.stage === 'gpp_generate_done')
  const scoreStartEvent = events.find(e => e.stage === 'gpp_score_start') as unknown as { n_field_lineups: number; n_field_samples: number } | undefined
  const fieldProgressEvents = events.filter(e => e.stage === 'gpp_field_progress') as unknown as GppFieldProgressEvent[]
  const latestFieldProgress = fieldProgressEvents[fieldProgressEvents.length - 1]
  const scoreProgressEvents = events.filter(e => e.stage === 'gpp_score_progress') as unknown as GppScoreProgressEvent[]
  const latestScoreProgress = scoreProgressEvents[scoreProgressEvents.length - 1]
  const selectProgressEvents = events.filter(e => e.stage === 'gpp_select_progress') as unknown as GppSelectProgressEvent[]
  const scoreDone = events.some(e => e.stage === 'gpp_score_done')
  const fieldInjectEvent = events.find(e => e.stage === 'gpp_field_inject') as unknown as { n_field: number; n_k: number } | undefined

  let gppPct = 0
  let gppLabel = ''
  if (isGpp) {
    if (selectProgressEvents.length > 0) {
      const lastSel = selectProgressEvents[selectProgressEvents.length - 1]
      gppPct = 100
      gppLabel = `Portfolio selection: round ${lastSel.round + 1} — ${lastSel.pct_covered.toFixed(1)}% covered`
    } else if (latestScoreProgress) {
      gppPct = Math.round((latestScoreProgress.batches_done / latestScoreProgress.batches_total) * 100)
      gppLabel = `Scoring batch ${latestScoreProgress.batches_done} / ${latestScoreProgress.batches_total}`
    } else if (fieldInjectEvent) {
      gppPct = 100
      gppLabel = `Field lineups loaded from cache`
    } else if (latestFieldProgress && !scoreDone) {
      gppPct = Math.round((latestFieldProgress.n_done / latestFieldProgress.n_total) * 100)
      gppLabel = `Generating field: ${latestFieldProgress.n_done.toLocaleString()} / ${latestFieldProgress.n_total.toLocaleString()} lineups`
    } else if (scoreStartEvent && fieldProgressEvents.length === 0) {
      const nTotal = scoreStartEvent.n_field_lineups * scoreStartEvent.n_field_samples
      gppPct = 0
      gppLabel = `Generating field: 0 / ${nTotal.toLocaleString()} lineups`
    } else if (generateDone) {
      gppPct = 100
      gppLabel = 'Candidates generated'
    } else if (latestGenerateProgress && generateStartEvent) {
      gppPct = Math.round((latestGenerateProgress.n / generateStartEvent.n_candidates) * 100)
      gppLabel = `Generating candidates: ${latestGenerateProgress.n.toLocaleString()} / ${generateStartEvent.n_candidates.toLocaleString()}`
    } else if (generateStartEvent) {
      gppPct = 0
      gppLabel = 'Generating candidates…'
    }
  }

  // --- ETA ---
  let etaMs: number | null = null
  if (isGpp) {
    if (running && latestGenerateProgress && !generateDone && generateStartEvent) {
      const recent = generateProgressEvents.slice(-4)
      if (recent.length >= 2) {
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const avgPerChunk = recentElapsed / (recent.length - 1)
        const remainingChunks = (generateStartEvent.n_candidates - latestGenerateProgress.n) / 500
        if (remainingChunks > 0) etaMs = avgPerChunk * remainingChunks
      }
    } else if (running && latestFieldProgress && !scoreDone && latestScoreProgress === undefined) {
      const remaining = latestFieldProgress.n_total - latestFieldProgress.n_done
      if (remaining > 0 && fieldProgressEvents.length >= 2) {
        const recent = fieldProgressEvents.slice(-4)
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const recentLineups = recent[recent.length - 1].n_done - recent[0].n_done
        if (recentLineups > 0) etaMs = (recentElapsed / recentLineups) * remaining
      }
    } else if (running && latestScoreProgress && !scoreDone) {
      const recent = scoreProgressEvents.slice(-4)
      if (recent.length >= 2) {
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const avgPerBatch = recentElapsed / (recent.length - 1)
        const remaining = latestScoreProgress.batches_total - latestScoreProgress.batches_done
        if (remaining > 0) etaMs = avgPerBatch * remaining
      }
    }
  } else {
    if (running && current > 0 && total > current) {
      const recent = lineupEvents.slice(-4) // up to 4 events → 3 intervals
      if (recent.length >= 2) {
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const avgPerLineup = recentElapsed / (recent.length - 1)
        etaMs = avgPerLineup * (total - current)
      }
    }
  }

  const liveElapsedMs = running && first && (current > 0 || isGpp) ? now - first.timestamp : null

  return (
    <div className="progress-panel">
      <h3>
        Run Progress
        {liveElapsedMs !== null && (
          <span className="muted" style={{ marginLeft: 8, fontWeight: 400, fontSize: '0.9em' }}>
            {formatMsWhole(liveElapsedMs)} elapsed
            {etaMs !== null && etaMs > 0 && (
              <span style={{ marginLeft: 12 }}>
                ~{formatMsWhole(etaMs)} remaining
              </span>
            )}
          </span>
        )}
        {elapsed !== null && !running && (
          <span className="muted" style={{ marginLeft: 8, fontWeight: 400, fontSize: '0.9em' }}>
            ({formatMs(elapsed)} total)
          </span>
        )}
      </h3>

      {/* Non-GPP progress bar */}
      {!isGpp && (running || latestLineup) && total > 0 && (
        <div className="progress-bar-wrap">
          <div className="progress-bar" style={{ width: `${pct}%` }} />
          <span className="progress-label">
            Lineup {current} / {total}
          </span>
        </div>
      )}

      {/* GPP progress bar */}
      {isGpp && running && gppLabel && (
        <div className="progress-bar-wrap">
          <div className="progress-bar" style={{ width: `${gppPct}%` }} />
          <span className="progress-label">{gppLabel}</span>
        </div>
      )}

      <div className="event-list">
        {buildDisplayEvents(events).map((item, i) => (
          <div key={i} className={`event-row event-${item.stage}`}>
            <span className="event-stage">{item.label}</span>
            <span className="event-detail">{item.detail}</span>
          </div>
        ))}
        {running && !latestLineup && !isGpp && (
          <div className="event-row">
            <span className="event-stage muted">…</span>
          </div>
        )}
      </div>

      {/* Non-GPP lineup grid */}
      {events.some(e => e.stage === 'optimize_lineup') && (
        <div className="event-list event-list-four-col">
          {events.filter(e => e.stage === 'optimize_lineup').map((e, i) => {
            const ev = e as OptimizeLineupEvent
            return (
              <div key={i} className="event-row event-optimize_lineup">
                <span className="event-stage event-stage-lineup">{ev.lineup_index}/{ev.total}</span>
                <span className="event-detail">{renderDetail(e)}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* GPP selection grid */}
      {selectProgressEvents.length > 0 && (
        <div className="event-list event-list-four-col">
          {selectProgressEvents.map((ev, i) => (
            <div key={i} className="event-row event-gpp_select_progress">
              <span className="event-stage event-stage-lineup">{ev.round + 1}</span>
              <span className="event-detail">
                EV ${ev.lineup_ev.toFixed(2)} · {ev.n_covered.toLocaleString()} sims ({ev.pct_covered.toFixed(1)}%)
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function buildConfigDetail(events: SSEEvent[]): string {
  const sim = events.find(e => e.stage === 'simulate') as SimulateEvent | undefined
  const target = events.find(e => e.stage === 'compute_target') as unknown as { target: number; percentile: number | null } | undefined
  const beta = [...events].reverse().find(e => e.stage === 'calibrate_beta') as unknown as { payout_beta?: number } | undefined

  const parts: string[] = []
  if (sim) parts.push(`${sim.n_sims.toLocaleString()} simulations`)
  if (target && sim?.objective !== 'leverage_surplus') {
    const tStr = target.percentile
      ? `${target.target.toFixed(1)} pts (p${target.percentile})`
      : `${target.target.toFixed(1)} pts (manual)`
    parts.push(`target: ${tStr}`)
  }
  if (beta?.payout_beta != null) parts.push(`payout beta: ${beta.payout_beta}`)
  return parts.join(', ')
}

function buildDisplayEvents(events: SSEEvent[]): Array<{ stage: string; label: string; detail: string }> {
  const result: Array<{ stage: string; label: string; detail: string }> = []
  let configInserted = false

  const hasEvent = (stage: string) => events.some(e => e.stage === stage)

  for (const e of events) {
    if (e.stage === 'optimize_lineup' || e.stage === 'upload_files') continue
    if (GPP_PROGRESS_STAGES.has(e.stage) || e.stage === 'gpp_field_progress') continue
    // Skip start event once done event is present (collapse into one row)
    if (e.stage === 'gpp_generate_start' && hasEvent('gpp_generate_done')) continue
    if (e.stage === 'gpp_score_start' && hasEvent('gpp_score_done')) continue
    // field_inject is a one-shot cache notification; skip once score is done
    if (e.stage === 'gpp_field_inject' && hasEvent('gpp_score_done')) continue
    if (CONFIG_STAGES.has(e.stage)) {
      if (!configInserted) {
        result.push({ stage: 'config', label: 'Configuration', detail: buildConfigDetail(events) })
        configInserted = true
      }
      continue
    }
    result.push({ stage: e.stage, label: STAGE_LABELS[e.stage] ?? e.stage, detail: renderDetail(e) })
  }

  return result
}

function renderDetail(e: SSEEvent): string {
  switch (e.stage) {
    case 'load_slate': {
      const ev = e as unknown as {
        n_teams: number; n_batters: number; n_pitchers: number;
        multi_pitcher_teams: Record<string, number>;
        n_teams_excluded: number; n_batters_ind_excluded: number; n_pitchers_ind_excluded: number;
        n_pitchers_value_excluded: number; n_batters_value_excluded: number;
      }
      const multiPitcher = ev.multi_pitcher_teams && Object.keys(ev.multi_pitcher_teams).length > 0
        ? ` (${Object.entries(ev.multi_pitcher_teams).map(([t, n]) => `${n} ${t}`).join(', ')})`
        : ''
      const loaded = `${ev.n_teams} teams, ${ev.n_batters} batters, ${ev.n_pitchers} pitchers${multiPitcher} loaded`
      const exclParts: string[] = []
      if (ev.n_teams_excluded > 0) exclParts.push(`${ev.n_teams_excluded} team${ev.n_teams_excluded !== 1 ? 's' : ''}`)
      if (ev.n_batters_ind_excluded > 0) exclParts.push(`${ev.n_batters_ind_excluded} batter${ev.n_batters_ind_excluded !== 1 ? 's' : ''}`)
      if (ev.n_pitchers_ind_excluded > 0) exclParts.push(`${ev.n_pitchers_ind_excluded} pitcher${ev.n_pitchers_ind_excluded !== 1 ? 's' : ''}`)
      const valueParts: string[] = []
      if (ev.n_pitchers_value_excluded > 0) valueParts.push(`${ev.n_pitchers_value_excluded} pitcher${ev.n_pitchers_value_excluded !== 1 ? 's' : ''}`)
      if (ev.n_batters_value_excluded > 0) valueParts.push(`${ev.n_batters_value_excluded} batter${ev.n_batters_value_excluded !== 1 ? 's' : ''}`)
      let detail = exclParts.length > 0 ? `${loaded}. ${exclParts.join(', ')} excluded` : loaded
      if (valueParts.length > 0) detail += `. ${valueParts.join(', ')} below value cutoff`
      return detail
    }
    case 'simulate': {
      const ev = e as unknown as { n_sims: number }
      return `${ev.n_sims.toLocaleString()} simulations`
    }
    case 'ppd_applied': {
      const ev = e as unknown as { games: { game: string; ppd_pct: number; n_sims_zeroed: number }[]; n_sims_total: number }
      const parts = ev.games.map(g => `${g.game} ${g.ppd_pct}% (${g.n_sims_zeroed.toLocaleString()} sims)`)
      return `${parts.join(', ')} — zeroed independently`
    }
    case 'compute_target': {
      const ev = e as unknown as { target: number; percentile: number | null }
      return ev.percentile
        ? `Target: ${ev.target.toFixed(1)} pts (p${ev.percentile})`
        : `Target: ${ev.target.toFixed(1)} pts (manual)`
    }
    case 'calibrate_beta': {
      const ev = e as unknown as { payout_beta?: number; payout_cash_line?: number }
      return ev.payout_beta != null ? `Payout beta: ${ev.payout_beta}` : 'Calibrating…'
    }
    case 'optimize_lineup': {
      const ev = e as OptimizeLineupEvent
      const pctFmt = (v: number | null | undefined) => v != null ? `${v.toFixed(1)}%` : '—'
      const ptLabel = ev.target_percentile != null ? `p${ev.target_percentile}` : 'target'
      return `p90: ${pctFmt(ev.pct_above_p90)} · ${ptLabel}: ${pctFmt(ev.pct_above_target)} · p99: ${pctFmt(ev.pct_above_p99)}`
    }
    case 'gpp_generate_start': {
      const ev = e as unknown as { n_candidates: number; n_from_cache?: number }
      if (ev.n_from_cache != null && ev.n_from_cache > 0) {
        return `Generating ${ev.n_candidates.toLocaleString()} + ${ev.n_from_cache.toLocaleString()} from cache…`
      }
      return `Generating ${ev.n_candidates.toLocaleString()} candidates…`
    }
    case 'gpp_generate_done': {
      const ev = e as unknown as { n_generated: number; from_cache?: boolean }
      const suffix = ev.from_cache ? ' (from cache)' : ''
      return `${ev.n_generated.toLocaleString()} candidates${suffix}`
    }
    case 'gpp_field_inject': {
      const ev = e as unknown as { n_field: number; n_k: number }
      return `${(ev.n_field * ev.n_k).toLocaleString()} lineups loaded from cache`
    }
    case 'gpp_score_start': {
      const ev = e as unknown as { n_candidates: number; n_field_samples: number }
      return `${ev.n_candidates.toLocaleString()} candidates × ${ev.n_field_samples} field samples`
    }
    case 'gpp_score_done':
      return 'Scoring complete'
    case 'gpp_holdout': {
      const ev = e as unknown as { holdout_mean_payout: number }
      return `Holdout mean payout: ${ev.holdout_mean_payout.toFixed(4)}`
    }
    case 'complete': {
      const ev = e as unknown as { n_lineups: number }
      return `${ev.n_lineups} lineups built`
    }
    case 'error': {
      const ev = e as unknown as { message: string }
      return ev.message
    }
    default:
      return ''
  }
}
