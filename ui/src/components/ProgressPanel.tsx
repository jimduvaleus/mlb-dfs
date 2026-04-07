import { useEffect, useRef, useState } from 'react'
import type { SSEEvent, OptimizeLineupEvent } from '../types'

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
  compute_target: 'Compute target',
  calibrate_beta: 'Calibrate beta',
  optimize_lineup: 'Optimize lineups',
  complete: 'Complete',
  stopped: 'Stopped',
  error: 'Error',
}

const CONFIG_STAGES = new Set(['simulate', 'compute_target', 'calibrate_beta'])

export function ProgressPanel({ events, running }: Props) {
  const [now, setNow] = useState(() => Date.now())
  const tickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastLineupTimeRef = useRef<number | null>(null)

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

  // Update now on each optimize_lineup event; schedule 30s-boundary ticks until the next lineup
  useEffect(() => {
    const last = events[events.length - 1]
    if (last?.stage !== 'optimize_lineup') return

    const ts = Date.now()
    lastLineupTimeRef.current = ts
    setNow(ts)

    if (tickTimerRef.current) clearTimeout(tickTimerRef.current)

    function scheduleTick() {
      const lineupTime = lastLineupTimeRef.current!
      const sinceLineup = Date.now() - lineupTime
      const nextTick = Math.ceil((sinceLineup + 1) / 30000) * 30000
      tickTimerRef.current = setTimeout(() => {
        setNow(Date.now())
        if (lastLineupTimeRef.current === lineupTime) scheduleTick()
      }, nextTick - sinceLineup)
    }

    scheduleTick()
  }, [events])

  if (events.length === 0 && !running) return null

  const first = events[0]
  const last = events[events.length - 1]
  const elapsed = first && last ? last.timestamp - first.timestamp : null

  const latestLineup = [...events]
    .reverse()
    .find(e => e.stage === 'optimize_lineup') as OptimizeLineupEvent | undefined

  const total = latestLineup?.total ?? 0
  const current = latestLineup?.lineup_index ?? 0
  const pct = total > 0 ? Math.round((current / total) * 100) : 0

  // ETA: use the last 3 lineup intervals (recent performance) to avoid optimism
  // from fast early lineups skewing the average downward.
  const lineupEvents = events.filter(e => e.stage === 'optimize_lineup') as OptimizeLineupEvent[]
  let etaMs: number | null = null
  if (running && current > 0 && total > current) {
    let avgPerLineup: number
    const recent = lineupEvents.slice(-4) // up to 4 events → 3 intervals
    if (recent.length >= 2) {
      const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
      avgPerLineup = recentElapsed / (recent.length - 1)
    } else {
      // Only one lineup done — use time since it completed as a floor
      avgPerLineup = now - recent[0].timestamp
    }
    etaMs = avgPerLineup * (total - current)
  }

  const liveElapsedMs = running && first && current > 0 ? now - first.timestamp : null

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

      {(running || latestLineup) && total > 0 && (
        <div className="progress-bar-wrap">
          <div className="progress-bar" style={{ width: `${pct}%` }} />
          <span className="progress-label">
            Lineup {current} / {total}
          </span>
        </div>
      )}

      <div className="event-list">
        {buildDisplayEvents(events).map((item, i) => (
          <div key={i} className={`event-row event-${item.stage}`}>
            <span className="event-stage">{item.label}</span>
            <span className="event-detail">{item.detail}</span>
          </div>
        ))}
        {running && !latestLineup && (
          <div className="event-row">
            <span className="event-stage muted">…</span>
          </div>
        )}
      </div>

      {events.some(e => e.stage === 'optimize_lineup') && (
        <div className="event-list event-list-three-col">
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
    </div>
  )
}

function buildConfigDetail(events: SSEEvent[]): string {
  const sim = events.find(e => e.stage === 'simulate') as unknown as { n_sims: number } | undefined
  const target = events.find(e => e.stage === 'compute_target') as unknown as { target: number; percentile: number | null } | undefined
  const beta = [...events].reverse().find(e => e.stage === 'calibrate_beta') as unknown as { payout_beta?: number } | undefined

  const parts: string[] = []
  if (sim) parts.push(`${sim.n_sims.toLocaleString()} simulations`)
  if (target) {
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

  for (const e of events) {
    if (e.stage === 'optimize_lineup' || e.stage === 'upload_files') continue
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
      }
      const multiPitcher = ev.multi_pitcher_teams && Object.keys(ev.multi_pitcher_teams).length > 0
        ? ` (${Object.entries(ev.multi_pitcher_teams).map(([t, n]) => `${n} ${t}`).join(', ')})`
        : ''
      const loaded = `${ev.n_teams} teams, ${ev.n_batters} batters, ${ev.n_pitchers} pitchers${multiPitcher} loaded`
      const exclParts: string[] = []
      if (ev.n_teams_excluded > 0) exclParts.push(`${ev.n_teams_excluded} team${ev.n_teams_excluded !== 1 ? 's' : ''}`)
      if (ev.n_batters_ind_excluded > 0) exclParts.push(`${ev.n_batters_ind_excluded} batter${ev.n_batters_ind_excluded !== 1 ? 's' : ''}`)
      if (ev.n_pitchers_ind_excluded > 0) exclParts.push(`${ev.n_pitchers_ind_excluded} pitcher${ev.n_pitchers_ind_excluded !== 1 ? 's' : ''}`)
      return exclParts.length > 0 ? `${loaded}. ${exclParts.join(', ')} excluded` : loaded
    }
    case 'simulate': {
      const ev = e as unknown as { n_sims: number }
      return `${ev.n_sims.toLocaleString()} simulations`
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
      if (ev.objective === 'marginal_payout') {
        const p90 = ev.p90 != null ? ev.p90.toFixed(1) : '—'
        const ptLabel = ev.target_percentile != null ? `p${ev.target_percentile}` : 'target'
        const ptVal = ev.p_target != null ? ev.p_target.toFixed(1) : '—'
        const p99 = ev.p99 != null ? ev.p99.toFixed(1) : '—'
        return `p90: ${p90} · ${ptLabel}: ${ptVal} · p99: ${p99}`
      }
      return `Lineup ${ev.lineup_index}/${ev.total} — ${(ev.sims_covered ?? 0).toLocaleString()} sims removed, ${(ev.sims_remaining ?? 0).toLocaleString()} remaining`
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
