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

export function ProgressPanel({ events, running }: Props) {
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

  return (
    <div className="progress-panel">
      <h3>
        Run Progress
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
        {events.filter(e => e.stage !== 'optimize_lineup').map((e, i) => (
          <div key={i} className={`event-row event-${e.stage}`}>
            <span className="event-stage">{STAGE_LABELS[e.stage] ?? e.stage}</span>
            <span className="event-detail">{renderDetail(e)}</span>
          </div>
        ))}
        {running && !latestLineup && (
          <div className="event-row">
            <span className="event-stage muted">…</span>
          </div>
        )}
      </div>

      {events.some(e => e.stage === 'optimize_lineup') && (
        <div className="event-list event-list-two-col">
          {events.filter(e => e.stage === 'optimize_lineup').map((e, i) => (
            <div key={i} className="event-row event-optimize_lineup">
              <span className="event-stage">{STAGE_LABELS[e.stage] ?? e.stage}</span>
              <span className="event-detail">{renderDetail(e)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
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
      const ev = e as unknown as { payout_beta: number }
      return `Payout beta: ${ev.payout_beta}`
    }
    case 'optimize_lineup': {
      const ev = e as OptimizeLineupEvent
      return `Lineup ${ev.lineup_index}/${ev.total} — ${ev.sims_covered.toLocaleString()} sims removed, ${ev.sims_remaining.toLocaleString()} remaining`
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
