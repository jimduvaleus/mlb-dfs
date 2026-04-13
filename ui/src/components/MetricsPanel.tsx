import type { LineupResult, SSEEvent, OptimizeLineupEvent, PortfolioStatsEvent } from '../types'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, Legend, ReferenceLine,
} from 'recharts'
import { getStackNotation } from '../utils'
import TeamBadge from './TeamBadge'

interface Props {
  lineups: LineupResult[]
  events: SSEEvent[]
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  const m = Math.floor(ms / 60000)
  const s = Math.round((ms % 60000) / 1000)
  return `${m}m ${s}s`
}

export function MetricsPanel({ lineups, events }: Props) {
  if (lineups.length === 0) return null

  // --- Timing ---
  const firstEvent = events[0]
  const lastEvent = events[events.length - 1]
  const totalMs = firstEvent && lastEvent ? lastEvent.timestamp - firstEvent.timestamp : null

  const lineupEvents = events.filter(e => e.stage === 'optimize_lineup') as OptimizeLineupEvent[]
  let avgLineupMs: number | null = null
  if (lineupEvents.length >= 2) {
    const lineupStart = lineupEvents[0].timestamp
    const lineupEnd = lineupEvents[lineupEvents.length - 1].timestamp
    avgLineupMs = (lineupEnd - lineupStart) / (lineupEvents.length - 1)
  }

  // --- Coverage analysis (marginal_payout objective only) ---
  const portfolioStatsEvent = events.find(e => e.stage === 'portfolio_stats') as PortfolioStatsEvent | undefined

  const payoutLineupEvents = lineupEvents.filter(e => e.objective === 'marginal_payout')
  const coverageEvolution = payoutLineupEvents.map(ev => {
    const n = (ev.sims_great ?? 0) + (ev.sims_good ?? 0) + (ev.sims_uncovered ?? 0)
    if (n === 0) return null
    return {
      lineup: ev.lineup_index,
      great: Math.round(((ev.sims_great ?? 0) / n) * 100),
      good: Math.round(((ev.sims_good ?? 0) / n) * 100),
      uncovered: Math.round(((ev.sims_uncovered ?? 0) / n) * 100),
    }
  }).filter(Boolean) as { lineup: number; great: number; good: number; uncovered: number }[]

  // --- Exposure ---
  const pitcherExposure: Record<string, { name: string; team: string; count: number }> = {}
  const batterExposure: Record<string, { name: string; team: string; count: number }> = {}
  for (const lineup of lineups) {
    for (const p of lineup.players) {
      const key = String(p.player_id)
      const pos = p.assigned_position ?? p.position
      const bucket = pos === 'P' ? pitcherExposure : batterExposure
      if (!bucket[key]) {
        bucket[key] = { name: p.name, team: p.team, count: 0 }
      }
      bucket[key].count++
    }
  }
  const pitcherExposureList = Object.values(pitcherExposure).sort((a, b) => b.count - a.count)
  const batterExposureList = Object.values(batterExposure).sort((a, b) => b.count - a.count)

  // --- Value metrics ---
  // value = mean / (salary / 1000), scoped to players in the portfolio
  const pitcherValues: number[] = []
  const batterValues: number[] = []
  const seenPids = new Set<number>()
  for (const lineup of lineups) {
    for (const p of lineup.players) {
      if (seenPids.has(p.player_id)) continue
      seenPids.add(p.player_id)
      if (p.mean == null || !p.salary) continue
      const val = p.mean / (p.salary / 1000)
      const pos = p.assigned_position ?? p.position
      if (pos === 'P') {
        pitcherValues.push(val)
      } else {
        batterValues.push(val)
      }
    }
  }
  const minPitcherValue = pitcherValues.length > 0 ? Math.min(...pitcherValues) : null
  const minBatterValue = batterValues.length > 0 ? Math.min(...batterValues) : null

  // --- Salary distribution ---
  const salaries = lineups.map(l => l.lineup_salary)
  const minSal = Math.min(...salaries)
  const maxSal = Math.max(...salaries)
  const avgSal = Math.round(salaries.reduce((a, b) => a + b, 0) / salaries.length)
  const salaryBuckets: { label: string; count: number }[] = []
  const bucketSize = 500
  const bucketStart = Math.floor(minSal / bucketSize) * bucketSize
  const bucketEnd = Math.ceil(maxSal / bucketSize) * bucketSize
  for (let s = bucketStart; s < bucketEnd; s += bucketSize) {
    const count = salaries.filter(sal => sal >= s && sal < s + bucketSize).length
    salaryBuckets.push({ label: `$${(s / 1000).toFixed(1)}k`, count })
  }

  // --- Stacking breakdown ---
  const stackCounts: Record<string, number> = {}
  for (const lineup of lineups) {
    const stack = getStackNotation(lineup.players).split(' ')[0]
    stackCounts[stack] = (stackCounts[stack] ?? 0) + 1
  }
  const stackList = Object.entries(stackCounts).sort((a, b) => b[1] - a[1])

  return (
    <div className="metrics-panel">
      <h3>Portfolio Metrics</h3>

      {/* Timing — only shown when SSE events from a live run are present */}
      {(totalMs !== null || avgLineupMs !== null) && (
        <div className="metrics-section">
          <h4>Build Time</h4>
          <div className="metrics-row">
            {totalMs !== null && (
              <span className="metric-chip">Total: {formatMs(totalMs)}</span>
            )}
            {avgLineupMs !== null && (
              <span className="metric-chip">Avg/lineup: {formatMs(avgLineupMs)}</span>
            )}
          </div>
        </div>
      )}

      {/* Coverage analysis — marginal_payout objective only */}
      {(portfolioStatsEvent || coverageEvolution.length > 0) && (
        <div className="metrics-section">
          <h4>Coverage Analysis</h4>

          {/* Summary stats */}
          {portfolioStatsEvent && (() => {
            const s = portfolioStatsEvent
            const covPct = Math.round((s.covered_count / s.n_sims) * 100)
            const margin = (v: number | null) => v != null ? `+${(v - s.target).toFixed(1)}` : '—'
            return (
              <div>
                <div className="metrics-row" style={{ marginBottom: 6 }}>
                  <span className="metric-chip">Covered: {covPct}%</span>
                  {s.covered_mean != null && (
                    <span className="metric-chip">Avg margin: {margin(s.covered_mean)} pts</span>
                  )}
                  {s.covered_p50 != null && (
                    <span className="metric-chip">Median margin: {margin(s.covered_p50)} pts</span>
                  )}
                </div>
                <div className="metrics-row" style={{ marginBottom: 12 }}>
                  <span className="metric-chip" style={{ opacity: 0.75 }}>Portfolio p90: {s.overall_p90} pts</span>
                  <span className="metric-chip" style={{ opacity: 0.75 }}>p95: {s.overall_p95} pts</span>
                  <span className="metric-chip" style={{ opacity: 0.75 }}>p99: {s.overall_p99} pts</span>
                </div>

                {/* Score distribution histogram */}
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                  Best lineup score per simulation
                </div>
                <div style={{ height: 140 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={s.histogram} margin={{ top: 4, right: 4, left: 0, bottom: 4 }} barCategoryGap={1}>
                      <XAxis
                        dataKey="mid"
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={(v: number) => v.toFixed(0)}
                        tick={{ fontSize: 9 }}
                        interval="preserveStartEnd"
                      />
                      <YAxis allowDecimals={false} tick={{ fontSize: 9 }} width={24} />
                      <Tooltip
                        formatter={(v) => [`${v} sims`, '']}
                        labelFormatter={(mid) => `${Number(mid).toFixed(1)} pts`}
                      />
                      <ReferenceLine x={s.target} stroke="#fbbf24" strokeDasharray="4 2" strokeWidth={1.5} />
                      <ReferenceLine x={s.great_threshold} stroke="#4ade80" strokeDasharray="4 2" strokeWidth={1.5} />
                      <Bar dataKey="count" isAnimationActive={false}>
                        {s.histogram.map((bucket, i) => (
                          <Cell
                            key={i}
                            fill={
                              bucket.mid >= s.great_threshold ? '#4ade80' :
                              bucket.mid >= s.target ? '#fbbf24' :
                              '#64748b'
                            }
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ display: 'flex', gap: 12, fontSize: 10, color: 'var(--text-muted)', marginTop: 4, marginBottom: 12 }}>
                  <span style={{ color: '#64748b' }}>▌ Uncovered</span>
                  <span style={{ color: '#fbbf24' }}>▌ Good (target – {s.great_threshold.toFixed(0)} pts)</span>
                  <span style={{ color: '#4ade80' }}>▌ Great (≥{s.great_threshold.toFixed(0)} pts)</span>
                </div>
              </div>
            )
          })()}

          {/* Coverage evolution line chart */}
          {coverageEvolution.length > 1 && (
            <>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                Coverage by lineup added
              </div>
              <div style={{ height: 160 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={coverageEvolution} margin={{ top: 4, right: 4, left: 0, bottom: 4 }}>
                    <XAxis dataKey="lineup" tick={{ fontSize: 9 }} label={{ value: 'Lineup #', position: 'insideBottomRight', offset: -4, fontSize: 9 }} />
                    <YAxis domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} tick={{ fontSize: 9 }} width={32} />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Line type="monotone" dataKey="great" name="Great" stroke="#4ade80" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                    <Line type="monotone" dataKey="good" name="Good" stroke="#fbbf24" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                    <Line type="monotone" dataKey="uncovered" name="Uncovered" stroke="#64748b" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>
      )}

      {/* Stacking */}
      <div className="metrics-section">
        <h4>Stack Types</h4>
        <div className="stack-list">
          {stackList.map(([stack, count]) => (
            <div key={stack} className="stack-row">
              <span className="stack-tag">{stack || 'None'}</span>
              <span className="stack-bar-wrap">
                <span
                  className="stack-bar"
                  style={{ width: `${(count / lineups.length) * 100}%` }}
                />
              </span>
              <span className="stack-count">{count} lineup{count !== 1 ? 's' : ''}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Salary distribution */}
      <div className="metrics-section">
        <h4>Salary Distribution</h4>
        <div className="metrics-row">
          <span className="metric-chip">Min: ${minSal.toLocaleString()}</span>
          <span className="metric-chip">Avg: ${avgSal.toLocaleString()}</span>
          <span className="metric-chip">Max: ${maxSal.toLocaleString()}</span>
        </div>
        <div style={{ height: 120, marginTop: 8 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={salaryBuckets} margin={{ top: 4, right: 4, left: 0, bottom: 4 }}>
              <XAxis dataKey="label" tick={{ fontSize: 10 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 10 }} width={24} />
              <Tooltip formatter={(v) => [`${v} lineups`, '']} />
              <Bar dataKey="count">
                {salaryBuckets.map((_, i) => (
                  <Cell key={i} fill="#4f9de8" />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Value metrics */}
      {(minPitcherValue !== null || minBatterValue !== null) && (
        <div className="metrics-section">
          <h4>Portfolio Value</h4>
          <div className="metrics-row">
            {minPitcherValue !== null && (
              <span className="metric-chip">Min P value: {minPitcherValue.toFixed(2)}</span>
            )}
            {minBatterValue !== null && (
              <span className="metric-chip">Min batter value: {minBatterValue.toFixed(2)}</span>
            )}
          </div>
        </div>
      )}

      {/* Pitcher exposure */}
      {pitcherExposureList.length > 0 && (
        <div className="metrics-section">
          <h4>Pitcher Exposure</h4>
          <div className="exposure-list">
            {pitcherExposureList.map(({ name, team, count }) => (
              <div key={name} className="exposure-row">
                <span className="exposure-name">{name}</span>
                <TeamBadge team={team} className="exposure-team" />
                <span className="exposure-bar-wrap">
                  <span
                    className="exposure-bar"
                    style={{ width: `${(count / lineups.length) * 100}%` }}
                  />
                </span>
                <span className="exposure-count">{count}/{lineups.length}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Batter exposure */}
      {batterExposureList.length > 0 && (
        <div className="metrics-section">
          <h4>Batter Exposure</h4>
          <div className="exposure-list">
            {batterExposureList.map(({ name, team, count }) => (
              <div key={name} className="exposure-row">
                <span className="exposure-name">{name}</span>
                <TeamBadge team={team} className="exposure-team" />
                <span className="exposure-bar-wrap">
                  <span
                    className="exposure-bar"
                    style={{ width: `${(count / lineups.length) * 100}%` }}
                  />
                </span>
                <span className="exposure-count">{count}/{lineups.length}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
