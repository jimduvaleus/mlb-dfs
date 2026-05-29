import type { LineupResult, SSEEvent, OptimizeLineupEvent, GppGenerateDoneEvent } from '../types'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
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

  // --- Candidate pool team distribution ---
  const generateDoneEvent = [...events].reverse().find(e => e.stage === 'gpp_generate_done') as GppGenerateDoneEvent | undefined
  const teamDist = generateDoneEvent?.team_distribution
  const teamDistList = teamDist
    ? Object.entries(teamDist).sort((a, b) => b[1] - a[1])
    : null
  const teamDistTotal = teamDistList ? teamDistList.reduce((s, [, c]) => s + c, 0) : 0
  const teamDistExpected = teamDistList && teamDistList.length > 0
    ? teamDistTotal / teamDistList.length
    : 0
  const teamDistMax = teamDistList ? Math.max(...teamDistList.map(([, c]) => c)) : 1

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

      {/* Candidate pool team distribution */}
      {teamDistList && teamDistList.length > 0 && (
        <div className="metrics-section">
          <h4>Candidate Pool — Primary Stack Distribution</h4>
          <div className="metrics-row" style={{ marginBottom: 6 }}>
            <span className="metric-chip">
              {teamDistList.length} teams · {teamDistTotal.toLocaleString()} main-stack lineups · expected {Math.round(teamDistExpected)}/team
            </span>
          </div>
          <div className="stack-list">
            {teamDistList.map(([team, count]) => {
              const pct = (count / teamDistMax) * 100
              const expectedPct = (teamDistExpected / teamDistMax) * 100
              const deviation = teamDistExpected > 0
                ? (count - teamDistExpected) / teamDistExpected
                : 0
              const outside = Math.abs(deviation) > 0.33
              const barColor = outside ? (deviation < 0 ? '#e07b54' : '#4fb36c') : '#4f9de8'
              const deviationLabel = `${deviation >= 0 ? '+' : ''}${(deviation * 100).toFixed(0)}%`
              return (
                <div key={team} className="stack-row">
                  <span style={{ width: 28, display: 'inline-flex', alignItems: 'center' }}>
                    <TeamBadge team={team} className="exposure-team" />
                  </span>
                  <span style={{ position: 'relative', flex: 1, height: 14, margin: '0 6px' }}>
                    <span
                      className="stack-bar"
                      style={{ width: `${pct}%`, background: barColor, position: 'absolute', top: 0, left: 0, height: '100%' }}
                    />
                    {/* tick mark at uniform expected */}
                    <span style={{
                      position: 'absolute', top: 0, left: `${expectedPct}%`,
                      width: 2, height: '100%', background: 'rgba(255,255,255,0.45)',
                      transform: 'translateX(-50%)',
                    }} />
                  </span>
                  <span className="stack-count" style={{ minWidth: 38, textAlign: 'right' }}>{count.toLocaleString()}</span>
                  <span className="muted" style={{ minWidth: 38, textAlign: 'right', fontSize: '0.82em' }}>{deviationLabel}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

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
