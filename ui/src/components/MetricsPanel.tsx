import type { LineupResult, SSEEvent, OptimizeLineupEvent } from '../types'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { getStackNotation } from '../utils'

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
  const exposure: Record<string, { name: string; team: string; count: number }> = {}
  for (const lineup of lineups) {
    for (const p of lineup.players) {
      const key = String(p.player_id)
      if (!exposure[key]) {
        exposure[key] = { name: p.name, team: p.team, count: 0 }
      }
      exposure[key].count++
    }
  }
  const exposureList = Object.values(exposure).sort((a, b) => b.count - a.count)

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
    const stack = getStackNotation(lineup.players)
    stackCounts[stack] = (stackCounts[stack] ?? 0) + 1
  }
  const stackList = Object.entries(stackCounts).sort((a, b) => b[1] - a[1])

  return (
    <div className="metrics-panel">
      <h3>Portfolio Metrics</h3>

      {/* Timing */}
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

      {/* Player exposure */}
      <div className="metrics-section">
        <h4>Player Exposure</h4>
        <div className="exposure-list">
          {exposureList.map(({ name, team, count }) => (
            <div key={name} className="exposure-row">
              <span className="exposure-name">{name}</span>
              <span className="exposure-team">{team}</span>
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
    </div>
  )
}
