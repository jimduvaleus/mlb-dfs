import type { LineupResult, PlayerRow } from '../types'

const POSITION_ORDER = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF'] as const

interface Props {
  lineups: LineupResult[]
}

function getStackNotation(players: PlayerRow[]): string {
  const hitters = players.filter(p => p.position !== 'P')
  const counts: Record<string, number> = {}
  for (const p of hitters) {
    counts[p.team] = (counts[p.team] ?? 0) + 1
  }
  return Object.values(counts)
    .sort((a, b) => b - a)
    .join('-')
}

function sortPlayersByPosition(players: PlayerRow[]): PlayerRow[] {
  const pitchers = players.filter(p => p.position === 'P')
  const posOrder = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
  const batters = players.filter(p => p.position !== 'P')

  const filled: PlayerRow[] = [...pitchers]
  const remaining = [...batters]
  for (const pos of posOrder) {
    const idx = remaining.findIndex(p => p.position === pos)
    if (idx >= 0) {
      filled.push(remaining.splice(idx, 1)[0])
    }
  }
  filled.push(...remaining)
  return filled
}

export function PortfolioTable({ lineups }: Props) {
  if (lineups.length === 0) return null

  return (
    <div className="portfolio-table-wrap">
      <h3>Portfolio — {lineups.length} Lineups</h3>
      <div className="portfolio-table-scroll">
        <table className="portfolio-table">
          <thead>
            <tr>
              <th>#</th>
              <th>P(hit)</th>
              <th>Salary</th>
              <th>Stack</th>
              {POSITION_ORDER.map((pos, i) => (
                <th key={i}>{pos}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {lineups.map(lineup => {
              const sorted = sortPlayersByPosition(lineup.players)
              const stack = getStackNotation(lineup.players)
              return (
                <tr key={lineup.lineup_index}>
                  <td className="lineup-num">{lineup.lineup_index}</td>
                  <td className="p-hit">{(lineup.p_hit_target * 100).toFixed(1)}%</td>
                  <td className="salary">${lineup.lineup_salary.toLocaleString()}</td>
                  <td className="stack">{stack}</td>
                  {sorted.map((p, i) => (
                    <td key={i} className="player-cell">
                      <span className="player-name">{p.name}</span>
                      <span className="player-team">{p.team}</span>
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
