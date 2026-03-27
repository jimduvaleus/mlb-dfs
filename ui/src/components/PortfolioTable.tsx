import type { LineupResult, PlayerRow } from '../types'
import { getStackNotation } from '../utils'

interface Props {
  lineups: LineupResult[]
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
      <div className="portfolio-cards">
        {lineups.map(lineup => {
          const sorted = sortPlayersByPosition(lineup.players)
          const stack = getStackNotation(lineup.players)
          return (
            <div key={lineup.lineup_index} className="lineup-card">
              <div className="lineup-card-header">
                <span className="lineup-card-num">#{lineup.lineup_index}</span>
                <span className="lineup-card-salary">${lineup.lineup_salary.toLocaleString()}</span>
                {stack && <span className="lineup-card-stack">{stack}</span>}
              </div>
              <div className="lineup-card-players">
                {sorted.map((p, i) => (
                  <div key={i} className="lineup-player">
                    <span className="lineup-player-pos">{p.position}</span>
                    <span className="lineup-player-name">{p.name}</span>
                    <span className="lineup-player-team">{p.team}</span>
                    <span className="lineup-player-sal">${(p.salary / 1000).toFixed(1)}k</span>
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
