import type { ProjectionPlayerRow, PlatformType } from '../types'
import TeamBadge from './TeamBadge'

interface Props {
  players: ProjectionPlayerRow[]
  platform?: PlatformType
}

export function ProjectionsTable({ players }: Props) {
  if (players.length === 0) {
    return (
      <div className="projections-table-wrap">
        <p className="muted">No projections available. Fetch projections on the Config tab.</p>
      </div>
    )
  }

  const byTeam = new Map<string, ProjectionPlayerRow[]>()
  for (const p of players) {
    if (!byTeam.has(p.team)) byTeam.set(p.team, [])
    byTeam.get(p.team)!.push(p)
  }
  const teams = [...byTeam.keys()].sort()

  return (
    <div className="projections-table-wrap">
      <h3>Projections — {players.length} players, {teams.length} teams</h3>
      <div className="portfolio-cards">
        {teams.map(team => {
          const teamPlayers = byTeam.get(team)!
          const pitchers = teamPlayers.filter(p => p.position === 'P')
          const batters  = teamPlayers
            .filter(p => p.position !== 'P')
            .sort((a, b) => (a.slot ?? 99) - (b.slot ?? 99))
          const hitterProj = batters.reduce((sum, p) => sum + p.mean, 0)

          return (
            <div key={team} className="lineup-card">
              <div className="lineup-card-header">
                <TeamBadge team={team} />
                <div className="lineup-card-header-right">
                  <span className="projections-team-total">{hitterProj.toFixed(1)} pts</span>
                </div>
              </div>
              <div className="lineup-card-players projections-card-players">
                {[...pitchers, ...batters].map((p, i) => {
                  const isPitcher = p.position === 'P'
                  const slotNum = !isPitcher && p.slot != null && p.slot >= 1 && p.slot <= 9 ? p.slot : null
                  return (
                    <div key={i} className="lineup-player">
                      <span className="lineup-player-pos">{p.position}</span>
                      <span className="lineup-player-name">
                        {p.name}
                        {!isPitcher && (
                          p.slot_confirmed
                            ? <span className="batting-slot-bubble batting-slot-bubble--confirmed" title="Confirmed lineup slot">{slotNum ?? '?'}</span>
                            : <span className="batting-slot-bubble batting-slot-bubble--projected" title="Projected lineup slot">{slotNum ?? '?'}</span>
                        )}
                      </span>
                      <TeamBadge team={p.team} className="lineup-player-team" />
                      <span className="lineup-player-sal">${(p.salary / 1000).toFixed(1)}k</span>
                      <span className="lineup-player-proj">{p.mean.toFixed(1)}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
