import type { LineupResult, PlayerRow } from '../types'
import { getStackNotation } from '../utils'
import TeamBadge from './TeamBadge'

interface Props {
  lineups: LineupResult[]
  unconfirmedPlayerIds?: number[]
  onDeleteLineup?: (lineupIndex: number) => void
  replacingLineupIndex?: number | null
}

function assignedPos(p: PlayerRow): string {
  return p.assigned_position ?? p.position.split('/')[0]
}

function sortPlayersByPosition(players: PlayerRow[]): PlayerRow[] {
  const pitchers = players.filter(p => assignedPos(p) === 'P')
  const posOrder = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
  const batters = players.filter(p => assignedPos(p) !== 'P')

  const filled: PlayerRow[] = [...pitchers]
  const remaining = [...batters]
  for (const pos of posOrder) {
    const idx = remaining.findIndex(p => assignedPos(p) === pos)
    if (idx >= 0) {
      filled.push(remaining.splice(idx, 1)[0])
    }
  }
  filled.push(...remaining)
  return filled
}

export function PortfolioTable({ lineups, unconfirmedPlayerIds, onDeleteLineup, replacingLineupIndex }: Props) {
  if (lineups.length === 0) return null
  const unconfirmedSet = new Set(unconfirmedPlayerIds ?? [])

  const unconfirmedByPlayer = new Map<number, { name: string; count: number }>()
  for (const lineup of lineups) {
    for (const p of lineup.players) {
      if (unconfirmedSet.has(p.player_id)) {
        const entry = unconfirmedByPlayer.get(p.player_id)
        if (entry) {
          entry.count++
        } else {
          unconfirmedByPlayer.set(p.player_id, { name: p.name, count: 1 })
        }
      }
    }
  }

  const totalUnconfirmed = Array.from(unconfirmedByPlayer.values()).reduce((sum, e) => sum + e.count, 0)

  const sortedUnconfirmedPlayers = Array.from(unconfirmedByPlayer.values()).sort((a, b) => b.count - a.count)
  const breakdown = unconfirmedByPlayer.size <= 5
    ? ' — ' + sortedUnconfirmedPlayers.map(e => `${e.count} ${e.name}`).join(', ')
    : ''

  return (
    <div className="portfolio-table-wrap">
      <h3>Portfolio — {lineups.length} Lineups</h3>
      <div className={`portfolio-unconfirmed-banner ${totalUnconfirmed === 0 ? 'portfolio-unconfirmed-banner--clear' : ''}`}>
        {totalUnconfirmed === 0
          ? '✓ All lineup slots confirmed'
          : `✕ ${totalUnconfirmed} unconfirmed lineup slot${totalUnconfirmed !== 1 ? 's' : ''} across portfolio${breakdown}`}
      </div>
      <div className="portfolio-cards">
        {lineups.map(lineup => {
          const sorted = sortPlayersByPosition(lineup.players)
          const stack = getStackNotation(lineup.players)
          const isReplacing = replacingLineupIndex === lineup.lineup_index
          return (
            <div key={lineup.lineup_index} className="lineup-card">
              <div className="lineup-card-header">
                <span className="lineup-card-num">#{lineup.lineup_index}</span>
                <span className="lineup-card-salary">${lineup.lineup_salary.toLocaleString()}</span>
                <div className="lineup-card-header-right">
                  {isReplacing ? (
                    <span className="lineup-card-generating">Generating…</span>
                  ) : (
                    <>
                      {stack && <span className="lineup-card-stack">{stack}</span>}
                      {onDeleteLineup && (
                        <button
                          className="lineup-card-delete"
                          onClick={() => onDeleteLineup(lineup.lineup_index)}
                          title="Delete and replace this lineup"
                          disabled={replacingLineupIndex != null}
                        >
                          🗑
                        </button>
                      )}
                    </>
                  )}
                </div>
              </div>
              {lineup.upload_tag && (
                <div className="lineup-card-entry-info">
                  {lineup.upload_tag}{lineup.entry_fee ? ` ${lineup.entry_fee}` : ''}{lineup.contest_name ? ` ${lineup.contest_name}` : ''}
                </div>
              )}
              <div className="lineup-card-players">
                {sorted.map((p, i) => (
                  <div key={i} className="lineup-player">
                    <span className="lineup-player-pos">{assignedPos(p)}</span>
                    <span className="lineup-player-name">
                      {p.name}
                      {assignedPos(p) !== 'P' && (() => {
                        const slotNum = p.slot != null && p.slot >= 1 && p.slot <= 9 ? p.slot : null
                        if (p.slot_confirmed) {
                          return <span className="batting-slot-bubble batting-slot-bubble--confirmed" title="Confirmed lineup slot">{slotNum ?? '?'}</span>
                        }
                        return <span className="batting-slot-bubble batting-slot-bubble--projected" title="Projected lineup slot">{slotNum ?? '?'}</span>
                      })()}
                    </span>
                    <TeamBadge team={p.team} className="lineup-player-team" />
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
