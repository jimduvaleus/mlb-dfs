import { useCallback, useEffect, useState } from 'react'
import type { ExclusionsUpdate, GameStatus, SlateGamesResponse } from '../types'
import { fetchSlateGames, saveSlateExclusions } from '../api'

interface Props {
  disabled?: boolean
}

export function SlatePanel({ disabled }: Props) {
  const [slate, setSlate] = useState<SlateGamesResponse | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchSlateGames()
      .then(setSlate)
      .catch(e => setError(String(e)))
  }, [])

  const buildUpdate = useCallback(
    (games: GameStatus[]): ExclusionsUpdate => {
      const excluded_games = games.filter(g => g.excluded).map(g => g.game)
      // Only include teams that are individually excluded (game not fully excluded)
      const excluded_teams = games
        .filter(g => !g.excluded)
        .flatMap(g => g.teams.filter(t => t.excluded).map(t => t.team))
      return { slate_id: slate?.slate_id ?? '', excluded_games, excluded_teams }
    },
    [slate]
  )

  const persist = useCallback(
    async (updated: SlateGamesResponse) => {
      setSaving(true)
      setError(null)
      try {
        const result = await saveSlateExclusions(buildUpdate(updated.games))
        setSlate(result)
      } catch (e) {
        setError(String(e))
      } finally {
        setSaving(false)
      }
    },
    [buildUpdate]
  )

  const toggleGame = useCallback(
    (gameStr: string) => {
      if (!slate || disabled || saving) return
      const updated: SlateGamesResponse = {
        ...slate,
        games: slate.games.map(g => {
          if (g.game !== gameStr) return g
          const nowExcluded = !g.excluded
          return {
            ...g,
            excluded: nowExcluded,
            teams: g.teams.map(t => ({ ...t, excluded: nowExcluded })),
          }
        }),
      }
      persist(updated)
    },
    [slate, disabled, saving, persist]
  )

  const toggleTeam = useCallback(
    (gameStr: string, team: string) => {
      if (!slate || disabled || saving) return
      const updated: SlateGamesResponse = {
        ...slate,
        games: slate.games.map(g => {
          if (g.game !== gameStr) return g
          const updatedTeams = g.teams.map(t =>
            t.team === team ? { ...t, excluded: !t.excluded } : t
          )
          const allExcluded = updatedTeams.every(t => t.excluded)
          return { ...g, excluded: allExcluded, teams: updatedTeams }
        }),
      }
      persist(updated)
    },
    [slate, disabled, saving, persist]
  )

  if (error) return <p className="slate-error">{error}</p>
  if (!slate) return <p className="slate-muted">Loading slate…</p>
  if (slate.games.length === 0)
    return <p className="slate-muted">No slate loaded. Set the DK Slate path in Config.</p>

  const totalTeams = slate.games.length * 2
  const activeTeams = slate.games.reduce(
    (n, g) => n + g.teams.filter(t => !t.excluded).length,
    0
  )

  return (
    <div className="slate-panel">
      <div className="slate-header">
        <h3 className="slate-title">Slate Games</h3>
        <span className="slate-summary">
          {activeTeams} of {totalTeams} teams active
          {saving && <span className="slate-saving"> — saving…</span>}
        </span>
      </div>

      <div className="slate-games-grid">
        {slate.games.map(g => (
          <div key={g.game} className={`game-card${g.excluded ? ' game-excluded' : ''}`}>
            <div className="game-card-header">
              <span className="game-label">
                {g.away} @ {g.home}
              </span>
              <button
                className={`btn-game-toggle${g.excluded ? ' btn-game-excluded' : ' btn-game-included'}`}
                onClick={() => toggleGame(g.game)}
                disabled={disabled || saving}
                title={g.excluded ? 'Re-include entire game' : 'Exclude entire game'}
              >
                {g.excluded ? 'Excluded' : 'Included'}
              </button>
            </div>
            <div className="game-teams">
              {g.teams.map(t => (
                <button
                  key={t.team}
                  className={`team-chip${t.excluded ? ' team-excluded' : ' team-active'}`}
                  onClick={() => toggleTeam(g.game, t.team)}
                  disabled={disabled || saving}
                  title={t.excluded ? `Re-include ${t.team}` : `Exclude ${t.team}`}
                >
                  {t.team}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
