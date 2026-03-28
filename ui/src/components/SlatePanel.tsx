import { useCallback, useEffect, useRef, useState } from 'react'
import type { ExclusionsUpdate, GameStatus, PlayerExclusionStatus, PlayerExclusionsUpdate, SlateGamesResponse, SlatePlayersResponse } from '../types'
import { fetchSlateGames, fetchSlatePlayers, savePlayerExclusions, saveSlateExclusions } from '../api'

interface Props {
  disabled?: boolean
}

export function SlatePanel({ disabled }: Props) {
  const [slate, setSlate] = useState<SlateGamesResponse | null>(null)
  const [players, setPlayers] = useState<SlatePlayersResponse | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const searchRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchSlateGames()
      .then(data => { setSlate(data) })
      .catch(e => setError(String(e)))
    fetchSlatePlayers()
      .then(setPlayers)
      .catch(e => setError(String(e)))
  }, [])

  // Close search dropdown when clicking outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setSearchOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
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
        // Refresh player list since team/game exclusions affect the pool
        const updatedPlayers = await fetchSlatePlayers()
        setPlayers(updatedPlayers)
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

  const excludePlayer = useCallback(
    async (player: PlayerExclusionStatus) => {
      if (!players || disabled || saving) return
      const newIds = [...players.players.filter(p => p.excluded).map(p => p.player_id), player.player_id]
      const update: PlayerExclusionsUpdate = { slate_id: players.slate_id, excluded_player_ids: newIds }
      setSaving(true)
      setError(null)
      setSearch('')
      setSearchOpen(false)
      try {
        const result = await savePlayerExclusions(update)
        setPlayers(result)
      } catch (e) {
        setError(String(e))
      } finally {
        setSaving(false)
      }
    },
    [players, disabled, saving]
  )

  const includePlayer = useCallback(
    async (playerId: number) => {
      if (!players || disabled || saving) return
      const newIds = players.players.filter(p => p.excluded && p.player_id !== playerId).map(p => p.player_id)
      const update: PlayerExclusionsUpdate = { slate_id: players.slate_id, excluded_player_ids: newIds }
      setSaving(true)
      setError(null)
      try {
        const result = await savePlayerExclusions(update)
        setPlayers(result)
      } catch (e) {
        setError(String(e))
      } finally {
        setSaving(false)
      }
    },
    [players, disabled, saving]
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

  const searchLower = search.toLowerCase().trim()
  const activePlayers = players?.players.filter(p => !p.excluded) ?? []
  const excludedPlayers = players?.players.filter(p => p.excluded) ?? []
  const searchResults = searchLower.length > 0
    ? activePlayers.filter(p => p.name.toLowerCase().includes(searchLower)).slice(0, 8)
    : []

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

      <div className="player-exclusions">
        <div className="player-exclusions-header">
          <h3 className="slate-title">Player Exclusions</h3>
          {excludedPlayers.length > 0 && (
            <span className="slate-summary">{excludedPlayers.length} excluded</span>
          )}
        </div>

        <div className="player-search-wrap" ref={searchRef}>
          <input
            className="player-search-input"
            type="text"
            placeholder="Search players to exclude…"
            value={search}
            onChange={e => { setSearch(e.target.value); setSearchOpen(true) }}
            onFocus={() => setSearchOpen(true)}
            disabled={disabled || saving}
          />
          {searchOpen && searchResults.length > 0 && (
            <ul className="player-search-results">
              {searchResults.map(p => (
                <li key={p.player_id}>
                  <button
                    className="player-search-result-btn"
                    onMouseDown={e => { e.preventDefault(); excludePlayer(p) }}
                  >
                    <span className="psr-name">{p.name}</span>
                    <span className="psr-meta">{p.position} · {p.team} · ${p.salary.toLocaleString()}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {searchOpen && searchLower.length > 0 && searchResults.length === 0 && (
            <div className="player-search-empty">No active players match</div>
          )}
        </div>

        {excludedPlayers.length > 0 && (
          <div className="excluded-players-list">
            {excludedPlayers.map(p => (
              <span key={p.player_id} className="excluded-player-chip">
                <span className="epc-name">{p.name}</span>
                <span className="epc-meta">{p.position} · {p.team}</span>
                <button
                  className="epc-remove"
                  onClick={() => includePlayer(p.player_id)}
                  disabled={disabled || saving}
                  title={`Re-include ${p.name}`}
                >
                  ✕
                </button>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
