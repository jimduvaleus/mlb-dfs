import { useCallback, useEffect, useRef, useState } from 'react'
import type { ExclusionsUpdate, GameStatus, PlayerExclusionStatus, PlayerExclusionsUpdate, PlatformType, SlateGamesResponse, SlatePlayersResponse, TwitterLineupRecord, TwitterNotification } from '../types'
import { fetchSlateGames, fetchSlatePlayers, savePlayerExclusions, saveSlateExclusions } from '../api'

interface Props {
  disabled?: boolean
  projFetchExcluded?: string[]
  onProjFetchFilterChange?: (excluded: string[]) => void
  platform?: PlatformType
  notifications?: TwitterNotification[]
  onDismissNotification?: (id: string) => void
  twitterLineups?: TwitterLineupRecord[]
  onParseNotification?: (notif: TwitterNotification) => void
  onDismissTwitterLineup?: (team: string) => void
}

function formatGameTime(iso: string | null | undefined): string {
  if (!iso) return ''
  const timePart = iso.substring(11, 16)
  const [hStr, mStr] = timePart.split(':')
  const h = parseInt(hStr, 10)
  if (isNaN(h)) return ''
  const period = h >= 12 ? 'PM' : 'AM'
  const h12 = h % 12 || 12
  return `${h12}:${mStr} ${period} ET`
}

export function SlatePanel({ disabled, projFetchExcluded = [], onProjFetchFilterChange, platform = 'draftkings', notifications = [], onDismissNotification, twitterLineups = [], onParseNotification, onDismissTwitterLineup }: Props) {
  const [slate, setSlate] = useState<SlateGamesResponse | null>(null)
  const [players, setPlayers] = useState<SlatePlayersResponse | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const searchRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setSlate(null)
    setPlayers(null)
    setError(null)
    fetchSlateGames()
      .then(data => {
        setSlate(data)
        if (onProjFetchFilterChange) {
          const slateExcluded = data.games
            .filter(g => g.excluded)
            .map(g => `${g.away}@${g.home}`)
          const merged = [...new Set([...projFetchExcluded, ...slateExcluded])]
          if (merged.length !== projFetchExcluded.length) {
            onProjFetchFilterChange(merged)
          }
        }
      })
      .catch(e => setError(String(e)))
    fetchSlatePlayers()
      .then(setPlayers)
      .catch(e => setError(String(e)))
  }, [platform])

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
      const toggledGame = slate.games.find(g => g.game === gameStr)
      const nowExcluded = toggledGame ? !toggledGame.excluded : false
      const updated: SlateGamesResponse = {
        ...slate,
        games: slate.games.map(g => {
          if (g.game !== gameStr) return g
          return {
            ...g,
            excluded: nowExcluded,
            teams: g.teams.map(t => ({ ...t, excluded: nowExcluded })),
          }
        }),
      }
      persist(updated)
      if (onProjFetchFilterChange && toggledGame) {
        const gameKey = `${toggledGame.away}@${toggledGame.home}`
        onProjFetchFilterChange(
          nowExcluded
            ? [...new Set([...projFetchExcluded, gameKey])]
            : projFetchExcluded.filter(k => k !== gameKey)
        )
      }
    },
    [slate, disabled, saving, persist, projFetchExcluded, onProjFetchFilterChange]
  )

  const toggleProjFetch = useCallback(
    (gameKey: string) => {
      if (!onProjFetchFilterChange) return
      const isExcluded = projFetchExcluded.includes(gameKey)
      onProjFetchFilterChange(
        isExcluded
          ? projFetchExcluded.filter(g => g !== gameKey)
          : [...projFetchExcluded, gameKey]
      )
    },
    [projFetchExcluded, onProjFetchFilterChange]
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
    return <p className="slate-muted">No slate loaded. Set the {platform === 'fanduel' ? 'FD' : 'DK'} Slate CSV path in Config.</p>

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
        {slate.games.map(g => {
          const gameKey = `${g.away}@${g.home}`
          const fetchSkipped = projFetchExcluded.includes(gameKey)
          return (
          <div key={g.game} className={`game-card${g.excluded ? ' game-excluded' : ''}`}>
            <div className="game-card-header">
              <div className="game-label-group">
                <span className="game-label">{g.away} @ {g.home}</span>
                {formatGameTime(g.game_start_time) && (
                  <span className="game-time">{formatGameTime(g.game_start_time)}</span>
                )}
              </div>
              <div className="game-card-actions">
                {!g.excluded && (
                  <button
                    className={`btn-proj-fetch${fetchSkipped ? ' btn-proj-fetch-off' : ' btn-proj-fetch-on'}`}
                    onClick={() => toggleProjFetch(gameKey)}
                    title={fetchSkipped ? 'Include this game in projection fetch' : 'Exclude this game from projection fetch (keeps existing projections for these players)'}
                  >
                    {fetchSkipped ? '⊘ Proj' : '↓ Proj'}
                  </button>
                )}
                <button
                  className={`btn-game-toggle${g.excluded ? ' btn-game-excluded' : ' btn-game-included'}`}
                  onClick={() => toggleGame(g.game)}
                  disabled={disabled || saving}
                  title={g.excluded ? 'Re-include entire game' : 'Exclude entire game'}
                >
                  {g.excluded ? 'Excluded' : 'Included'}
                </button>
              </div>
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
        )})}

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

      {(notifications.length > 0 || twitterLineups.length > 0) && (
        <div className="twitter-columns">
          {notifications.length > 0 && (
            <div className="twitter-notifications">
              <div className="twitter-notifications-header">
                <h3 className="slate-title">X Notifications</h3>
                <span className="slate-summary">{notifications.length} unread</span>
              </div>
              <div className="twitter-notifications-list">
                {notifications.map(n => (
                  <div key={n.id} className="twitter-notif-item">
                    <div className="twitter-notif-body">
                      <div className="twitter-notif-header">
                        <span className="twitter-notif-summary">{n.summary}</span>
                        <span className="twitter-notif-time">{new Date(n.captured_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      </div>
                      {n.body && <span className="twitter-notif-text">{n.body}</span>}
                    </div>
                    <div className="twitter-notif-actions">
                      <button
                        className="twitter-notif-parse"
                        onClick={() => onParseNotification?.(n)}
                        title="Parse lineup from this notification"
                      >
                        Parse
                      </button>
                      <button
                        className="twitter-notif-dismiss"
                        onClick={() => onDismissNotification?.(n.id)}
                        title="Dismiss"
                        aria-label="Dismiss notification"
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {twitterLineups.length > 0 && (
            <div className="twitter-confirmed-lineups">
              <div className="twitter-notifications-header">
                <h3 className="slate-title">Confirmed Lineups</h3>
                <span className="slate-summary">{twitterLineups.length} team{twitterLineups.length !== 1 ? 's' : ''}</span>
              </div>
              <div className="twitter-confirmed-list">
                {twitterLineups.map(tl => (
                  <div key={tl.team} className="twitter-confirmed-item">
                    <div className="twitter-confirmed-header">
                      <span className="twitter-confirmed-team">{tl.team}</span>
                      <span className="twitter-confirmed-time">
                        {new Date(tl.confirmed_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                      <button
                        className="twitter-notif-dismiss"
                        onClick={() => onDismissTwitterLineup?.(tl.team)}
                        title={`Remove confirmed lineup for ${tl.team}`}
                        aria-label={`Dismiss ${tl.team} lineup`}
                      >
                        ✕
                      </button>
                    </div>
                    <div className="twitter-confirmed-slots">
                      {tl.slots.map(s => (
                        <span key={s.slot} className="twitter-confirmed-slot">
                          <span className="batting-slot-bubble batting-slot-bubble--confirmed">{s.slot}</span>
                          {s.name}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
