import { useCallback, useEffect, useRef, useState } from 'react'
import type { ExclusionScope, ExclusionsUpdate, GameStatus, PlayerExclusionStatus, PlayerExclusionsUpdate, PlatformType, SlateGamesResponse, SlatePlayersResponse, TwitterLineupRecord, TwitterNotification } from '../types'
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

function nextScope(scope: ExclusionScope): ExclusionScope {
  if (scope === 'none') return 'candidates'
  if (scope === 'candidates') return 'both'
  return 'none'
}

function scopeLabel(scope: ExclusionScope): string {
  if (scope === 'both') return 'Excl.'
  if (scope === 'candidates') return 'Cands'
  return 'Included'
}


export function SlatePanel({ disabled, projFetchExcluded = [], onProjFetchFilterChange, platform = 'draftkings', notifications = [], onDismissNotification, twitterLineups = [], onParseNotification, onDismissTwitterLineup }: Props) {
  const [slate, setSlate] = useState<SlateGamesResponse | null>(null)
  const [players, setPlayers] = useState<SlatePlayersResponse | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searchCands, setSearchCands] = useState('')
  const [searchCandsOpen, setSearchCandsOpen] = useState(false)
  const searchCandsRef = useRef<HTMLDivElement>(null)
  const [searchFull, setSearchFull] = useState('')
  const [searchFullOpen, setSearchFullOpen] = useState(false)
  const searchFullRef = useRef<HTMLDivElement>(null)
  const [ppdPcts, setPpdPcts] = useState<Record<string, string>>({})

  const syncPpdFromSlate = useCallback((data: SlateGamesResponse) => {
    const pcts: Record<string, string> = {}
    for (const g of data.games) {
      if (g.ppd_pct != null && g.ppd_pct > 0) {
        pcts[g.game] = String(g.ppd_pct)
      }
    }
    setPpdPcts(pcts)
  }, [])

  useEffect(() => {
    setSlate(null)
    setPlayers(null)
    setError(null)
    setPpdPcts({})
    fetchSlateGames()
      .then(data => {
        setSlate(data)
        syncPpdFromSlate(data)
        if (onProjFetchFilterChange) {
          // Only "both"-excluded games auto-sync to projFetchExcluded
          const slateExcluded = data.games
            .filter(g => g.exclusion_scope === 'both')
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

  // Auto-dismiss lineup notifications for teams not on the current slate
  useEffect(() => {
    if (!slate || !onDismissNotification) return
    const teams = new Set(slate.games.flatMap(g => [g.away, g.home]))
    for (const n of notifications) {
      if (n.lineup_team && !teams.has(n.lineup_team)) {
        onDismissNotification(n.id)
      }
    }
  }, [notifications, slate])

  // Close search dropdowns when clicking outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (searchCandsRef.current && !searchCandsRef.current.contains(e.target as Node))
        setSearchCandsOpen(false)
      if (searchFullRef.current && !searchFullRef.current.contains(e.target as Node))
        setSearchFullOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const buildUpdate = useCallback(
    (games: GameStatus[]): ExclusionsUpdate => {
      const game_scopes: Record<string, ExclusionScope> = {}
      const team_scopes: Record<string, ExclusionScope> = {}
      for (const g of games) {
        game_scopes[g.game] = g.exclusion_scope
        for (const t of g.teams) {
          team_scopes[t.team] = t.exclusion_scope
        }
      }
      const game_ppd_pcts: Record<string, number> = {}
      for (const [gameKey, val] of Object.entries(ppdPcts)) {
        const n = parseFloat(val)
        if (!isNaN(n) && n > 0) game_ppd_pcts[gameKey] = n
      }
      return { slate_id: slate?.slate_id ?? '', game_scopes, team_scopes, game_ppd_pcts }
    },
    [slate, ppdPcts]
  )

  const persist = useCallback(
    async (updated: SlateGamesResponse) => {
      setSaving(true)
      setError(null)
      try {
        const result = await saveSlateExclusions(buildUpdate(updated.games))
        setSlate(result)
        syncPpdFromSlate(result)
        // Refresh player list since team/game scopes affect the pool
        const updatedPlayers = await fetchSlatePlayers()
        setPlayers(updatedPlayers)
      } catch (e) {
        setError(String(e))
      } finally {
        setSaving(false)
      }
    },
    [buildUpdate, syncPpdFromSlate]
  )

  const persistPpd = useCallback(
    async () => {
      if (!slate || disabled || saving) return
      setSaving(true)
      setError(null)
      try {
        const result = await saveSlateExclusions(buildUpdate(slate.games))
        setSlate(result)
        syncPpdFromSlate(result)
      } catch (e) {
        setError(String(e))
      } finally {
        setSaving(false)
      }
    },
    [slate, disabled, saving, buildUpdate, syncPpdFromSlate]
  )

  const cycleGameScope = useCallback(
    (gameStr: string) => {
      if (!slate || disabled || saving) return
      const game = slate.games.find(g => g.game === gameStr)
      if (!game) return
      const newScope = nextScope(game.exclusion_scope)
      const updated: SlateGamesResponse = {
        ...slate,
        games: slate.games.map(g => {
          if (g.game !== gameStr) return g
          return {
            ...g,
            excluded: newScope !== 'none',
            exclusion_scope: newScope,
            teams: g.teams.map(t => ({ ...t, excluded: newScope !== 'none', exclusion_scope: newScope })),
          }
        }),
      }
      persist(updated)
      // Sync projFetchExcluded: only "both"-excluded games skip projection fetch
      if (onProjFetchFilterChange) {
        const gameKey = `${game.away}@${game.home}`
        if (newScope === 'none') {
          onProjFetchFilterChange(projFetchExcluded.filter(k => k !== gameKey))
        } else if (newScope === 'both') {
          onProjFetchFilterChange([...new Set([...projFetchExcluded, gameKey])])
        }
        // 'candidates' scope — keep projections fetching (no change to projFetchExcluded)
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

  const cycleTeamScope = useCallback(
    (gameStr: string, team: string) => {
      if (!slate || disabled || saving) return
      const updated: SlateGamesResponse = {
        ...slate,
        games: slate.games.map(g => {
          if (g.game !== gameStr) return g
          const updatedTeams = g.teams.map(t => {
            if (t.team !== team) return t
            const ns = nextScope(t.exclusion_scope)
            return { ...t, exclusion_scope: ns, excluded: ns !== 'none' }
          })
          // If all teams share the same non-none scope, promote to game level
          const allScopes = updatedTeams.map(t => t.exclusion_scope)
          const allSame = allScopes.every(s => s === allScopes[0]) && allScopes[0] !== 'none'
          const gameScope: ExclusionScope = allSame ? allScopes[0] as ExclusionScope : 'none'
          return { ...g, excluded: gameScope !== 'none', exclusion_scope: gameScope, teams: updatedTeams }
        }),
      }
      persist(updated)
    },
    [slate, disabled, saving, persist]
  )

  const buildPlayerScopes = useCallback(
    (playersList: PlayerExclusionStatus[]): Record<string, ExclusionScope> => {
      const scopes: Record<string, ExclusionScope> = {}
      for (const p of playersList) {
        if (p.individual_scope !== 'none') {
          scopes[String(p.player_id)] = p.individual_scope
        }
      }
      return scopes
    },
    []
  )

  const excludePlayerWithScope = useCallback(
    async (player: PlayerExclusionStatus, scope: 'candidates' | 'both') => {
      if (!players || disabled || saving) return
      const scopes = buildPlayerScopes(players.players)
      scopes[String(player.player_id)] = scope
      const update: PlayerExclusionsUpdate = { slate_id: players.slate_id, player_scopes: scopes }
      setSaving(true)
      setError(null)
      if (scope === 'candidates') { setSearchCands(''); setSearchCandsOpen(false) }
      else { setSearchFull(''); setSearchFullOpen(false) }
      try {
        const result = await savePlayerExclusions(update)
        setPlayers(result)
      } catch (e) {
        setError(String(e))
      } finally {
        setSaving(false)
      }
    },
    [players, disabled, saving, buildPlayerScopes]
  )

  const cyclePlayerScope = useCallback(
    async (playerId: number) => {
      if (!players || disabled || saving) return
      const player = players.players.find(p => p.player_id === playerId)
      if (!player) return
      const newScope = nextScope(player.exclusion_scope)
      const scopes = buildPlayerScopes(players.players)
      if (newScope === 'none') {
        delete scopes[String(playerId)]
      } else {
        scopes[String(playerId)] = newScope
      }
      const update: PlayerExclusionsUpdate = { slate_id: players.slate_id, player_scopes: scopes }
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
    [players, disabled, saving, buildPlayerScopes]
  )

  if (error) return <p className="slate-error">{error}</p>
  if (!slate) return <p className="slate-muted">Loading slate…</p>

  const slateTeams = new Set(slate.games.flatMap(g => [g.away, g.home]))

  if (slate.games.length === 0)
    return <p className="slate-muted">No slate loaded. Set the {platform === 'fanduel' ? 'FD' : 'DK'} Slate CSV path in Config.</p>

  const totalTeams = slate.games.length * 2
  const activeTeams = slate.games.reduce(
    (n, g) => n + g.teams.filter(t => t.exclusion_scope === 'none').length,
    0
  )
  const candidateExcludedTeams = slate.games.reduce(
    (n, g) => n + g.teams.filter(t => t.exclusion_scope === 'candidates').length,
    0
  )

  const searchCandsLower = searchCands.toLowerCase().trim()
  const searchFullLower = searchFull.toLowerCase().trim()

  // Display lists: only individually-excluded players (not team/game-implied)
  const candsExcludedList = players?.players.filter(p => p.individual_scope === 'candidates') ?? []
  const bothExcludedList  = players?.players.filter(p => p.individual_scope === 'both') ?? []

  // Candidates search pool: players with no effective exclusion (truly active)
  const activeForCandsSearch = players?.players.filter(p => p.exclusion_scope === 'none') ?? []
  // Full exclusion search pool: not yet individually fully excluded (allows upgrading team-implied cands)
  const activeForFullSearch = players?.players.filter(p => p.individual_scope !== 'both') ?? []

  const searchCandsResults = searchCandsLower.length > 0
    ? activeForCandsSearch.filter(p => p.name.toLowerCase().includes(searchCandsLower)).slice(0, 8)
    : []
  const searchFullResults = searchFullLower.length > 0
    ? activeForFullSearch.filter(p => p.name.toLowerCase().includes(searchFullLower)).slice(0, 8)
    : []

  return (
    <div className="slate-panel">
      <div className="slate-header">
        <h3 className="slate-title">Slate Games</h3>
        <span className="slate-summary">
          {activeTeams} of {totalTeams} teams active
          {candidateExcludedTeams > 0 && <span className="slate-cands-note"> · {candidateExcludedTeams} cands-excl</span>}
          {saving && <span className="slate-saving"> — saving…</span>}
        </span>
      </div>

      <div className="slate-games-grid">
        {slate.games.map(g => {
          const gameKey = `${g.away}@${g.home}`
          const fetchSkipped = projFetchExcluded.includes(gameKey)
          return (
          <div key={g.game} className={`game-card game-card--${g.exclusion_scope}${parseFloat(ppdPcts[g.game] ?? '0') > 0 ? ' game-card--ppd' : ''}`}>
            <div className="game-card-header">
              <div className="game-label-group">
                <span className="game-label">{g.away} @ {g.home}</span>
                {formatGameTime(g.game_start_time) && (
                  <span className="game-time">{formatGameTime(g.game_start_time)}</span>
                )}
              </div>
              <button
                className={`btn-game-toggle btn-game-toggle--${g.exclusion_scope}`}
                onClick={() => cycleGameScope(g.game)}
                disabled={disabled || saving}
                title={
                  g.exclusion_scope === 'none' ? 'Click to exclude from candidates only' :
                  g.exclusion_scope === 'candidates' ? 'Click to exclude from both candidates and field' :
                  'Click to re-include game'
                }
              >
                {scopeLabel(g.exclusion_scope)}
              </button>
            </div>
            {g.exclusion_scope !== 'both' && (
              <div className="game-card-secondary-actions">
                <button
                  className={`btn-proj-fetch${fetchSkipped ? ' btn-proj-fetch-off' : ' btn-proj-fetch-on'}`}
                  onClick={() => toggleProjFetch(gameKey)}
                  title={fetchSkipped ? 'Include this game in projection fetch' : 'Exclude this game from projection fetch (keeps existing projections for these players)'}
                >
                  {fetchSkipped ? '⊘ Proj' : '↓ Proj'}
                </button>
                <label className="ppd-label">
                  PPD
                  <input
                    className="ppd-input"
                    type="number"
                    min={0}
                    max={99}
                    step={1}
                    placeholder="0"
                    value={ppdPcts[g.game] ?? ''}
                    onChange={e => setPpdPcts(prev => ({ ...prev, [g.game]: e.target.value }))}
                    onBlur={() => persistPpd()}
                    onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }}
                    disabled={disabled || saving}
                  />
                  %
                </label>
              </div>
            )}
            <div className="game-teams">
              {g.teams.map(t => (
                <button
                  key={t.team}
                  className={`team-chip team-chip--${t.exclusion_scope}`}
                  onClick={() => cycleTeamScope(g.game, t.team)}
                  disabled={disabled || saving}
                  title={
                    t.exclusion_scope === 'none' ? `Click to exclude ${t.team} from candidates only` :
                    t.exclusion_scope === 'candidates' ? `Click to exclude ${t.team} from everything` :
                    `Click to re-include ${t.team}`
                  }
                >
                  {t.team}
                  {t.exclusion_scope !== 'none' && (
                    <span className={`team-chip-scope team-chip-scope--${t.exclusion_scope}`}>
                      {t.exclusion_scope === 'candidates' ? 'C' : 'X'}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        )})}

      </div>

      <div className="player-exclusions">
        <div className="player-exclusions-header">
          <h3 className="slate-title">Player Exclusions</h3>
          {(candsExcludedList.length + bothExcludedList.length) > 0 && (
            <span className="slate-summary">{candsExcludedList.length + bothExcludedList.length} excluded</span>
          )}
        </div>

        <div className="player-search-group">
          <div className="player-search-group-label">Exclude from candidates</div>
          <div className="player-search-wrap" ref={searchCandsRef}>
            <input
              className="player-search-input"
              type="text"
              placeholder="Search players to exclude from candidates…"
              value={searchCands}
              onChange={e => { setSearchCands(e.target.value); setSearchCandsOpen(true) }}
              onFocus={() => setSearchCandsOpen(true)}
              disabled={disabled || saving}
            />
            {searchCandsOpen && searchCandsResults.length > 0 && (
              <ul className="player-search-results">
                {searchCandsResults.map(p => (
                  <li key={p.player_id}>
                    <button
                      className="player-search-result-btn"
                      onMouseDown={e => { e.preventDefault(); excludePlayerWithScope(p, 'candidates') }}
                    >
                      <span className="psr-name">{p.name}</span>
                      <span className="psr-meta">{p.position} · {p.team} · ${p.salary.toLocaleString()}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {searchCandsOpen && searchCandsLower.length > 0 && searchCandsResults.length === 0 && (
              <div className="player-search-empty">No active players match</div>
            )}
          </div>
          {candsExcludedList.length > 0 && (
            <div className="excluded-players-list">
              {candsExcludedList.map(p => (
                <span key={p.player_id} className="excluded-player-chip excluded-player-chip--candidates">
                  <span className="epc-name">{p.name}</span>
                  <span className="epc-meta">{p.position} · {p.team}</span>
                  <button
                    className="epc-scope scope-candidates"
                    onClick={() => cyclePlayerScope(p.player_id)}
                    disabled={disabled || saving}
                    title="Excluded from candidates only — click to exclude from everything"
                  >
                    Cands
                  </button>
                  <button
                    className="epc-remove"
                    onClick={async () => {
                      if (!players || disabled || saving) return
                      const scopes = buildPlayerScopes(players.players)
                      delete scopes[String(p.player_id)]
                      const update: PlayerExclusionsUpdate = { slate_id: players.slate_id, player_scopes: scopes }
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
                    }}
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

        <div className="player-search-group">
          <div className="player-search-group-label">Exclude from candidates + field</div>
          <div className="player-search-wrap" ref={searchFullRef}>
            <input
              className="player-search-input"
              type="text"
              placeholder="Search players to exclude from everything…"
              value={searchFull}
              onChange={e => { setSearchFull(e.target.value); setSearchFullOpen(true) }}
              onFocus={() => setSearchFullOpen(true)}
              disabled={disabled || saving}
            />
            {searchFullOpen && searchFullResults.length > 0 && (
              <ul className="player-search-results">
                {searchFullResults.map(p => (
                  <li key={p.player_id}>
                    <button
                      className="player-search-result-btn"
                      onMouseDown={e => { e.preventDefault(); excludePlayerWithScope(p, 'both') }}
                    >
                      <span className="psr-name">{p.name}</span>
                      <span className="psr-meta">
                        {p.position} · {p.team} · ${p.salary.toLocaleString()}
                        {p.exclusion_scope === 'candidates' && (
                          <span className="psr-upgrade-hint"> · upgrade from cands</span>
                        )}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {searchFullOpen && searchFullLower.length > 0 && searchFullResults.length === 0 && (
              <div className="player-search-empty">No players match</div>
            )}
          </div>
          {bothExcludedList.length > 0 && (
            <div className="excluded-players-list">
              {bothExcludedList.map(p => (
                <span key={p.player_id} className="excluded-player-chip excluded-player-chip--both">
                  <span className="epc-name">{p.name}</span>
                  <span className="epc-meta">{p.position} · {p.team}</span>
                  <button
                    className="epc-scope scope-both"
                    onClick={() => cyclePlayerScope(p.player_id)}
                    disabled={disabled || saving}
                    title="Excluded from everything — click to re-include"
                  >
                    Excl.
                  </button>
                  <button
                    className="epc-remove"
                    onClick={async () => {
                      if (!players || disabled || saving) return
                      const scopes = buildPlayerScopes(players.players)
                      delete scopes[String(p.player_id)]
                      const update: PlayerExclusionsUpdate = { slate_id: players.slate_id, player_scopes: scopes }
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
                    }}
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
                      {n.could_be_lineup && (!n.lineup_team || slateTeams.has(n.lineup_team)) && (
                        <button
                          className="twitter-notif-parse"
                          onClick={() => onParseNotification?.(n)}
                          title="Parse lineup from this notification"
                        >
                          Parse
                        </button>
                      )}
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
