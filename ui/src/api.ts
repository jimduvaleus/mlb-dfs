import type { AppConfig, CacheStatus, ExclusionsUpdate, OwnershipSyncResult, PlayerExclusionsUpdate, PlayerProjectionOverridesResponse, PlayerProjectionOverridesUpdate, ProjectionPlayerRow, ProjectionsStatus, LineupResult, SlateGamesResponse, SlateListResponse, SlatePlayersResponse, TeamOwnershipReductionsResponse, TeamOwnershipReductionsUpdate, TwitterLineupParseResponse, TwitterLineupRecord, TwitterLineupSaveRequest, TwitterNotification } from './types'

export async function fetchConfig(): Promise<AppConfig> {
  const res = await fetch('/api/config')
  if (!res.ok) throw new Error(`Failed to load config: ${res.statusText}`)
  return res.json()
}

export async function saveConfig(cfg: AppConfig): Promise<AppConfig> {
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Failed to save config: ${detail}`)
  }
  return res.json()
}

export async function fetchProjectionsStatus(): Promise<ProjectionsStatus> {
  const res = await fetch('/api/projections/status')
  if (!res.ok) throw new Error(`Failed to get projections status: ${res.statusText}`)
  return res.json()
}

export async function fetchMergeInfoState(): Promise<Record<string, unknown>> {
  const res = await fetch('/api/projections/merge_info')
  if (!res.ok) return {}
  return res.json()
}

export async function fetchUnconfirmedPlayerIds(): Promise<number[]> {
  const res = await fetch('/api/projections/unconfirmed')
  if (!res.ok) return []
  const data = await res.json()
  return data.player_ids ?? []
}

export async function fetchProjectionPlayers(): Promise<ProjectionPlayerRow[]> {
  const res = await fetch('/api/projections/players')
  if (!res.ok) return []
  return res.json()
}

export async function fetchOwnershipSync(): Promise<OwnershipSyncResult> {
  const res = await fetch('/api/projections/ownership_sync')
  if (!res.ok) return { status: 'error', reason: res.statusText }
  return res.json()
}

export async function fetchTeamTotals(): Promise<Record<string, number>> {
  const res = await fetch('/api/projections/team_totals')
  if (!res.ok) return {}
  return res.json()
}

export async function fetchProjectionSlates(): Promise<SlateListResponse> {
  const res = await fetch('/api/projections/slates')
  if (!res.ok) throw new Error(`Failed to load projection slates: ${res.statusText}`)
  return res.json()
}

export async function fetchPortfolio(platform?: string): Promise<LineupResult[]> {
  const url = platform ? `/api/portfolio?platform=${encodeURIComponent(platform)}` : '/api/portfolio'
  const res = await fetch(url)
  if (res.status === 404) return []
  if (!res.ok) throw new Error(`Failed to load portfolio: ${res.statusText}`)
  const data = await res.json()
  return data
}

export async function fetchRunStatus(): Promise<{ status: string; error: string | null }> {
  const res = await fetch('/api/run/status')
  if (!res.ok) throw new Error(`Failed to get run status`)
  return res.json()
}

export async function stopRun(): Promise<void> {
  const res = await fetch('/api/run/stop', { method: 'POST' })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Failed to stop run: ${detail}`)
  }
}

export async function writeUploadFiles(): Promise<{ paths: string[] }> {
  const res = await fetch('/api/run/write_upload', { method: 'POST' })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Failed to write upload files: ${detail}`)
  }
  return res.json()
}

export async function replaceLineup(lineupIndex: number): Promise<LineupResult[]> {
  const res = await fetch(`/api/portfolio/replace/${lineupIndex}`, { method: 'POST' })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Failed to replace lineup: ${detail}`)
  }
  return res.json()
}

export async function fetchSlateGames(): Promise<SlateGamesResponse> {
  const res = await fetch('/api/slate/games')
  if (!res.ok) throw new Error(`Failed to load slate games: ${res.statusText}`)
  return res.json()
}

export async function saveSlateExclusions(update: ExclusionsUpdate): Promise<SlateGamesResponse> {
  const res = await fetch('/api/slate/exclusions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(update),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Failed to save exclusions: ${detail}`)
  }
  return res.json()
}

export async function fetchSlatePlayers(): Promise<SlatePlayersResponse> {
  const res = await fetch('/api/slate/players')
  if (!res.ok) throw new Error(`Failed to load slate players: ${res.statusText}`)
  return res.json()
}

export async function savePlayerExclusions(update: PlayerExclusionsUpdate): Promise<SlatePlayersResponse> {
  const res = await fetch('/api/slate/player-exclusions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(update),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Failed to save player exclusions: ${detail}`)
  }
  return res.json()
}

export async function fetchNotifications(): Promise<TwitterNotification[]> {
  const res = await fetch('/api/notifications')
  if (!res.ok) return []
  return res.json()
}

export async function dismissNotification(id: string): Promise<void> {
  await fetch(`/api/notifications/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export async function parseTwitterLineup(notificationId: string, body: string): Promise<TwitterLineupParseResponse> {
  const res = await fetch('/api/twitter-lineups/parse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ notification_id: notificationId, body }),
  })
  if (!res.ok) throw new Error(`Failed to parse lineup: ${res.statusText}`)
  return res.json()
}

export async function fetchTwitterLineups(): Promise<TwitterLineupRecord[]> {
  const res = await fetch('/api/twitter-lineups')
  if (!res.ok) return []
  return res.json()
}

export async function saveTwitterLineup(req: TwitterLineupSaveRequest): Promise<TwitterLineupRecord> {
  const res = await fetch('/api/twitter-lineups', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Failed to save lineup: ${detail}`)
  }
  return res.json()
}

export async function dismissTwitterLineup(team: string): Promise<void> {
  await fetch(`/api/twitter-lineups/${encodeURIComponent(team)}`, { method: 'DELETE' })
}

export async function fetchTeamOwnershipReductions(): Promise<TeamOwnershipReductionsResponse> {
  const res = await fetch('/api/slate/ownership-reductions')
  if (!res.ok) return { slate_id: '', team_ownership_reductions: {} }
  return res.json()
}

export async function saveTeamOwnershipReductions(update: TeamOwnershipReductionsUpdate): Promise<TeamOwnershipReductionsResponse> {
  const res = await fetch('/api/slate/ownership-reductions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(update),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Failed to save ownership reductions: ${detail}`)
  }
  return res.json()
}

export async function fetchPlayerProjectionOverrides(): Promise<PlayerProjectionOverridesResponse> {
  const res = await fetch('/api/slate/projection-overrides')
  if (!res.ok) return { slate_id: '', player_projection_overrides: {} }
  return res.json()
}

export async function savePlayerProjectionOverrides(update: PlayerProjectionOverridesUpdate): Promise<PlayerProjectionOverridesResponse> {
  const res = await fetch('/api/slate/projection-overrides', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(update),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Failed to save projection overrides: ${detail}`)
  }
  return res.json()
}

export async function fetchCacheStatus(): Promise<CacheStatus> {
  try {
    const res = await fetch('/api/run/cache_status')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return { is_gpp: false, fingerprint: '', candidates: null, field_k: null, n_configured_candidates: 0, n_configured_field_k: 0 }
  }
}
