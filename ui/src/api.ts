import type { AppConfig, ExclusionsUpdate, PlayerExclusionsUpdate, ProjectionsStatus, LineupResult, SlateGamesResponse, SlateListResponse, SlatePlayersResponse } from './types'

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

export async function fetchUnconfirmedPlayerIds(): Promise<number[]> {
  const res = await fetch('/api/projections/unconfirmed')
  if (!res.ok) return []
  const data = await res.json()
  return data.player_ids ?? []
}

export async function fetchProjectionSlates(): Promise<SlateListResponse> {
  const res = await fetch('/api/projections/slates')
  if (!res.ok) throw new Error(`Failed to load projection slates: ${res.statusText}`)
  return res.json()
}

export async function fetchPortfolio(): Promise<LineupResult[]> {
  const res = await fetch('/api/portfolio')
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
