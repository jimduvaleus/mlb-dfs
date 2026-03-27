import type { AppConfig, ProjectionsStatus, LineupResult } from './types'

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
