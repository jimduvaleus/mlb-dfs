// Mirrors Pydantic models from src/api/models.py

export interface PathsConfig {
  dk_slate: string
  copula: string
  output_dir: string
  projections: string | null
  batter_pca_model: string | null
  batter_score_grid: string | null
}

export interface SimulationConfig {
  n_sims: number
}

export interface OptimizerConfig {
  n_chains: number
  temperature: number
  n_steps: number
  niter_success: number
  n_workers: number
  early_stopping_window: number
  early_stopping_threshold: number
  salary_floor: number | null
  rng_seed: number | null
}

export interface PortfolioConfig {
  size: number
  target_percentile: number
  target_score: number | null
}

export interface AppConfig {
  paths: PathsConfig
  simulation: SimulationConfig
  optimizer: OptimizerConfig
  portfolio: PortfolioConfig
}

export interface PlayerRow {
  player_id: number
  name: string
  position: string
  team: string
  salary: number
}

export interface LineupResult {
  lineup_index: number
  p_hit_target: number
  lineup_salary: number
  players: PlayerRow[]
}

export interface ProjectionsStatus {
  exists: boolean
  path: string | null
  last_modified: number | null  // Unix timestamp (seconds)
  age_seconds: number | null
  row_count: number | null
}

// SSE event payloads
export type SSEStage =
  | 'load_slate'
  | 'simulate'
  | 'compute_target'
  | 'optimize_lineup'
  | 'complete'
  | 'error'

export interface SSEEvent {
  stage: SSEStage
  timestamp: number  // Unix ms
  [key: string]: unknown
}

export interface LoadSlateEvent extends SSEEvent {
  stage: 'load_slate'
  n_players: number
  n_teams: number
  n_units: number
}

export interface SimulateEvent extends SSEEvent {
  stage: 'simulate'
  n_sims: number
}

export interface ComputeTargetEvent extends SSEEvent {
  stage: 'compute_target'
  target: number
  percentile: number | null
}

export interface OptimizeLineupEvent extends SSEEvent {
  stage: 'optimize_lineup'
  lineup_index: number
  total: number
  score: number
}

export interface CompleteEvent extends SSEEvent {
  stage: 'complete'
  portfolio: LineupResult[]
  n_lineups: number
}

export interface ErrorEvent extends SSEEvent {
  stage: 'error'
  message: string
}

export type RunStatus = 'idle' | 'running' | 'complete' | 'error'

// Slate game/team exclusion types
export interface TeamStatus {
  team: string
  excluded: boolean
}

export interface GameStatus {
  game: string
  away: string
  home: string
  excluded: boolean
  teams: TeamStatus[]
}

export interface SlateGamesResponse {
  slate_id: string
  games: GameStatus[]
}

export interface ExclusionsUpdate {
  slate_id: string
  excluded_teams: string[]
  excluded_games: string[]
}
