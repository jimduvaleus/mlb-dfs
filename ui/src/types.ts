// Mirrors Pydantic models from src/api/models.py

export interface PathsConfig {
  dk_slate: string
  copula: string
  output_dir: string
  projections: string | null
  batter_pca_model: string | null
  batter_score_grid: string | null
  projections_source: string
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
  objective: string
  payout_beta: number | null
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
  assigned_position?: string
  team: string
  salary: number
  slot?: number | null
  slot_confirmed?: boolean
}

export interface LineupResult {
  lineup_index: number
  p_hit_target: number
  lineup_salary: number
  players: PlayerRow[]
  upload_tag?: string | null
  entry_fee?: string | null
  contest_name?: string | null
}

export interface SlateOption {
  slate_id: string
  name: string
  is_default: boolean
}

export interface SlateListResponse {
  date: string | null
  slates: SlateOption[]
}

export interface ProjectionsStatus {
  exists: boolean
  path: string | null
  last_modified: number | null  // Unix timestamp (seconds)
  age_seconds: number | null
  row_count: number | null
  fetch_timestamp_utc: number | null  // Unix seconds, from metadata
  unconfirmed_count: number | null
  no_changes: boolean | null  // null = fewer than 2 fetches recorded
  is_fresh: boolean | null  // true=fresh, false=stale, null=unknown
}

// SSE event payloads
export type SSEStage =
  | 'load_slate'
  | 'simulate'
  | 'compute_target'
  | 'calibrate_beta'
  | 'optimize_lineup'
  | 'complete'
  | 'stopped'
  | 'error'

export interface SSEEvent {
  stage: SSEStage
  timestamp: number  // Unix ms
  [key: string]: unknown
}

export interface LoadSlateEvent extends SSEEvent {
  stage: 'load_slate'
  n_teams: number
  n_batters: number
  n_pitchers: number
  multi_pitcher_teams: Record<string, number>
  n_teams_excluded: number
  n_batters_ind_excluded: number
  n_pitchers_ind_excluded: number
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
  sims_covered: number
  sims_remaining: number
}

export interface CompleteEvent extends SSEEvent {
  stage: 'complete'
  portfolio: LineupResult[]
  n_lineups: number
}

export interface StoppedEvent extends SSEEvent {
  stage: 'stopped'
  portfolio: LineupResult[]
  n_lineups: number
}

export interface ErrorEvent extends SSEEvent {
  stage: 'error'
  message: string
}

export type RunStatus = 'idle' | 'running' | 'complete' | 'stopped' | 'error' | 'replacing'

export interface MergePlayer {
  name: string
  team: string
}

export interface MergeInfo {
  secondarySource: string
  count: number
  players: MergePlayer[]
}

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
  excluded_player_ids: number[]
}

export interface ExclusionsUpdate {
  slate_id: string
  excluded_teams: string[]
  excluded_games: string[]
}

// Player-level exclusion types
export interface PlayerExclusionStatus {
  player_id: number
  name: string
  position: string
  team: string
  salary: number
  excluded: boolean
}

export interface SlatePlayersResponse {
  slate_id: string
  players: PlayerExclusionStatus[]
}

export interface PlayerExclusionsUpdate {
  slate_id: string
  excluded_player_ids: number[]
}
