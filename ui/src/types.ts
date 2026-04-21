// Mirrors Pydantic models from src/api/models.py

export type PlatformType = 'draftkings' | 'fanduel'

export interface PathsConfig {
  dk_slate: string
  fd_slate: string
  copula: string
  output_dir: string
  projections: string | null
  fd_projections: string | null
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
  payout_coverage_bonus: number
  min_pitcher_value: number | null
  min_batter_value: number | null
}

export interface PortfolioConfig {
  size: number
  target_percentile: number
  target_score: number | null
}

export interface AppConfig {
  platform: PlatformType
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
  mean?: number | null
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
  | 'portfolio_stats'
  | 'upload_files'
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
  n_pitchers_value_excluded: number
  n_batters_value_excluded: number
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
  // Non-marginal_payout objectives (p_hit, expected_surplus)
  sims_covered?: number
  sims_remaining?: number
  // marginal_payout objective — three-tier coverage counts (sum to n_sims)
  sims_great?: number
  sims_good?: number
  sims_uncovered?: number
  // marginal_payout objective — portfolio coverage fractions (0–100) against fixed sim thresholds
  pct_above_p90?: number | null
  pct_above_p99?: number | null
  pct_above_target?: number | null
  target_percentile?: number | null
  objective?: string
}

export interface PortfolioStatsEvent extends SSEEvent {
  stage: 'portfolio_stats'
  target: number
  great_threshold: number
  n_sims: number
  covered_count: number
  covered_mean: number | null
  covered_p50: number | null
  covered_p90: number | null
  covered_p95: number | null
  covered_p99: number | null
  overall_p90: number
  overall_p95: number
  overall_p99: number
  histogram: Array<{ lo: number; hi: number; mid: number; count: number }>
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
  reason?: string
  player_id?: number
  is_pitcher?: boolean
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

export interface TwitterNotification {
  id: string
  summary: string
  body: string
  app_name: string
  captured_at: number
}
