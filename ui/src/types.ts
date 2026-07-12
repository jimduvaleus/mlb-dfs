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
  min_pitcher_value: number | null
  min_batter_value: number | null
}

export interface PortfolioConfig {
  size: number
  target_percentile: number
  target_score: number | null
}

export interface GppConfig {
  n_candidates: number
  n_field_lineups: number
  n_field_samples: number
  holdout_fraction: number
  candidate_batch_size: number
  max_attempts_multiplier: number
  seed_optimal_lineups: boolean
  seed_sim_optimal_lineups: boolean
  n_sim_optimals: number
  dump_candidate_pool: boolean
  measure_sim_ceiling: number
  candidate_floor_relief: number
  refine_rounds: number
  refine_top: number
  refine_mutants: number
  refine_holdout_fraction: number
  final_n_field_samples: number
  final_rescore_top: number
  tail_bypass_n: number
  tail_bypass_ev_floor: number
  compute_tail_metrics: boolean
  tail_ev_min_gross: number
  evw_base: number
  evw_max: number
  ev_floor: number
  field_source: 'simulated' | 'historical'
  historical_n_slates: number
  dupe_penalty: boolean
  dupe_intercept: number
  dupe_log_own_coef: number
  dupe_salary_coef: number
  dupe_stack_coef: number
  dupe_min_gross_payout: number
}

export interface AppConfig {
  platform: PlatformType
  paths: PathsConfig
  simulation: SimulationConfig
  optimizer: OptimizerConfig
  portfolio: PortfolioConfig
  gpp: GppConfig
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

export interface ProjectionPlayerRow {
  player_id: number
  name: string
  position: string
  team: string
  salary: number
  slot: number | null      // 1–9 batters, 10 pitchers, null if missing
  slot_confirmed: boolean
  mean: number             // final scaled projection value (override applied if set)
  ownership_pct: number | null  // heuristic projected ownership %
  is_overridden: boolean
}

export interface LineupResult {
  lineup_index: number
  p_hit_target: number
  lineup_salary: number
  mean_ev?: number | null
  players: PlayerRow[]
  upload_tag?: string | null
  entry_fee?: string | null
  contest_name?: string | null
  entry_sort_order?: number | null
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
  | 'ppd_applied'
  | 'compute_target'
  | 'optimize_lineup'
  | 'portfolio_stats'
  | 'contest_ev_start'
  | 'contest_ev_complete'
  | 'upload_files'
  | 'gpp_optimal_start'
  | 'gpp_optimal_progress'
  | 'gpp_optimal_done'
  | 'gpp_sim_optimal_start'
  | 'gpp_sim_optimal_progress'
  | 'gpp_sim_optimal_done'
  | 'gpp_generate_start'
  | 'gpp_generate_progress'
  | 'gpp_generate_done'
  | 'gpp_score_start'
  | 'gpp_field_progress'
  | 'gpp_score_progress'
  | 'gpp_score_done'
  | 'gpp_refine_start'
  | 'gpp_refine_progress'
  | 'gpp_refine_done'
  | 'gpp_rescore_start'
  | 'gpp_rescore_field_progress'
  | 'gpp_rescore_score_progress'
  | 'gpp_rescore_done'
  | 'gpp_field_inject'
  | 'gpp_select_progress'
  | 'gpp_mv_select_progress'
  | 'gpp_hybrid_select_progress'
  | 'gpp_det_risk_start'
  | 'gpp_det_select_progress'
  | 'gpp_holdout'
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

export interface ContestEvResult {
  lineup_index: number
  cash_rate: number | null
  beat_pct: number | null
  ev_gap: number | null
  field_gap: number | null
  fragile: boolean | null
}

export interface ContestEvCompleteEvent extends SSEEvent {
  stage: 'contest_ev_complete'
  ev_results: ContestEvResult[]
}

export interface GppOptimalProgressEvent extends SSEEvent {
  stage: 'gpp_optimal_progress'
  n: number
  total: number
}

export interface GppGenerateProgressEvent extends SSEEvent {
  stage: 'gpp_generate_progress'
  n: number
}

export interface GppGenerateDoneEvent extends SSEEvent {
  stage: 'gpp_generate_done'
  n_generated: number
  from_cache?: boolean
  team_distribution?: Record<string, number>
}

export interface GppFieldProgressEvent extends SSEEvent {
  stage: 'gpp_field_progress'
  n_done: number
  n_total: number
}

export interface GppScoreProgressEvent extends SSEEvent {
  stage: 'gpp_score_progress'
  batches_done: number
  batches_total: number
}

export interface GppRescoreFieldProgressEvent extends SSEEvent {
  stage: 'gpp_rescore_field_progress'
  n_done: number
  n_total: number
}

export interface GppRescoreScoreProgressEvent extends SSEEvent {
  stage: 'gpp_rescore_score_progress'
  batches_done: number
  batches_total: number
}

export interface GppRefineProgressEvent extends SSEEvent {
  stage: 'gpp_refine_progress'
  round: number
  rounds: number
  n_parents: number
  n_mutants: number
  pool_size: number
  best_ev: number
  n_beat_parent: number
  top_k: number
  n_in_topk: number
  topk_ev_before: number
  topk_ev_after: number
  best_swap_out: string[]
  best_swap_in: string[]
  best_swap_ev_delta: number
  best_mutant_ev: number
  // Present only when refine_holdout_fraction > 0
  holdout_fraction?: number
  topk_ev_holdout_before?: number
  topk_ev_holdout_after?: number
  best_swap_ev_delta_holdout?: number
}

export interface GppSelectProgressEvent extends SSEEvent {
  stage: 'gpp_select_progress'
  round: number
  lineup_index: number
  lineup_ev: number
  n_covered: number
  pct_covered: number
}

export interface GppDetRiskStartEvent extends SSEEvent {
  stage: 'gpp_det_risk_start'
  risk: number
  risk_index: number
  total_risks: number
}

export interface GppDetSelectProgressEvent extends SSEEvent {
  stage: 'gpp_det_select_progress'
  step: number
  portfolio_size: number
  lineup_ev: number
  partial_var: number
  score: number
  n_remaining: number
  risk: number
  risk_index: number
  total_risks: number
}

export interface PortfolioSweepEntry {
  risk: number
  lineups: LineupResult[]
}

export interface CompleteEvent extends SSEEvent {
  stage: 'complete'
  portfolio: LineupResult[]
  n_lineups: number
  optimal_lineups?: LineupResult[]
  portfolio_sweep?: PortfolioSweepEntry[]
}

export interface StoppedEvent extends SSEEvent {
  stage: 'stopped'
  portfolio: LineupResult[]
  n_lineups: number
  optimal_lineups?: LineupResult[]
  portfolio_sweep?: PortfolioSweepEntry[]
}

export interface ErrorEvent extends SSEEvent {
  stage: 'error'
  message: string
}

export interface GppFieldInjectEvent extends SSEEvent {
  stage: 'gpp_field_inject'
  n_field: number
  n_k: number
}

export interface CacheStatus {
  fingerprint: string
  candidates: number | null   // null = no cache
  field_k: number | null      // null = no cache
  n_configured_candidates: number
  n_configured_field_k: number
  is_gpp: boolean
  n_batter_teams: number
}

export type RunStatus = 'idle' | 'running' | 'complete' | 'stopped' | 'error' | 'replacing' | 'reselecting'

export interface MergePlayer {
  name: string
  team: string
  reason?: string
  player_id?: number
  is_pitcher?: boolean
  partial_mean?: number
  partial_std_dev?: number
}

export interface CappedPlayer {
  name: string
  team: string
  markets: string[]  // raw market keys e.g. ["home_runs", "stolen_bases"]
}

export interface LowTeamProjection {
  team: string
  total: number
}

export interface FallbackTeam {
  team: string
  game: string
  count: number
}

export interface MissingOptPlayer {
  name: string
  team: string
  markets: string[]
}

export interface TeamNameWarning {
  game: string       // e.g. "NYY@MIL"
  team_name: string  // raw name CNO used, e.g. "Milwaukee Brewers"
  market: string     // "Run Line" or "Total Runs"
}

export interface HeuristicPlayer {
  player_id: number
  name: string
  team: string
  salary: number
  mean: number
  source: string
  reason?: string
}

export interface MergeInfo {
  secondarySource: string
  count: number
  players: MergePlayer[]
  cappedPlayers?: CappedPlayer[]
  lowTeamProjections?: LowTeamProjection[]
  fallbackTeams?: FallbackTeam[]
  missingOptPlayers?: MissingOptPlayer[]
  heuristicPlayers?: HeuristicPlayer[]
  teamNameWarnings?: TeamNameWarning[]
}

export interface OwnershipSyncResult {
  status: 'synced' | 'out_of_sync' | 'unavailable' | 'error'
  spearman_r?: number
  max_diff?: number
  n_checked?: number
  reason?: string
}

// Slate game/team exclusion types
export type ExclusionScope = 'none' | 'candidates' | 'both'

export interface TeamStatus {
  team: string
  excluded: boolean
  exclusion_scope: ExclusionScope
}

export interface GameStatus {
  game: string
  away: string
  home: string
  excluded: boolean
  exclusion_scope: ExclusionScope
  ppd_pct?: number | null
  teams: TeamStatus[]
  game_start_time?: string | null
}

export interface SlateGamesResponse {
  slate_id: string
  games: GameStatus[]
  excluded_player_ids: number[]
}

export interface ExclusionsUpdate {
  slate_id: string
  game_scopes: Record<string, ExclusionScope>
  team_scopes: Record<string, ExclusionScope>
  game_ppd_pcts?: Record<string, number>
}

export interface TeamOwnershipReductionsUpdate {
  slate_id: string
  team_ownership_reductions: Record<string, number>
}

export interface TeamOwnershipReductionsResponse {
  slate_id: string
  team_ownership_reductions: Record<string, number>
}

export interface PlayerProjectionOverridesUpdate {
  slate_id: string
  player_projection_overrides: Record<number, number>
}

export interface PlayerProjectionOverridesResponse {
  slate_id: string
  player_projection_overrides: Record<number, number>
}

// Player-level exclusion types
export interface PlayerExclusionStatus {
  player_id: number
  name: string
  position: string
  team: string
  salary: number
  excluded: boolean
  exclusion_scope: ExclusionScope
  individual_scope: ExclusionScope
}

export interface SlatePlayersResponse {
  slate_id: string
  players: PlayerExclusionStatus[]
}

export interface PlayerExclusionsUpdate {
  slate_id: string
  player_scopes: Record<string, ExclusionScope>  // key is string(player_id)
}

export interface TwitterNotification {
  id: string
  summary: string
  body: string
  app_name: string
  captured_at: number
  could_be_lineup?: boolean
  lineup_team?: string | null
  is_current_slate?: boolean
  lineup_team_in_slate?: boolean
}

export interface PlayerMatch {
  player_id: number
  name: string
  team: string
  position: string
  salary: number
  match_confidence: 'exact' | 'fuzzy' | 'none'
}

export interface ParsedSlot {
  slot: number
  raw_name: string
  position: string
  matches: PlayerMatch[]
}

export interface TwitterLineupParseResponse {
  team: string | null
  notification_id: string
  slots: ParsedSlot[]
  team_in_slate: boolean
  warning: string | null
  is_updated: boolean
}

export interface TwitterLineupSlot {
  slot: number
  player_id: number | null  // null for players not in the slate CSV (placeholders)
  name: string
}

export interface TwitterLineupRecord {
  team: string
  notification_id: string
  confirmed_at: number
  slots: TwitterLineupSlot[]
  locked: boolean
  needs_game_confirmation: boolean
}

export interface TwitterLineupSaveRequest {
  team: string
  notification_id: string
  slots: TwitterLineupSlot[]
  locked: boolean
}

export interface DoubleheaderStatusResponse {
  date: string
  doubleheader_teams: string[]
  is_fresh: boolean
}

export interface ContestNameCollisionCandidate {
  player_id: number
  team: string
  salary: number
  fpts: number
  drafted: number
  suggested: boolean
}

export interface ContestNameCollision {
  name: string
  candidates: ContestNameCollisionCandidate[]
}

export interface ContestAnalysisResponse {
  player_fpts: Record<string, number>
  collisions: ContestNameCollision[]
}

// ---------------------------------------------------------------------------
// Late swap
// ---------------------------------------------------------------------------

export interface LateSwapPlayer {
  player_id: number | null
  name: string
  team: string
  position: string
  eligible_positions: string[]
  salary: number | null
  mean: number | null
  game: string
  game_start_time: string
}

export interface LateSwapSwappedIn extends LateSwapPlayer {
  locked: boolean
}

export interface LateSwapSlot {
  slot_index: number
  slot_position: string
  player: LateSwapPlayer | null   // null = empty cell (unfilled reservation)
  locked: boolean
  missing_from_slate: boolean
  swapped_in: LateSwapSwappedIn | null
  swap_source: 'auto' | 'manual' | null
}

export interface LateSwapWarning {
  slot_index: number | null
  reason: string
}

export interface LateSwapEntry {
  entry_id: string
  source_file: string
  contest_name: string
  contest_id: string
  entry_fee: string
  n_swappable: number
  warnings: LateSwapWarning[]
  slots: LateSwapSlot[]
}

export type LateSwapStatus = 'ok' | 'no_slate' | 'no_entries' | 'unsupported_platform'

export interface LateSwapState {
  status: LateSwapStatus
  now: string | null
  files: { file_name: string; n_entries: number }[]
  entries: LateSwapEntry[]
  bulk_marked_player_ids: number[]
  bulk_marked_teams: string[]
  teams: string[]
  last_run_at: string | null
  written_files: string[]
}

export interface LateSwapRunRequest {
  entry_marks: Record<string, number[]>
  bulk_marked_player_ids: number[]
  bulk_marked_teams: string[]
}

export interface LateSwapCandidate {
  player_id: number
  name: string
  team: string
  position: string
  eligible_positions: string[]
  salary: number
  mean: number | null
  score: number
  newly_confirmed?: boolean
}

export interface LateSwapCandidatesResponse {
  candidates: LateSwapCandidate[]
  max_salary: number | null
}

export interface LateSwapOverrideResponse {
  entry: LateSwapEntry
  written_files: string[]
  last_run_at?: string | null
}
