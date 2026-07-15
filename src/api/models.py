"""Pydantic models for API request/response."""
from typing import Optional
from pydantic import BaseModel

from src.platforms.base import Platform


class PathsConfig(BaseModel):
    dk_slate: str = ""
    fd_slate: str = ""
    copula: str = ""
    output_dir: str = "outputs"
    projections: Optional[str] = None
    fd_projections: Optional[str] = None
    batter_pca_model: Optional[str] = None
    batter_score_grid: Optional[str] = None
    batter_pca_model_fd: Optional[str] = None
    batter_score_grid_fd: Optional[str] = None
    projections_source: str = "rotowire"  # "rotowire" or "dailyfantasyfuel"


class SimulationConfig(BaseModel):
    n_sims: int = 15000


class OptimizerConfig(BaseModel):
    n_chains: int = 250
    temperature: float = 0.001
    n_steps: int = 100
    niter_success: int = 25
    n_workers: int = 8
    early_stopping_window: int = 25
    early_stopping_threshold: float = 0.001
    salary_floor: Optional[float] = 45000
    rng_seed: Optional[int] = None
    min_pitcher_value: Optional[float] = None
    min_batter_value: Optional[float] = None


class PortfolioConfig(BaseModel):
    size: int = 20
    target_percentile: int = 90
    target_score: Optional[float] = None


class GppConfig(BaseModel):
    n_candidates: int = 10000
    n_field_lineups: int = 5000
    n_field_samples: int = 3
    holdout_fraction: float = 0.0
    candidate_batch_size: int = 500
    max_attempts_multiplier: int = 50
    seed_optimal_lineups: bool = False
    # Seed the pool with per-sim optimal lineups: the roster ILP solved
    # against individual simulation draws' realized scores (each seed wins at
    # least one simulated world). n_sim_optimals = how many sims to solve,
    # stratified across slate-total deciles; duplicates are dropped.
    seed_sim_optimal_lineups: bool = False
    n_sim_optimals: int = 300
    dump_candidate_pool: bool = False
    # Diagnostic: N > 0 samples N sims post-run and solves the per-sim optimal
    # lineup ILP to measure how much of the model's own ceiling the candidate
    # pool captures (writes pool_ceiling_sim.csv). 0 disables (no overhead).
    measure_sim_ceiling: int = 0
    candidate_floor_relief: int = 2500
    refine_rounds: int = 2
    refine_top: int = 150
    refine_mutants: int = 8
    refine_holdout_fraction: float = 0.3
    final_n_field_samples: int = 5
    # Tail bypass: admit the top N below-ev_floor candidates by per-candidate
    # sim-p99 (a ceiling statistic mean EV undervalues) into the fresh
    # re-score; they must keep fresh EV >= tail_bypass_ev_floor to reach the
    # selector. 0 disables.
    # Shape constraints for per-sim optimal seeds (ceiling-first round 3):
    # unconstrained per-world argmax optima are structurally unlike real
    # top-1% lineups. min_secondary >= 1 requires a second team with that
    # many batters; salary_floor overrides the optimizer floor for these
    # solves only (null = optimizer floor).
    sim_optimal_min_stack: int = 4
    sim_optimal_min_secondary: int = 0
    sim_optimal_salary_floor: Optional[float] = None
    # Sim-winner seeding (ceiling-first redesign): sampled lineups from many
    # simulated worlds via per-world score-rank weights — the scaled,
    # diversity-preserving successor to per-sim exact ILP optima.
    seed_sim_winners: bool = False
    n_sim_winner_worlds: int = 8000
    sim_winner_per_world: int = 1
    sim_winner_temp: float = 0.15
    sim_winner_own_blend: float = 0.25
    # Shape-preserving seed mutation (ceiling-first round-6 follow-up): each
    # seed parent (sim_optimal + sim_winner) is expanded with N mutants whose
    # team-stack profile matches the parent exactly (same-team batter swaps;
    # pitcher swaps re-checked for opponent conflicts). Additive on top of
    # n_candidates, like refinement mutants. 0 disables.
    seed_mutants_per_parent: int = 0
    seed_mutant_salary_locality: float = 2000.0
    seed_mutant_pitcher_weight: float = 0.10
    tail_bypass_n: int = 2000
    tail_bypass_ev_floor: float = -1.0
    # Tail-metric computation in ContestScorer (ceiling-first redesign):
    # tail_ev = expected gross dollars from payout ranks paying
    # >= tail_ev_min_gross only; p_beat99 = P(candidate beats the simulated
    # field's p99). Adds a second kernel pass (~doubles scoring time).
    compute_tail_metrics: bool = True
    tail_ev_min_gross: float = 100.0
    # Funnel + selector currency (ceiling-first redesign, Phases 2e/3).
    # funnel_mode: "ev_first" (EV floor primary, tail lane = tail_bypass_n
    # side door) | "tail_first" (top tail_admit_n by tail_metric admitted,
    # held only to ev_guardrail; EV floor lane persists as cash anchor).
    # selector_score: "mean_ev" | "tail" (EV term = fresh tail currency,
    # first ceil(cash_anchor_fraction × size) picks stay on mean EV).
    funnel_mode: str = "ev_first"
    tail_metric: str = "tail_ev"
    tail_admit_n: int = 6000
    ev_guardrail: float = -1.0
    selector_score: str = "mean_ev"
    cash_anchor_fraction: float = 0.25
    # Safety cap on the fresh-rescore slice. The slice itself is defined by
    # ev_floor (rescore everything at/above it, then drop what falls below on
    # fresh EVs); this cap only bounds memory/time on pathological slates.
    final_rescore_top: int = 20000
    evw_base: float = 0.10
    evw_max: float = 0.40
    ev_floor: float = 0.20
    field_source: str = "simulated"
    historical_n_slates: int = 10
    dupe_penalty: bool = False
    # Coefficients fitted by scripts/fit_dupe_model.py on the contest-standings
    # archive (32 contests, 2026-07-04); intercept is calibrated to the
    # reference 14,863-entry DK Classic GPP.
    dupe_intercept: float = 3.698
    dupe_log_own_coef: float = 0.212
    dupe_salary_coef: float = 0.089
    dupe_stack_coef: float = 0.024
    dupe_min_gross_payout: float = 15.0


class AppConfig(BaseModel):
    platform: Platform = Platform.DRAFTKINGS
    paths: PathsConfig = PathsConfig()
    simulation: SimulationConfig = SimulationConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    gpp: GppConfig = GppConfig()


class PlayerRow(BaseModel):
    player_id: int
    name: str
    position: str
    team: str
    salary: int
    mean: Optional[float] = None


class LineupResult(BaseModel):
    lineup_index: int
    p_hit_target: float
    lineup_salary: int
    mean_ev: Optional[float] = None
    players: list[PlayerRow]
    upload_tag: Optional[str] = None
    entry_fee: Optional[str] = None
    contest_name: Optional[str] = None


class PortfolioResult(BaseModel):
    lineups: list[LineupResult]


class SlateOption(BaseModel):
    slate_id: str
    name: str
    is_default: bool


class SlateListResponse(BaseModel):
    date: Optional[str] = None
    slates: list[SlateOption]


class ProjectionsStatus(BaseModel):
    exists: bool
    path: Optional[str] = None
    last_modified: Optional[float] = None  # Unix timestamp
    age_seconds: Optional[float] = None
    row_count: Optional[int] = None
    fetch_timestamp_utc: Optional[float] = None  # Unix seconds, from metadata
    unconfirmed_count: Optional[int] = None
    no_changes: Optional[bool] = None  # None = fewer than 2 fetches recorded
    is_fresh: Optional[bool] = None  # True=fresh, False=stale, None=unknown


class TeamStatus(BaseModel):
    team: str
    excluded: bool
    exclusion_scope: str = "none"   # 'none' | 'candidates' | 'both'


class GameStatus(BaseModel):
    game: str
    away: str
    home: str
    excluded: bool
    exclusion_scope: str = "none"   # 'none' | 'candidates' | 'both'
    ppd_pct: float | None = None
    teams: list[TeamStatus]
    game_start_time: str | None = None


class SlateGamesResponse(BaseModel):
    slate_id: str
    games: list[GameStatus]
    excluded_player_ids: list[int] = []


class ExclusionsUpdate(BaseModel):
    slate_id: str
    game_scopes: dict[str, str] = {}    # game_str → 'none'|'candidates'|'both'
    team_scopes: dict[str, str] = {}    # team_str → 'none'|'candidates'|'both'
    game_ppd_pcts: dict[str, float] = {}


class TeamOwnershipReductionsUpdate(BaseModel):
    slate_id: str
    team_ownership_reductions: dict[str, float] = {}


class TeamOwnershipReductionsResponse(BaseModel):
    slate_id: str
    team_ownership_reductions: dict[str, float] = {}


class PlayerProjectionOverridesUpdate(BaseModel):
    slate_id: str
    player_projection_overrides: dict[int, float] = {}


class PlayerProjectionOverridesResponse(BaseModel):
    slate_id: str
    player_projection_overrides: dict[int, float] = {}


class PlayerExclusionStatus(BaseModel):
    player_id: int
    name: str
    position: str
    team: str
    salary: int
    excluded: bool
    exclusion_scope: str = "none"        # effective scope (player + team/game combined)
    individual_scope: str = "none"       # player-level only (ignores team/game)


class SlatePlayersResponse(BaseModel):
    slate_id: str
    players: list[PlayerExclusionStatus]


class PlayerExclusionsUpdate(BaseModel):
    slate_id: str
    player_scopes: dict[str, str] = {}  # str(player_id) → 'none'|'candidates'|'both'


class TwitterNotification(BaseModel):
    id: str
    summary: str
    body: str
    app_name: str
    captured_at: float  # Unix timestamp


class PlayerMatch(BaseModel):
    player_id: int
    name: str
    team: str
    position: str
    salary: int
    match_confidence: str  # "exact" | "fuzzy" | "none"


class ParsedSlot(BaseModel):
    slot: int
    raw_name: str
    position: str
    matches: list[PlayerMatch]


class TwitterLineupParseRequest(BaseModel):
    notification_id: str
    body: str


class TwitterLineupParseResponse(BaseModel):
    team: Optional[str]
    notification_id: str
    slots: list[ParsedSlot]
    team_in_slate: bool
    warning: Optional[str] = None
    is_updated: bool = False


class TwitterLineupSlot(BaseModel):
    slot: int
    player_id: Optional[int]  # None for players not in the slate CSV (placeholders)
    name: str


class TwitterLineupRecord(BaseModel):
    team: str
    notification_id: str
    confirmed_at: float
    slots: list[TwitterLineupSlot]
    locked: bool = True  # old records without the key are treated as locked
    needs_game_confirmation: bool = False  # team plays a doubleheader today; auto-lock was vetoed


class TwitterLineupSaveRequest(BaseModel):
    team: str
    notification_id: str
    slots: list[TwitterLineupSlot]
    locked: bool = True  # old clients that omit the field get locked=True


class DoubleheaderStatusResponse(BaseModel):
    date: str
    doubleheader_teams: list[str] = []
    is_fresh: bool = True
