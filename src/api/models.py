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
    objective: str = "expected_surplus"
    payout_beta: Optional[float] = None
    payout_cash_line: Optional[float] = None
    payout_coverage_bonus: float = 0.0
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
    risk: float = 0.0
    portfolio_n_iter: int = 10_000
    portfolio_n_restarts: int = 3
    dump_candidate_pool: bool = False
    candidate_floor_relief: int = 2500
    portfolio_method: str = "det_ev"
    hybrid_n_sims: int = 10_000
    hybrid_max_correlation: float = 0.9
    refine_rounds: int = 2
    refine_top: int = 150
    refine_mutants: int = 8


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


class TwitterLineupSaveRequest(BaseModel):
    team: str
    notification_id: str
    slots: list[TwitterLineupSlot]
    locked: bool = True  # old clients that omit the field get locked=True
