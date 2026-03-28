"""Pydantic models for API request/response."""
from typing import Optional
from pydantic import BaseModel


class PathsConfig(BaseModel):
    dk_slate: str = ""
    copula: str = ""
    output_dir: str = "outputs"
    projections: Optional[str] = None
    batter_pca_model: Optional[str] = None
    batter_score_grid: Optional[str] = None


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


class PortfolioConfig(BaseModel):
    size: int = 20
    target_percentile: int = 90
    target_score: Optional[float] = None


class AppConfig(BaseModel):
    paths: PathsConfig = PathsConfig()
    simulation: SimulationConfig = SimulationConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    portfolio: PortfolioConfig = PortfolioConfig()


class PlayerRow(BaseModel):
    player_id: int
    name: str
    position: str
    team: str
    salary: int


class LineupResult(BaseModel):
    lineup_index: int
    p_hit_target: float
    lineup_salary: int
    players: list[PlayerRow]


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


class GameStatus(BaseModel):
    game: str
    away: str
    home: str
    excluded: bool
    teams: list[TeamStatus]


class SlateGamesResponse(BaseModel):
    slate_id: str
    games: list[GameStatus]
    excluded_player_ids: list[int] = []


class ExclusionsUpdate(BaseModel):
    slate_id: str
    excluded_teams: list[str]
    excluded_games: list[str]


class PlayerExclusionStatus(BaseModel):
    player_id: int
    name: str
    position: str
    team: str
    salary: int
    excluded: bool


class SlatePlayersResponse(BaseModel):
    slate_id: str
    players: list[PlayerExclusionStatus]


class PlayerExclusionsUpdate(BaseModel):
    slate_id: str
    excluded_player_ids: list[int]
