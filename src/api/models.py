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
    projections_source: str = "rotowire"  # "rotowire", "dailyfantasyfuel", "market_odds", or "sabersim"


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
    # Round-10 selector objective (plans/variants_round10.yaml):
    # "det" (default) | "kelly" (greedy expected-log-growth on the fresh
    # robust_payout; risk tier → bankroll B = fee × size × {1.25,1.5,2,4,8},
    # kelly_bankroll_mult scales the table) | "coverage" (greedy max-coverage
    # on fresh per-world beat-p999 bits; single risk tier).
    selector_mode: str = "det"
    kelly_bankroll_mult: float = 1.0
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
    # External pool mode: which EV currency the greedy selector ranks on.
    #   "roi"     — the contest's SaberSim-simulated ROI column (default).
    #   "prj_own" — our own projected score minus projected ownership, the
    #               ownership penalty scaled by the contest's implied field
    #               size (prize pool / entry fee):
    #                   EV = proj_score - proj_ownership * (field_size / own_scale)
    #               See compute_prj_own_ev in src/api/external_pool.py.
    #   "p_win"   — simulated P(win): mean_over_worlds(percentile ** n)
    #               against an ownership-sampled opponent field, n = sharpness
    #               * implied field size. See compute_p_win in
    #               src/api/external_pool.py. Falls back to "prj_own" at
    #               runtime if field/sim generation fails.
    # Under "prj_own"/"p_win" Saber's ROI is not consulted at all, so
    # external_pool_roi_floor_pct, external_pool_ceiling_weight and
    # external_pool_cash_anchor_fraction are all inert; the pool-wide
    # external_pool_proj_score_pct cull still applies.
    external_pool_ev_type: str = "roi"
    # prj_own calibration constant: the field size at which one point of
    # summed lineup ownership costs one projected point. Calibrated
    # 2026-07-27 to 30,000 from two indifference anchors — at ~10,000
    # entries (proj 95, own 60) ties (proj 105, own 90), i.e. 10 projected
    # points per 30 ownership points; and 1,000 entries weighs ownership 10x
    # less, which the linear field-size scaling gives for free. Lower it to
    # make every contest more leverage-driven, raise it for more
    # projection-driven. See compute_prj_own_ev in external_pool.py.
    external_pool_own_scale: float = 30_000.0
    # p_win exponent multiplier: n = sharpness * implied_field_size. 1.0 is
    # literal P(win); lower values soften toward P(top X%), which has more
    # effective events per lineup at a fixed sim budget. See compute_p_win.
    # Calibrated 2026-07-27: a sharpness grid (0.05/0.1/0.3/0.5/1.0) over 4
    # settled slates showed top1_rate monotonically decreasing as sharpness
    # rises, in every slate individually — 1.0's aggressive "beat basically
    # the whole field" target amplifies the percentile-estimation noise near
    # q=1 (q^n is exponentially sensitive to small errors in q as n grows).
    # 0.05 was the best of the tested points (t=+5.78 vs 1.0, 4/4 slates) and
    # is monotonic through the whole tested range, so lower may do better
    # still — not yet tested below 0.05. See scripts/sim_evaluate_portfolios.py
    # --build --sharpness.
    external_pool_pwin_sharpness: float = 0.05
    # p_win two-stage winner's-curse guard: each contest's post-floor pool
    # is culled to the top N by a p_win estimate on one sim/field draw
    # BEFORE a second, independent draw ranks the survivors — a lineup that
    # only looks good on the draw used to pick it can't reach the draw used
    # to rank it (mirrors the internal pipeline's fresh-rescore pattern).
    # <= 0 disables the cull (rank the whole pool on the second draw alone).
    # With external_pool_pwin_admit_multiplier below, this value is a FLOOR,
    # not the literal cull size -- see that field for the full calibration
    # history, including a correction: an earlier flat-admit_n sweep (run
    # with scripts/sim_evaluate_portfolios.py --build, a single synthetic
    # 150-entry contest group) found flat 250 beating 1000/2000. That result
    # didn't survive a more faithful test (scripts/sweep_admit_n_scaling.py,
    # each slate's REAL multi-contest entry-count breakdown, 7 settled
    # slates, 07/27 excluded for an unrelated postponement confound): under
    # real conditions flat 250 was the WORST option tested, not the best --
    # every alternative (no cull, flat 500-5000, the scaled formula) beat it,
    # most with real significance. The single-contest test couldn't see this
    # because it has no cross-contest shared-pool depletion to get wrong.
    external_pool_pwin_admit_n: int = 250
    # Scales the p_win cull by each contest's OWN entry count -- the number
    # of OUR entries actually uploaded to that contest (ContestGroup.entries,
    # from the parsed DK Entries CSV), not the contest's field size or any
    # entry-max cap -- instead of applying one flat number everywhere:
    # effective_admit_n = max(external_pool_pwin_admit_n, round(multiplier *
    # n_entries)). Motivated by production data: a flat admit_n gives a
    # large-fill contest (e.g. 72 entries) a much tighter *relative*
    # reservoir than a small one (e.g. 14), and on two live slates the
    # single biggest-entry contest landed hit99=0 both times while every
    # smaller contest on the same slate caught at least one -- despite
    # having the most entries (most chances) of any of them.
    #
    # Calibrated 2026-07-28 (sweep_admit_n_scaling.py, 7 settled slates,
    # real per-contest entry counts, 07/27 excluded): floor=250/multiplier=12
    # was the only rule to beat the flat-250 baseline on every single slate
    # (7/7, t=+3.50 on mean real-field percentile) -- several flat values
    # (1500, 2000, 5000) scored a higher raw mean on this sample, but each
    # lost on 1-2 individual slates, and none was distinguishable from "no
    # cull at all" with any confidence (|t|<1.6 in every case). The scaled
    # rule was preferred over a bigger flat number specifically for that
    # consistency -- a flat number tuned to one sample's contest-size mix is
    # a bet that the mix doesn't shift; the scaled rule adapts by
    # construction. 0.0 disables scaling (flat admit_n).
    external_pool_pwin_admit_multiplier: float = 12.0
    # p_win simulated opponent field size. 0 = auto (ep.pwin_field_size:
    # grows gpp.n_field_lineups to the largest contest's implied entry
    # count, capped for memory).
    external_pool_pwin_field_size: int = 0
    # External pool mode: per-contest ROI percentile floor for the pre-Det
    # cull (see allocate_contests in src/api/external_pool.py). A raw ROI
    # cutoff doesn't generalize across contests of different sizes/payout
    # structures, so the floor is expressed as "cull the bottom N% of this
    # contest's own ROI distribution" — computed independently per contest.
    external_pool_roi_floor_pct: float = 40.0
    # Pool-wide floor (distinct from the per-contest ROI floor above): culls
    # the bottom N% of *projected score* (sum of each lineup's rostered
    # players' projected mean) once across the entire pool, before any
    # per-contest allocation runs — see compute_pool_proj_scores /
    # allocate_contests in external_pool.py. 0 disables the cull.
    external_pool_proj_score_pct: float = 0.0
    # Ceiling lean: ranks the post-floor pool by roi + weight * (residualized,
    # normalized ROI StDev) instead of plain roi (see compute_ceiling_ev in
    # external_pool.py) — no-ops when the export has no ROI StDev column.
    # cash_anchor_fraction mirrors the internal pipeline's ceiling-first
    # cash-anchor block: that fraction of each contest's picks still ranks
    # on plain roi regardless of ceiling_weight. Unvalidated against real
    # settled external-pool outcomes (too little archive data yet) — start
    # conservative and recalibrate as archive/analyze_external_pool.py
    # accumulates more settled slates.
    external_pool_ceiling_weight: float = 0.25
    external_pool_cash_anchor_fraction: float = 0.25


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
    # Projected ownership in percentage points (see _serialize_portfolio,
    # which normalizes the internal pipeline's fraction convention).
    ownership: Optional[float] = None


class LineupResult(BaseModel):
    lineup_index: int
    p_hit_target: float
    lineup_salary: int
    mean_ev: Optional[float] = None
    # Lineup totals: summed projected score, and summed ownership in
    # percentage points. None when any rostered player lacks the input.
    lineup_mean: Optional[float] = None
    lineup_ownership: Optional[float] = None
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
