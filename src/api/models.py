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
    #   "proj_top" — rank directly on projected mean, no ROI/ownership
    #               currency at all. Backtested as the best currency found
    #               for recovering a slate's own top-10-real-score lineups
    #               from its pool (needle-in-haystack framing, 50% of 14
    #               archived slates vs. 14% for random at the same budget) —
    #               see the ev_type docstring in allocate_contests for the
    #               full comparison. Trades this for a materially
    #               concentrated portfolio (far fewer distinct teams, high
    #               single-player exposure) that the needle metric doesn't
    #               price as a cost; use `risk` to dial diversity back in.
    #   "topn_coverage" — greedy top-N field-coverage allocation: fills each
    #               contest with whichever remaining candidate would have
    #               finished top-`external_pool_topn_rank` most often against
    #               a sub-sampled opponent field, across many simulated
    #               worlds, then removes the worlds a pick "claimed" so later
    #               picks have to prove themselves elsewhere (hard-threshold
    #               exact set-cover, bit-packed popcount greedy — see
    #               allocate_contests_topn_coverage in external_pool.py).
    #               Produces a SINGLE portfolio (no risk sweep) since greedy
    #               coverage is diversified by construction. Saber's ROI is
    #               not consulted; the pool-wide external_pool_proj_score_pct
    #               cull still applies.
    # Under "prj_own"/"p_win"/"proj_top"/"topn_coverage" Saber's ROI is not consulted at all,
    # so external_pool_roi_floor_pct, external_pool_ceiling_weight and
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
    #
    # RECALIBRATED 2026-07-30 to max(100, 1.5 * entries). Two sweeps, real
    # payout tables, per-contest grading, 8 slates x risks 1/3/5:
    #   round 1  flat250 beat 250x12 / 1000x12 / 2000x12 / flat1000 / no-cull
    #            on 8/8 slates (+2.66% vs the old rule, paired p=0.0006)
    #   round 2  flat100 5.8498 > flat150 5.6973 > flat200 5.5632 >
    #            flat250 5.4661 > flat300 5.4129 > flat400 5.2828
    #            (+7.02% vs flat250, 8/8 slates, sign test p=0.0078)
    # The response is monotone in window size on every individual slate --
    # tighter is better, without exception, and the trend NEVER REVERSED. It
    # stopped only at the fill constraint: mini-MAX takes up to 95 of our
    # entries, so ~95 is the narrowest window that can still fill it.
    #
    # Hence the small multiplier rather than a bare flat 100. mini-MAX permits
    # 150 entries; a fixed 100 would silently leave 50 UNSUBMITTED if volume
    # ever rises -- the same partial-fill trap that made an experimental
    # flat-50 arm post the highest score in round 2a (it kept only its best 50
    # lineups and the per-entry metric divided by entries FILLED, not
    # intended). max(100, 1.5*k) yields 100-143 across this sample, entirely
    # inside the tested-best band, and 225 at a 150-entry contest.
    #
    # This supersedes the 2026-07-28 calibration (sweep_admit_n_scaling.py),
    # which chose floor=250/multiplier=12 and whose result does not survive.
    # That run had BOTH of the evaluation flaws found on 07/30:
    #   * it graded every entry against ONE borrowed contest field
    #     (load_real_field_points) rather than its own contest;
    #   * it used `_FLAT_IMPLIED_ENTRIES = 10_000`, i.e. a FLAT p_win exponent
    #     -- since measured worse than per-contest scaling and reverted
    #     (bc4b59c).
    # The 07/30 rerun fixes both and adds REAL DK payout tables per contest
    # (data/payout_structures/, structure_for_contest).
    external_pool_pwin_admit_n: int = 100
    # Scales the p_win cull by each contest's OWN entry count:
    # effective_admit_n = max(admit_n, round(multiplier * n_entries)).
    # 0.0 disables scaling (flat admit_n) and is the CALIBRATED DEFAULT.
    #
    # The multiplier was introduced 2026-07-28 to give large-fill contests a
    # wider relative reservoir, after the biggest-entry contest landed
    # hit99=0 on two live slates. Under correct evaluation it does the
    # opposite of what was intended: it widens the window exactly where a
    # tight one is best. Per-contest ROI, flat 250 vs 250x12 --
    #     mini-MAX (57-95 of our entries):  2.2260 vs 2.0422   +9.0%
    #     Four-Seamer / Bat Flip:           unchanged to +0.1%
    #     Skipper / Base Hit (1-3 entries): identical -- a 250 reservoir is
    #                                       never exhausted by 3 picks
    # So the contest the multiplier was built for is the one it damages, and
    # every contest small enough to be unaffected is unaffected either way.
    external_pool_pwin_admit_multiplier: float = 0.0
    # p_win simulated opponent field size. 0 = auto (ep.pwin_field_size:
    # grows gpp.n_field_lineups to the largest contest's implied entry
    # count, capped for memory).
    external_pool_pwin_field_size: int = 0
    # Large-field ownership cap for "proj_top" (phase-in, off by default).
    # Restricts proj_top's per-contest candidate pool, for contests with
    # implied_field_size >= 5,000 only, to lineups whose summed ownership
    # (own_scores) falls at or below a percentile of that contest's own
    # ownership distribution (the pool-wide floor's survivors). The
    # percentile itself phases in linearly by contest size, from
    # own_cap_start_pct at the 5,000-entry threshold to own_cap_end_pct at
    # the largest implied field size among that day's own contests (a
    # self-calibrating anchor, not a hardcoded number -- mirrors the
    # backtest's own methodology). 100/100 (the default) is a no-op:
    # percentile(dist, 100) == dist.max(), which nothing exceeds, so every
    # proj_top run is byte-identical to today's until one of these is dialed
    # below 100. See the ev_type docstring in allocate_contests.
    #
    # UNVALIDATED, shape-only finding, not a specific calibrated pair (10
    # archived slates, real payout tables, not yet an EVIDENCE_LOG.md
    # entry): a FLAT percentile cap at any single value (50-95) consistently
    # hurt cash%/top1% vs. uncapped proj_top, monotonically worse as the cap
    # tightened, at both a 6,000+ and a 5,000+ field-size threshold. A
    # GRADUAL phase-in -- loose (near 90-100) right at the threshold,
    # tightening only for the largest fields seen that day -- traded some of
    # that cash-rate consistency (uncapped proj_top still wins cash%/top1%
    # in every tested variant) for more "big" finishes (payout >= 10x entry,
    # the ev_tail convention in tests/bt_core.py's accumulate_currencies)
    # instead of occasional lucky ones. Small sample: a nearby start value
    # (85) performed erratically worse than its neighbors, a sign the exact
    # percentile values aren't yet distinguishable from noise at n=10 -- it's
    # the SHAPE (loose start, gentle tightening, large fields only) that's
    # considered worth shipping as a user-dialable control, not these
    # specific numbers. Leave at 100/100 unless deliberately exploring this.
    external_pool_proj_top_own_cap_start_pct: float = 100.0
    external_pool_proj_top_own_cap_end_pct: float = 100.0
    # External pool mode, proj_top only, off by default: swaps the *ranking
    # signal* itself by field size instead of touching eligibility. Below
    # 5,000 implied entries, proj_top always ranks on plain mean projected
    # score (proj_scores), unaffected by this flag. From 5,000 up to
    # external_pool_proj_top_medium_large_boundary, ranks on each lineup's
    # simulated 95th-percentile score; at/above the boundary, on the 99th
    # percentile. Both percentiles are computed once per run from the
    # already-simulated lineup score matrix (see sim_p95_scores/sim_p99_scores
    # in pipeline.py) — no extra simulation cost.
    #
    # Backtested across 10 archived slates, real payout tables, graded
    # against real contest fields: unlike the ownership cap above, this
    # passed the drop_max robustness check (leave-the-largest-payout-out) —
    # p95 and p99 both beat plain mean ranking with a positive drop_max,
    # not just a lucky top-line ROI number. The medium/large boundary showed
    # a genuine cliff at 10,000 implied entries, not a smooth optimum,
    # traced to a single recurring contest (Bat Flip, ~9,900 implied
    # entries) crossing from p95 to p99 treatment; 15,000 was chosen as a
    # defensible round number inside the flat, well-supported region above
    # that cliff — it does not distinguishably beat 10,000-14,000, it's just
    # not sensitive to the exact recurring-contest-size coincidence that
    # produced the cliff. Combining this with the ownership cap above was
    # also tested and found to hurt regardless of which ranking signal is
    # underneath — leave the cap at 100/100 (off) when using this.
    external_pool_proj_top_ceiling_tiers: bool = False
    external_pool_proj_top_medium_large_boundary: float = 15000.0
    # self_play: sims subsampled from the already-simulated matrix for each
    # round-loop pick (see self_play._ROUND_N_SIMS_DEFAULT / the SHORTLIST
    # RESTRICTION note in self_play.py for why this is small by default --
    # tractability, not accuracy). Clamped to the actual live simulation's
    # n_sims if configured higher.
    external_pool_self_play_round_n_sims: int = 2_000
    # self_play: sims for the bounded post-round-loop precision-refinement
    # pass (see self_play._PRECISE_N_SIMS_DEFAULT / PRECISION REFINEMENT
    # note). 0 disables refinement entirely. Also clamped to the live
    # simulation's n_sims.
    external_pool_self_play_precise_n_sims: int = 20_000
    # self_play: candidates re-scored per round after round 0's mandatory
    # full pass (see self_play.py's SHORTLIST RESTRICTION note).
    external_pool_self_play_shortlist_size: int = 1_000
    # topn_coverage: size of the one-time-per-slate opponent field pool
    # (ContestSimulator.generate_field, ownership-weighted) every contest's
    # per-contest field subsets are re-sliced from. Same order of magnitude
    # as _PWIN_FIELD_CAP/self_play's _SELF_PLAY_POOL_CAP.
    external_pool_topn_field_pool_size: int = 25_000
    # topn_coverage: the literal rank (e.g. 10 = "top-10") a candidate must
    # cross in a simulated world to count as covering it -- the FLOOR of the
    # effective per-contest bar (see external_pool_topn_percentile_floor
    # immediately below, which can push this UP for large fields; it never
    # goes below this flat value).
    external_pool_topn_rank: int = 10
    # topn_coverage: per-contest effective rank = max(external_pool_topn_rank,
    # ceil(this fraction * field_size_g)), clipped to field_size_g -- e.g.
    # 0.001 ("top 0.1%") makes a 17,000-entry field effectively top-17
    # instead of a literal top-10, while fields under ~10,000 entries stay
    # at the flat topn_rank floor (ceil(0.001 * field_size_g) < 10 there).
    # A fixed top-10 bar is a vastly more extreme ask in a huge field (top
    # 0.06% of 17,000) than a small one (top 2% of 500) -- this keeps the
    # bar's real difficulty comparable across contest sizes instead of
    # literally fixed. 0 disables this entirely (pure flat topn_rank, the
    # original behavior). See _topn_effective_rank in external_pool.py.
    external_pool_topn_percentile_floor: float = 0.001
    # topn_coverage: K independent field-pool column-subsets drawn per
    # contest (cheap re-slices of the same already-scored field pool, not K
    # separate field generations) -- avoids overfitting coverage to a single
    # field snapshot's idiosyncrasies. Mirrors ContestScorer's n_field_samples.
    #
    # 5 -> 3 (2026-08-11): scripts/diagnose_topn_variance_decomposition.py
    # varied the sim-world slices and the field draws independently (via the
    # allocator's field_rng_seed) and attributed ~95% of cross-seed portfolio
    # variance to sim-world resampling vs ~36% to field draws -- K was buying
    # little. Since field scoring is the dominant wall-clock cost and scales
    # linearly with K while the (n_sims_g x field_size_g) transient does NOT
    # scale with K at all, cutting K frees time to spend on sim worlds, which
    # is where the variance actually lives.
    external_pool_topn_field_samples: int = 3
    # topn_coverage: additional candidates drawn from the same stacked-
    # lineup generator the threshold field pool uses (see
    # ep.augment_topn_pool_with_generated), merged into the real external
    # pool after 9/10-overlap dedup (every real lineup always wins a
    # conflict, so this only ever adds new shapes, never displaces one).
    # Lets the greedy selector pick a high-performing lineup that's visible
    # in the simulated field but wasn't in the real SaberSim export --
    # previously structurally impossible (only pool.lineups was ever
    # eligible to be picked). Drawn with an independently seeded call from
    # the threshold field pool's own build, by design -- see that
    # function's docstring for why that avoids needing a self_play-style
    # "remove from field eligibility once picked" runtime guard. 0 (default)
    # disables this entirely -- unvalidated idea, off until backtested, same
    # posture as every other speculative topn_coverage/proj_top knob in this
    # file.
    external_pool_topn_generated_pool_size: int = 0
    # topn_coverage: blend weight for generated candidates' sampling
    # ownership vector, toward the self-referential "optimal ownership"
    # game-theoretic signal (src/optimization/leverage.py) instead of plain
    # projected ownership. 0.0 (default) = today's exact behavior --
    # optimal-ownership computation is skipped entirely, not just weighted
    # to zero. 1.0 = generated candidates drawn purely from optimal
    # ownership. Reuses external_pool_pwin_sharpness (already validated for
    # this self-referential-field mechanism, see
    # scripts/analyze_optimal_ownership.py) as the p_opt exponent's
    # sharpness, rather than adding a second unvalidated sharpness knob.
    #
    # UNVALIDATED as a candidate-GENERATION bias -- distinct from the
    # SELECTION-time finding that giving leverage MORE weight as a primary
    # EV/ranking term consistently hurt across 6+ tests (memory
    # project-leverage-session-handoff, "THE PATTERN"). This changes what
    # shapes are AVAILABLE to be picked, not how they're ranked once
    # generated -- untested, not contradicted by that finding -- but same
    # caution applies: off by default until backtested, dial not switch.
    # Only takes effect when external_pool_topn_generated_pool_size > 0.
    external_pool_topn_generated_leverage_weight: float = 0.0
    # topn_coverage: per-contest sim-world budget. Each contest draws its own
    # random sim-world subsample (both to bound cost by n_sims_g and to keep
    # two contests' coverage races from converging on near-duplicate lineups
    # by racing for identical worlds).
    #
    # Fallback only -- see external_pool_topn_sims_min/_reference_field_size/
    # _power below, which are the ACTIVE rule by default. This flat fraction
    # is what _topn_sims_for_field_size falls back to if any of those three
    # is left at 0 (e.g. a user clears one while experimenting).
    external_pool_topn_sims_per_contest_fraction: float = 0.5
    # topn_coverage field-size-aware sizing (ACTIVE by default -- all three
    # must be > 0, which they are; set any to 0 to fall back to the flat
    # fraction above instead):
    #   n_sims_g = clip(round(sims_min * (field_size_g / reference_field_size)
    #              ** power), sims_min, n_sims)
    #
    # CALIBRATED 2026-08-09 (scripts/calibrate_topn_sims_per_contest.py,
    # archived slate 07222026, real correlated sim + the real ContestSimulator
    # field generator + the slate's real 4,049-lineup external candidate
    # pool -- an earlier calibration attempt using i.i.d.-noise synthetic
    # data was discarded as a methodology artifact, see the script's
    # CORRECTNESS NOTE). Measured field-size -> sims-needed-for-0.9-Spearman-
    # convergence, monotonic across every field size tested: 392-1,189
    # entries needed 5,000 sims; 5,945-17,835 entries needed 10,000. Fit via
    # log-log regression across those points. Caveats: a single slate (6
    # field-size points) and a coarse sim-count grid
    # (250/500/1k/2k/5k/10k/25k, so "needed" is quantized, not finely
    # resolved) -- re-run against more slates before trusting the exact
    # numbers for anything beyond the compute-savings intent below. Not a
    # correctness fix: the flat 0.5 fraction this replaces (12,500 sims at
    # n_sims=25,000) already exceeded every measured "needed" point, so this
    # is a compute-savings tune for small contests, not a bug fix.
    #
    # 4,607 -> 9,214 (2x) on 2026-08-11. The calibration above targeted the
    # per-candidate SCORE ranking converging (0.9 Spearman); it did not check
    # that the resulting PORTFOLIO is stable, and it isn't at 1x. Varying only
    # the sim-world slices (scripts/sweep_topn_sim_budget.py, via the
    # allocator's field_rng_seed) moved player-exposure rho 0.847 and
    # stack rho 0.889 between two seeds. Doubling the budget lifts those to
    # 0.883 / 0.942; tripling reaches 0.908 / 0.983. Divergence decays at
    # ~1/sqrt(n) for exposure and faster for stack structure, so this keeps
    # paying, but 2x is the shipped point rather than 3x for MEMORY headroom:
    # peak RSS on the 08/10 slate ran 2.61 / 4.89 / 7.18 GB at 1x/2x/3x, and a
    # slate whose field_size_g reaches the 25,000 field-pool cap would push 3x
    # to ~9.5GB against a ~11GB budget. Wall clock 67s -> 156s there.
    external_pool_topn_sims_min: int = 9_214
    external_pool_topn_sims_reference_field_size: int = 392
    external_pool_topn_sims_power: float = 0.222
    # topn_coverage: SMOOTHED EXCEEDANCE. 0.0 (default) = the original hard
    # crossing indicator `1[score >= threshold]`. > 0 replaces it with
    # `P(threshold <= score)` under the rank-N order statistic's own sampling
    # distribution, whose sd in FPTS is
    # `sqrt(N*(1-N/F)) * dScore/dRank` -- a Rao-Blackwellization of the
    # indicator (strictly lower variance, unbiased for "would this cross a
    # freshly drawn field"). This attacks the binding constraint on every
    # top-heavy objective here: at rank 1 the hard indicator fires ~1.9 times
    # per candidate, split-half rho_full ~0.30
    # (scripts/diagnose_topn_rung_settling.py).
    # 1.0 = the statistically implied width; < 1 sharper, > 1 more aggressive.
    # See allocate_contests_topn_coverage's docstring in external_pool.py.
    # Set external_pool_topn_field_samples: 1 alongside this -- the K draws
    # Monte-Carlo the same threshold noise tau integrates analytically, so
    # keeping K > 1 only multiplies the soft array's memory.
    # DEFAULT OFF pending walk-forward validation, same posture as every other
    # speculative topn_coverage knob here.
    external_pool_topn_smooth_tau_scale: float = 0.0
    # Caps how many simulated worlds the DIVERSITY term's pool correlation is
    # estimated from (evenly strided, so it spans the full world range). 0 =
    # use every world.
    #
    # This exists so `simulation.n_sims` can be raised for p_win's benefit
    # without the correlation dragging peak memory up with it. The two
    # consumers want opposite things: p_win concentrates its weight on the few
    # worlds where a candidate tops the field, so more worlds directly buy it
    # reliability (split-half rho 0.880 at 12,500 worlds/stage -> 0.935 at
    # 25,000); the diversity ordering is a bulk statistic that is already
    # settled at split-half rho_full 0.976-0.999 and gains nothing. Meanwhile
    # precompute_pool is the single largest allocation in the path -- 5.73 GB
    # above baseline for a (5,139 x 50,000) float32 score matrix.
    #
    # 25,000 is a NO-OP at the shipped simulation.n_sims of 25,000 (step=1,
    # byte-identical), and only binds if n_sims is raised above it.
    # Measured, 08/17 slate, n_sims=50,000: 7.24 GB -> 4.24 GB peak together
    # with the p_win-stage interleave and score_field's byte-bounded batching.
    # See scripts/scalecheck_pwin_n_sims.py and compute_pool_corr.
    external_pool_corr_max_sims: int = 25_000
    # External pool mode: per-contest ROI percentile floor for the pre-Det
    # cull (see allocate_contests in src/api/external_pool.py). A raw ROI
    # cutoff doesn't generalize across contests of different sizes/payout
    # structures, so the floor is expressed as "cull the bottom N% of this
    # contest's own ROI distribution" — computed independently per contest.
    external_pool_roi_floor_pct: float = 40.0
    # Pool-wide floor (distinct from the per-contest ROI floor above): culls
    # the bottom N% of *ceiling* — each lineup's own SaberSim "99th"
    # (99th-percentile simulated score) column, falling back to a
    # mean+z*sigma proxy for any lineup without one — once across the
    # entire pool, before any per-contest allocation runs, so the floor
    # keeps high-upside lineups a mean-score floor would otherwise drop for
    # being merely median, and drops median-ceiling lineups a mean-score
    # floor would otherwise keep. See compute_pool_ceiling_scores /
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
    # Rank-normalise the Det selector's EV and diversity terms instead of
    # max-normalising EV and clipping the diversity distance -- see RANK
    # NORMALISATION in DeterminantPortfolioSelector's docstring. Changing
    # this changes what evw_base/evw_max MEAN, so the two move together
    # (see config.yaml's note on the recalibrated sweep range).
    external_pool_rank_normalize: bool = True

    # Zero-inflate the SaberSim quantile grids for batters: a DK hitter scores
    # exactly 0 when he never reaches base and drives nobody in, which happened
    # to 20.6% of rostered batters across 10 archived slates while the grids
    # priced it at 2.19% — a ~9x understatement that compounds into the lineup
    # ceiling (all 8 batters producing: modelled 0.98^8 = 85%, real 0.80^8
    # = 17%). See batter_blank_probability / _zero_inflate_grid in
    # external_pool.py. scratch_prob is the flat, projection-INDEPENDENT
    # component (late scratch after lineup confirmation); it is kept separate
    # because it takes out studs at the same rate as punts.
    # p_win exponent: >0 uses ONE exponent for every contest
    # (sharpness * flat_reference) instead of scaling each contest by its own
    # implied entry count. DEFAULT 0.0 (per-contest scaling) IS THE VALIDATED
    # SETTING. A flat exponent was briefly shipped 2026-07-30 and reverted: the
    # supporting evidence graded every entry against one archived contest's
    # field and payout curve. Re-graded per contest on REAL DK payout tables,
    # scaling wins -- flat costs 1.75% of $/entry (better on 1/8 slates,
    # p=0.0042) and cuts the chalk-to-small-fields ownership gradient from
    # +23.2 to +6.8. See pwin_exponents in external_pool.py.
    external_pool_pwin_flat_reference: float = 0.0

    external_pool_zero_inflate: bool = False
    external_pool_scratch_prob: float = 0.02

    # Empirical mean calibration for the SaberSim grids — the location fix,
    # kept separate from the zero-inflation shape fix above. Fitted 2026-07-30
    # over 10 archived slates (rostered players, usage-weighted, PPD excluded):
    # batters 0.878 (t=-3.32, p=0.009); pitchers 0.935 but p=0.30, i.e. not
    # distinguishable from 1.0, so pitchers default to no correction rather
    # than a fitted-on-noise haircut. Applied to the grids only, not to
    # players_df["mean"]. See _MEAN_CALIB_BATTER_SS in external_pool.py.
    external_pool_mean_calib_batter: float = 1.0
    external_pool_mean_calib_pitcher: float = 1.0


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
