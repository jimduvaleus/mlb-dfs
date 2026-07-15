# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Target device

The web UI is **desktop-only**. No consideration needs to be given to mobile or tablet layouts.

## Commands

All commands must be run inside the virtualenv:

```bash
source venv/bin/activate
```

**Run all tests:**
```bash
python -m pytest tests/ --ignore=tests/test_simulation_pipeline.py
```

**Run a single test file:**
```bash
python -m pytest tests/test_batter_model.py -v
```

**Run a single test:**
```bash
python -m pytest tests/test_batter_model.py::TestBatterPCAModel::test_fit_returns_self -v
```

**Run the simulation pipeline integration test** (requires built copula first):
```bash
python scripts/build_copula.py
python tests/test_simulation_pipeline.py
```

### Data flow

```
DK Salary CSV  ──► DraftKingsSlateIngestor ──► players_df (player_id, position, salary, team, game, slot)
                                                        │
Slate/Player Exclusions ────────────────────────────────► prune players_df
                                                        │
Projections CSV ──┐                                     │
RotoWire API  ────┼──────────────────────────────────── ► players_df.mean / std_dev
Salary heuristic ─┘                                     │
                                                        ▼
Retrosheet EVN ──► process_historical.py  ──► historical_logs.parquet
                                                        │
                             ┌──────────────────────────┤
                             ▼                          ▼
                     build_copula.py           fit_batter_pca.py
                             │                          │
                             ▼                          ▼
              empirical_copula.parquet    batter_pca_model.npz + batter_score_grid.npy
                             │                          │
                             └──────────┬───────────────┘
                                        ▼
                              SimulationEngine.simulate(n_sims)
                                        │
                                        ▼
                              SimulationResults  (n_sims × n_players matrix)
                                        │
                                        ▼
                   CandidateGenerator.generate()  [n_candidates stacked lineups]
                                        │
                                        ▼
                   ContestScorer.score_candidates()  [K simulated fields × M candidates]
                                        │
                                        ▼
                   DeterminantPortfolioSelector.select()  [portfolio_size lineups]
                                        │
                                        ▼
                               Portfolio (n lineups)
                                        │
                          [Optional entry file workflow]
                                        ▼
                   parse_entry_file() → assign_lineups_to_entries()
                        → write_upload_files()  (upload_*.csv for DK)
```

### The copula unit

Players are grouped by `(team, opponent)` into 10-player "units": the 9 batters of one team plus their opposing pitcher. The empirical copula is a `G × 10` matrix of historical rank-quantiles (one row per historical game-team observation). Simulation samples at the **game level**: `EmpiricalCopula.sample_games()` bootstraps whole historical games and assigns the two paired rows to the game's two opposing units, preserving the correlation between a team's batters and its own pitcher (~+0.10 at the quantile level) and the shared run environment. Units without a partner on the slate (or copulas lacking a `(game_id, team_id)` index) fall back to independent row sampling via `sample()`.

Sampled rows then pass through a **sim-time dependence overlay** (`_apply_env_overlay` in `src/models/copula.py`): a shared per-unit Gaussian factor with calibrated loadings (`_ENV_OVERLAY_GAMMA` for batters, `_ENV_OVERLAY_DELTA` for the opposing pitcher) rotates the Gaussianized ranks so the joint dependence matches *realized residual* dependence rather than the rows' unconditional historical dependence — the raw rows double-count identity/matchup effects that the matchup-conditioned marginals already price (batter-batter +0.13 → target +0.19; batter-vs-opp-pitcher −0.335 → target −0.18). Marginals stay exactly uniform; the two rows of a game get independent factors (cross-team corr ≈ 0 is preserved). Calibrated by `scripts/diagnose_sim_tails.py` (PIT calibration of replayed sims against realized FPTS across the contest archive) — re-fit as the archive grows.

### Marginal distributions

Precedence per player (highest first):

1. **Market-implied quantile grid** (`EmpiricalQuantileMarginal`) — when the market-odds fetcher produced a validated 101-point percentile grid for the player (`data/processed/projections_mo_dist.parquet`, built by a per-player Monte Carlo over the fitted market rates with structural couplings: R ≥ HR, RBI ≥ HR, K ≤ outs, 5-IP win rule, early-exit outs mixture; batter event rates share a mean-one lognormal run-environment factor `_ENV_SIGMA`, calibrated against Retrosheet box-score co-movement by `scripts/calibrate_env_sigma.py`). Market-implied means carry an **empirical mean calibration** (`_MEAN_CALIB_BATTER`/`_MEAN_CALIB_PITCHER` in the fetcher, fitted 2026-07-06 against realized FPTS across the archive: batters ×0.867, pitchers ×0.946 — residual vig the fair-odds devigging doesn't remove). Batters are deflated at the *rate* level (mean scales linearly, spread √-style, couplings intact); pitchers post-hoc on the whole marginal. RotoWire-fallback players are not scaled. Archives from 2026-07-06 onward contain calibrated means — account for this when re-fitting. Grids are matched defensively: applied only when the player's projected mean agrees with the grid mean within ±0.5 (`src/models/quantile_grids.py`), so fallback-sourced or stale entries keep the parametric path.
2. **Batters (slots 1-9)**: `BatterMixtureMarginal` — mixture of Exp(λ) + N(μ,σ). At runtime, `BatterPCAModel.project(mu_proj, sigma_proj)` solves a 2×2 linear system to find the point on the fitted PCA plane that satisfies the projection constraints, yielding the full `(w, λ, μ, σ)` parameter set.
3. **Fallback (pitchers without a grid; batters without a PCA model)**: `GaussianMarginal(mu, sigma)` — direct `scipy.stats.norm.ppf`.

`scripts/compare_marginals.py` reports before/after moment and percentile shifts (synthetic archetypes, or per-player when a dist parquet is present).

### Portfolio construction

Three stages in `src/optimization/`:

- **`CandidateGenerator`** (`candidate_generator.py`) — generates a large pool of stacked candidate lineups (default 20k). Uses ownership-weighted sampling to produce correlated batter stacks with matching pitchers. **Sim-winner seeding** (`generate_sim_winners`, behind `gpp.seed_sim_winners` + `gpp.n_sim_winner_worlds/sim_winner_per_world/sim_winner_temp/sim_winner_own_blend`): for each of many stratified sim worlds, draws lineups through the same `_sample_one` machinery with per-world rank-softmax weights over that world's realized scores — the scaled, sampling-based successor to per-sim exact ILP seeding (`seed_sim_optimal_lineups`). Dumped as `seed_source="sim_winner"`. **Shape-preserving seed mutation** (`generate_shape_mutants`, behind `gpp.seed_mutants_per_parent` + `gpp.seed_mutant_salary_locality/seed_mutant_pitcher_weight`): expands each seed parent (sim_optimal + sim_winner) with N mutants whose team-stack profile matches the parent *exactly* — batter slots swap only within the same team (swap-in must cover the outgoing primary position), pitcher slots swap conflict-checked against the rostered hitters; replacement sampling is salary-local. Built for seed-first pools (round 6): `refine_rounds: 0` gives the best pool but starves the Det selector of neighborhood diversity, and the generic `generate_mutants` refinement mutants restore it only by breaking seed shape. Additive on top of `n_candidates`, dumped as `seed_source="seed_mutant"`.
- **`ContestScorer`** (`gpp_portfolio.py`) — scores each candidate against K simulated opponent fields (drawn from `ContestSimulator`), producing a `robust_payout` matrix (M candidates × n_sims). A Numba JIT kernel computes per-lineup payouts against the sorted field with exact tie splitting (two binary searches bound the tie band; payout = prefix-sum mean over it). Optional duplicate-entry penalty (`gpp.dupe_penalty`): a log-linear model (Σ log ownership, unused salary, primary stack size — coefficient knobs in `gpp.dupe_*`, fitted by `scripts/fit_dupe_model.py` as a zero-truncated Poisson GLM on the archived contest standings; re-fit as the archive grows) estimates E[dupes] per candidate and dilutes top payout bands (gross ≥ `gpp.dupe_min_gross_payout`) by 1/(1+E[dupes]).
- **Fresh re-score** (`ContestScorer.rescore_fresh_fields`, wired in `pipeline.py`) — after refinement, every candidate with mined EV ≥ `gpp.ev_floor` is re-scored against `gpp.final_n_field_samples` freshly drawn fields (disjoint seeds), then candidates whose **fresh** EV falls below the floor are discarded — a lineup must clear the floor twice to reach the selector. This kills the EV winner's curse from mining on the first-stage fields while keeping the low-EV/high-diversity band above the floor available to the diversity-weighted selector (an EV-rank cutoff here would starve Det-EV of diverse picks). `gpp.final_rescore_top` is only a safety cap on the slice size (memory/time guard — a warning logs when it binds). Set `final_n_field_samples: 0` to disable. **Tail bypass**: the top `gpp.tail_bypass_n` *below-floor* candidates by per-candidate sim-p99 (`_candidate_sim_tail_scores` — a ceiling statistic that mean-dollar EV systematically undervalues) are also admitted to the fresh re-score; they must keep fresh EV ≥ `gpp.tail_bypass_ev_floor` to survive, and the selector's backstop cull is lowered to that floor when any survive. Bypass candidates are marked `tail_bypass=1` in the candidate-pool dump for post-contest evaluation. Set `tail_bypass_n: 0` to disable.
- **Tail metrics + funnel modes (ceiling-first redesign)** — with `gpp.compute_tail_metrics` on, `ContestScorer` also accumulates per-candidate **tail_ev** (expected gross dollars from payout ranks paying ≥ `gpp.tail_ev_min_gross` only — the steep band, blind to the min-cash plateau that dominates mean EV) and **p_beat99** (fraction of sims beating the simulated field's p99), in both mined and fresh flavors; all land in the pool dump (`sim_p99, tail_ev, p_beat99, fresh_tail_ev, fresh_p_beat99`). `gpp.funnel_mode`: `ev_first` (EV floor primary; tail lane = `tail_bypass_n` side door) or `tail_first` (top `gpp.tail_admit_n` by `gpp.tail_metric` admitted to the fresh re-score, held only to `gpp.ev_guardrail`; the EV-floor lane persists as the cash-anchor source). `gpp.tail_metric` picks the lane currency: `tail_ev` / `p_beat99` / `sim_p99`.
- **`DeterminantPortfolioSelector`** (`gpp_portfolio.py`) — greedy portfolio assembly via incremental determinant maximization. Balances EV and lineup diversity by selecting each successive lineup to maximally expand the payout covariance structure. `gpp.selector_score: tail` makes the EV term rank by the fresh tail currency (`ev_override`), with the first `ceil(gpp.cash_anchor_fraction × size)` picks still on mean dollar EV; the floor cull, diversity term, and reported EVs stay on mean dollar EV.

### Offline replay harness

`scripts/replay_slate.py` re-runs the full production pipeline on an archived slate's inputs (`archive/MMDDYYYY/DKSalaries.csv` + `market_odds_projections.csv`; team totals resolve automatically from the slate date; live exclusions/twitter scratches are inherited via the DKSalaries fingerprint) and grades every funnel stage — pool, floor set, fresh survivors, seed blocks, per-risk portfolios, arbitrary `--rank-col` slices — against the real contest field from the standings zip (hit99/hit99.9, cash rate, realized net $, per-column Spearman). Batch mode: `--recent N` / `--all` × `--variants <yaml>` (pre-registered matrices — see `outputs/replay/variants_phase123.yaml`), results appended to `outputs/replay/replay_summary.csv`. `--screen` = 10k sims/10k candidates coarse profile. Sims are cached per (slate, n_sims, seed, projections-hash) in `outputs/replay/sim_cache/` so config variants reuse one sim; `PipelineRunner(persist_caches=False, sim_cache_path=...)` are the replay-only hooks (never set by the live server). Pre-2026-07-06 archives get the mean calibration applied at load (`--no-precalib` disables). Fidelity gate (2026-07-12, 4 post-overlay slates): replayed pool hit99 pooled 1.05% vs live 0.98% — judge variants by deltas.

`scripts/fit_winner_shape.py` + `src/optimization/winner_shape.py` — sim-free logistic on real top-1% entry composition (walk-forward artifact `data/processed/winner_shape_model.json`). Measured weak (mean walk-forward AUC 0.554, 2026-07-12): composition alone barely discriminates the real ceiling; kept for reference, not wired as a funnel currency.

### Payout functions

`src/optimization/payout.py` provides `load_payout_structure()` and `payout_table_to_array()` for loading the reference GPP payout structure from `data/payout_structures/dk_classic_gpp.json`. Used by `ContestScorer` and `ContestSimulator` to compute per-rank net payouts.

### Entry file workflow

`src/api/dk_entries.py` supports the DraftKings upload workflow:
1. Parse an existing DK entry CSV with `parse_entry_file()` → list of `EntryRecord`.
2. Assign optimized lineups to entries with `assign_lineups_to_entries()`.
3. Write DK-ready upload CSVs with `write_upload_files()` → `outputs/upload_*.csv`.

### Slate and player exclusions

`src/api/slate_exclusions.py` manages persistent exclusions stored in `data/slate_exclusions.json`. Excluded games/teams and players are pruned from `players_df` before simulation. Server endpoints: `POST /api/slate/exclusions` and `POST /api/slate/player-exclusions`.

### Key data contracts

`players_df` passed to `SimulationEngine` and `BasinHoppingOptimizer` must have columns:
`player_id, team, opponent, slot, mean, std_dev, position, salary, game`

`SimulationResults` core properties: `results_matrix` (ndarray, shape `n_sims × n_players`), `player_ids` (list, column order matches matrix).

### Precomputed vs. runtime artifacts

| Artifact | Script | Used by |
|---|---|---|
| `data/processed/historical_logs.parquet` | `scripts/process_historical.py` | `build_copula.py`, `fit_batter_pca.py` |
| `data/processed/empirical_copula.parquet` | `scripts/build_copula.py` | `SimulationEngine` |
| `data/processed/batter_pca_model.npz` | `scripts/fit_batter_pca.py` | `SimulationEngine` |
| `data/processed/batter_score_grid.npy` | `scripts/fit_batter_pca.py` | `SimulationEngine` |
| `data/processed/ownership_calibrator.json` | `scripts/fit_ownership_calibrator.py` | pipeline ownership calibration (`load_ownership_calibrator`); stale-checked against the `ownership.py` constants hash — re-fit after any constants change |
| `data/slate_exclusions.json` | runtime (API) | `slate_exclusions.py` |
| `data/payout_structures/dk_classic_gpp.json` | manual / reference | `payout.py` (calibration) |

### Web UI and API

The web app is a React + TypeScript + Vite frontend (`ui/`) backed by a FastAPI server (`src/api/server.py`).

**Key API endpoints:**
- `GET /api/config` / `POST /api/config` — read/write `config.yaml`
- `GET /api/projections/fetch` — stream projections fetch output (SSE); `GET /api/projections/status` — freshness check
- `GET /api/projections/players` — current player projections; `GET /api/projections/team_totals`, `GET /api/projections/unconfirmed`, `GET /api/projections/merge_info`
- `GET /api/run/stream` — SSE stream of pipeline progress events
- `GET /api/run/status` / `POST /api/run/stop` — run state management
- `GET /api/portfolio` — last completed portfolio (persisted across restarts)
- `GET /api/portfolio/sweep` — risk-sweep results; `POST /api/portfolio/activate_risk` — select a risk tier; `POST /api/portfolio/replace/{lineup_index}` — swap one lineup
- `GET /api/contest/analyze` — contest analysis / field simulation
- `GET /api/slate/games` / `POST /api/slate/exclusions` — game exclusions
- `GET /api/slate/players` / `POST /api/slate/player-exclusions` — player exclusions
- `GET /api/slate/ownership-reductions` / `POST /api/slate/ownership-reductions` — per-player ownership fade
- `GET /api/slate/projection-overrides` / `POST /api/slate/projection-overrides` — per-player mean overrides (`dict[int, float]`; no std override exists). Applied to `players_df["mean"]` in `PipelineRunner._apply_exclusions()` before exclusion filtering, so both `sim_players_df` and `cand_players_df` (and therefore the optimal-lineup ILP solver) see the overridden mean.
- `GET /api/notifications` / `DELETE /api/notifications/{id}` — in-app notification log
- `GET /api/twitter-lineups` / `POST /api/twitter-lineups` / `DELETE /api/twitter-lineups/{team}` — confirmed opponent lineup tracking
- `POST /api/lineups/{team}/lock` / `DELETE /api/lineups/{team}/lock` — lock/unlock a lineup
- `GET /api/late-swap/state` / `POST /api/late-swap/run` / `POST /api/late-swap/override` / `POST /api/late-swap/reset` — late-swap workflow

**Pipeline progress events** emitted via SSE: `load_slate`, `simulate`, `compute_target`, `gpp_generate_start/progress/done`, `gpp_score_start/done`, `gpp_optimal_start/progress/done`, `gpp_sim_optimal_start/progress/done`, `gpp_seed_mutant_start/progress/done`, `gpp_refine_start/progress/done`, `gpp_rescore_start/field_progress/score_progress/done`, `portfolio_stats`, `complete`. The rescore field/score progress events are named distinctly from the first-stage `gpp_field_progress`/`gpp_score_progress` (same payload shape, different stage string) so the UI can show a live Fresh re-score readout instead of a stale first-stage label held over in `ProgressPanel.tsx`.

**UI components** (`ui/src/components/`): `ConfigForm`, `SlatePanel`, `ProjectionsPanel`, `ProjectionsTable`, `ProgressPanel`, `PortfolioTable`, `MetricsPanel`, `TeamBadge`, `StopUploadDialog`, `LateSwapPanel`, `LineupParserDialog`, `RunOptionsDialog`, `DeleteConfirmModal`.

Team logos for all 30 MLB franchises are in `ui/public/team-logos/`.

**Adding new config fields:** any field exposed in the UI (`ConfigForm.tsx` / `ui/src/types.ts` `GppConfig`) must also be declared in `src/api/models.py` `GppConfig`. Pydantic silently drops unknown fields on `POST /api/config`, so omitting a field there means Save Config is a no-op for that field. The three places that must stay in sync: `config.yaml` (default value), `src/api/models.py` `GppConfig` (Pydantic model + default), `ui/src/types.ts` `GppConfig` (TypeScript type).

## DK Classic GPP payout structure

`data/payout_structures/dk_classic_gpp.json` models a typical DK $4 Classic GPP: 14,863 entries, 3,855 paying positions (~26% cash rate). The cash cutoff is finishing in roughly the **top 26%**, which requires beating ~74% of the field. `ContestSimulator.eval_portfolio()` reads this file via `PipelineRunner._load_cash_threshold()` to derive the `cash_threshold` used in `cash_rate` computation — do not use `> 0.5` (beats half the field) as a cash proxy.

## CrazyNinjaOdds market names (confirmed)

The market dropdown option labels in `fetch_market_odds_projections.py` have been verified against the live site's `<select>` element. **Do not change these strings** — they are confirmed correct:

- `"Player Batting Walks"` (confirmed 2026-04-26; the dropdown also contains `"Player Batting Strikeouts"` as a distinct adjacent entry — do not conflate or rename either)

If a market appears to fail silently, investigate AJAX timing or sub-market ("All players") selection before assuming the label is wrong.