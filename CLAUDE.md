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
                            BasinHoppingOptimizer.optimize()  [per lineup]
                                        │
                                        ▼
                   PortfolioConstructor  OR  BeamPortfolioConstructor
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

### Marginal distributions

- **Pitchers (slot 10)**: Always `GaussianMarginal(mu, sigma)` — direct `scipy.stats.norm.ppf`.
- **Batters (slots 1-9), Phases 1-3**: `GaussianMarginal` used as a placeholder.
- **Batters (slots 1-9), Phase 4+**: `BatterMixtureMarginal` — mixture of Exp(λ) + N(μ,σ). At runtime, `BatterPCAModel.project(mu_proj, sigma_proj)` solves a 2×2 linear system to find the point on the fitted PCA plane that satisfies the projection constraints, yielding the full `(w, λ, μ, σ)` parameter set.

### Optimizer

`BasinHoppingOptimizer` runs `n_chains` (default 250) independent chains. Each chain: random valid lineup → perturbation (3 position-preserving swaps) → greedy local search (delta-update avoids recomputing full sums) → Metropolis acceptance. The chains run in parallel via `ProcessPoolExecutor` (`n_workers > 1`).

**Objective functions** (set via `optimizer.objective` in `config.yaml`):
- `p_hit` — `P(lineup_total ≥ target)` estimated from the simulation matrix.
- `expected_surplus` — expected score above target for lineups that hit.
- `marginal_payout` — `E[P(max(best_scores, lineup_scores))]` where `P(s) = max(0, s - target)^beta`. Maximizes marginal expected payout improvement given prior portfolio lineups' best scores. Controlled by `optimizer.payout_beta` (default 2.5).

**Additional constraints:**
- `early_stopping_window` / `early_stopping_threshold` — cross-chain early stopping when improvement stalls.

DraftKings Classic constraints enforced in `Lineup.is_valid()`: roster `{P×2, C, 1B, 2B, 3B, SS, OF×3}`, $50k salary cap, max 5 hitters from one team, min 2 different games.

### Portfolio construction

Two implementations in `src/optimization/portfolio.py`:

- **`PortfolioConstructor`** — Iterative greedy: each round optimizes a new lineup over remaining active simulation rows (rows where no prior lineup already hit target). Fast but can lock in on correlated first lineups. When objective is `marginal_payout`, switches to payout-weighted mode: tracks `best_scores` (per-sim best lineup score) instead of binary row consumption, and passes these to the optimizer so it maximizes marginal payout improvement.
- **`BeamPortfolioConstructor`** — Beam search: maintains `beam_width` (default 3) candidate portfolio paths in parallel, pruning by coverage (fewest active rows remaining). Mitigates greedy lock-in at the cost of extra optimization rounds.

### Payout functions

`src/optimization/payout.py` provides a power-law payout function `P(s) = max(0, s - cash_line)^beta` used by the `marginal_payout` objective. A reference GPP payout structure is stored in `data/payout_structures/dk_classic_gpp.json`. The `calibrate_beta()` function can fit beta to an actual payout table given simulated score percentiles.

### Entry file workflow

`src/api/dk_entries.py` supports the DraftKings upload workflow:
1. Parse an existing DK entry CSV with `parse_entry_file()` → list of `EntryRecord`.
2. Assign optimized lineups to entries with `assign_lineups_to_entries()`.
3. Write DK-ready upload CSVs with `write_upload_files()` → `outputs/upload_*.csv`.

### Slate and player exclusions

`src/api/slate_exclusions.py` manages persistent exclusions stored in `data/slate_exclusions.json`. Excluded games/teams and players are pruned from `players_df` before simulation. Server endpoints: `POST /api/slate/games` and `POST /api/slate/players`.

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
| `data/slate_exclusions.json` | runtime (API) | `slate_exclusions.py` |
| `data/payout_structures/dk_classic_gpp.json` | manual / reference | `payout.py` (calibration) |

### Web UI and API

The web app is a React + TypeScript + Vite frontend (`ui/`) backed by a FastAPI server (`src/api/server.py`).

**Key API endpoints:**
- `GET /api/config` / `PUT /api/config` — read/write `config.yaml`
- `POST /api/projections/fetch` — fetch RotoWire projections; `GET /api/projections/status` — freshness check
- `GET /api/run/stream` — SSE stream of pipeline progress events
- `GET /api/run/status` / `POST /api/run/stop` — run state management
- `GET /api/portfolio` — last completed portfolio (persisted across restarts)
- `GET /api/slate/games` / `POST /api/slate/games` — game exclusions
- `GET /api/slate/players` / `POST /api/slate/players` — player exclusions

**Pipeline progress events** emitted via SSE: `slate_loaded`, `projections_loaded`, `simulation_complete`, `portfolio_complete`. Run tab shows live n_sims countdown during simulation.

**UI components** (`ui/src/components/`): `ConfigForm`, `SlatePanel`, `ProjectionsPanel`, `ProgressPanel`, `PortfolioTable`, `MetricsPanel`, `TeamBadge`, `StopUploadDialog`.

Team logos for all 30 MLB franchises are in `ui/public/team-logos/`.

## DK Classic GPP payout structure

`data/payout_structures/dk_classic_gpp.json` models a typical DK $4 Classic GPP: 14,863 entries, 3,855 paying positions (~26% cash rate). The cash cutoff is finishing in roughly the **top 26%**, which requires beating ~74% of the field. `ContestSimulator.eval_portfolio()` reads this file via `PipelineRunner._load_cash_threshold()` to derive the `cash_threshold` used in `cash_rate` computation — do not use `> 0.5` (beats half the field) as a cash proxy.

## CrazyNinjaOdds market names (confirmed)

The market dropdown option labels in `fetch_market_odds_projections.py` have been verified against the live site's `<select>` element. **Do not change these strings** — they are confirmed correct:

- `"Player Batting Walks"` (confirmed 2026-04-26; the dropdown also contains `"Player Batting Strikeouts"` as a distinct adjacent entry — do not conflate or rename either)

If a market appears to fail silently, investigate AJAX timing or sub-market ("All players") selection before assuming the label is wrong.