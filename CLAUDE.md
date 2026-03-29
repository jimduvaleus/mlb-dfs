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

**Build the empirical copula** (uses dummy data if `historical_logs.parquet` is absent):
```bash
python scripts/build_copula.py
```

**Fit the batter PCA model** (requires real `historical_logs.parquet` from Retrosheet):
```bash
python scripts/fit_batter_pca.py --min-games 30
```

**Process Retrosheet data into `historical_logs.parquet`:**
```bash
python scripts/process_historical.py
```

**Start the web UI** (FastAPI backend + React dev server):
```bash
./scripts/start_ui.sh
```

**Stop the web UI:**
```bash
./scripts/stop_ui.sh
```

## Architecture

This is a modular numerical pipeline that transforms a DraftKings salary CSV and historical Retrosheet data into an optimized portfolio of DFS lineups. See `plans/mlb_dfs_architecture.md` for the full design.

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

Players are grouped by `(team, opponent)` into 10-player "units": the 9 batters of one team plus their opposing pitcher. The empirical copula is a `G × 10` matrix of historical rank-quantiles (one row per historical game-team observation). Simulation bootstrap-samples rows to produce correlated joint quantile vectors.

### Marginal distributions

- **Pitchers (slot 10)**: Always `GaussianMarginal(mu, sigma)` — direct `scipy.stats.norm.ppf`.
- **Batters (slots 1-9), Phases 1-3**: `GaussianMarginal` used as a placeholder.
- **Batters (slots 1-9), Phase 4+**: `BatterMixtureMarginal` — mixture of Exp(λ) + N(μ,σ). At runtime, `BatterPCAModel.project(mu_proj, sigma_proj)` solves a 2×2 linear system to find the point on the fitted PCA plane that satisfies the projection constraints, yielding the full `(w, λ, μ, σ)` parameter set.

### Optimizer

`BasinHoppingOptimizer` runs `n_chains` (default 250) independent chains. Each chain: random valid lineup → perturbation (3 position-preserving swaps) → greedy local search (delta-update avoids recomputing full sums) → Metropolis acceptance. The chains run in parallel via `ProcessPoolExecutor` (`n_workers > 1`).

**Objective functions** (set via `optimizer.objective` in `config.yaml`):
- `p_hit` — `P(lineup_total ≥ target)` estimated from the simulation matrix.
- `expected_surplus` — expected score above target for lineups that hit.

**Additional constraints:**
- `salary_floor` — minimum total salary (default 47500).
- `early_stopping_window` / `early_stopping_threshold` — cross-chain early stopping when improvement stalls.

DraftKings Classic constraints enforced in `Lineup.is_valid()`: roster `{P×2, C, 1B, 2B, 3B, SS, OF×3}`, $50k salary cap, max 5 hitters from one team, min 2 different games.

### Portfolio construction

Two implementations in `src/optimization/portfolio.py`:

- **`PortfolioConstructor`** — Iterative greedy: each round optimizes a new lineup over remaining active simulation rows (rows where no prior lineup already hit target). Fast but can lock in on correlated first lineups.
- **`BeamPortfolioConstructor`** — Beam search: maintains `beam_width` (default 3) candidate portfolio paths in parallel, pruning by coverage (fewest active rows remaining). Mitigates greedy lock-in at the cost of extra optimization rounds.

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

### Implementation roadmap

Phases 1-5 are complete. Remaining:
- **Phase 6**: Performance tuning (Numba/Ray) and prop market projection integration. Other candidates: weather integration, late-swap handling, additional projection sources (Fangraphs, prediction markets).
