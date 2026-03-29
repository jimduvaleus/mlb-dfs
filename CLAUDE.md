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

## Architecture

This is a modular numerical pipeline that transforms a DraftKings salary CSV and historical Retrosheet data into an optimized portfolio of DFS lineups. See `plans/mlb_dfs_architecture.md` for the full design.

### Data flow

```
DK Salary CSV  ──► DraftKingsSlateIngestor ──► players_df (player_id, position, salary, team, game, slot)
Retrosheet EVN ──► process_historical.py  ──► historical_logs.parquet (game_id, team_id, player_id, slot, dk_points)
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
                            BasinHoppingOptimizer.optimize()
                                        │
                                        ▼
                                   Lineup (10 player IDs)
```

### The copula unit

Players are grouped by `(team, opponent)` into 10-player "units": the 9 batters of one team plus their opposing pitcher. The empirical copula is a `G × 10` matrix of historical rank-quantiles (one row per historical game-team observation). Simulation bootstrap-samples rows to produce correlated joint quantile vectors.

### Marginal distributions

- **Pitchers (slot 10)**: Always `GaussianMarginal(mu, sigma)` — direct `scipy.stats.norm.ppf`.
- **Batters (slots 1-9), Phases 1-3**: `GaussianMarginal` used as a placeholder.
- **Batters (slots 1-9), Phase 4+**: `BatterMixtureMarginal` — mixture of Exp(λ) + N(μ,σ). At runtime, `BatterPCAModel.project(mu_proj, sigma_proj)` solves a 2×2 linear system to find the point on the fitted PCA plane that satisfies the projection constraints, yielding the full `(w, λ, μ, σ)` parameter set.

### Optimizer

`BasinHoppingOptimizer` runs `n_chains` (default 250) independent chains. Each chain: random valid lineup → perturbation (3 position-preserving swaps) → greedy local search (delta-update avoids recomputing full sums) → Metropolis acceptance. Objective: `P(lineup_total ≥ target)` estimated from the simulation matrix. The chains are parallelisable via `ProcessPoolExecutor` (`n_workers > 1`).

DraftKings Classic constraints enforced in `Lineup.is_valid()`: roster `{P×2, C, 1B, 2B, 3B, SS, OF×3}`, $50k salary cap, max 5 hitters from one team, min 2 different games.

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

### Implementation roadmap

Phases 1-4 are complete. Remaining:
- **Phase 5**: Portfolio construction — iterative greedy lineup selection that "consumes" simulation rows where the selected lineup already hits the target score.
- **Phase 6**: Performance tuning (Numba/Ray) and prop market projection integration.
