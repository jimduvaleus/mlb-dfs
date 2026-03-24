# Phase 2 Complete: Simulation Engine

## Summary

Phase 2 (Empirical Copula sampling and Monte Carlo scoring) is fully implemented and tested. All 23 unit tests pass and the end-to-end pipeline integration test passes.

---

## Deliverables

### `scripts/build_copula.py`
Builds the empirical copula from `data/processed/historical_logs.parquet`.
- Computes per-slot quantiles (slots 1–9 for batters, slot 10 globally for opposing pitchers) using `rank(pct=True)`.
- Pivots to a `G × 10` matrix indexed by `(game_id, team_id)`.
- Drops incomplete rows (any missing slot), logs the count.
- Saves to `data/processed/empirical_copula.parquet`.

### `src/models/copula.py` — `EmpiricalCopula`
- Loads the `G × 10` quantile matrix from parquet on init.
- `sample(n_sims)`: bootstrap-samples `n_sims` rows with replacement, returning an `(n_sims, 10)` NumPy array.
- Accepts an optional `context_filter` hook for future stratification (e.g. venue, weather).

### `src/models/marginals.py` — `GaussianMarginal`
- Wraps `scipy.stats.norm.ppf` with input validation and quantile clipping (avoids ±∞ at 0/1).
- Interface: `ppf(q: np.ndarray) -> np.ndarray`.

### `src/simulation/engine.py` — `SimulationEngine`
- Groups slate players by `(team, opponent)` to form 10-player copula units.
- For each unit, samples joint quantile vectors from the copula, then applies each player's `GaussianMarginal.ppf` at their batting slot index.
- Floors simulated points at 0 (DFS scores are non-negative).
- Returns a `SimulationResults` container.

### `src/simulation/results.py` — `SimulationResults`
- Stores the `(n_sims, n_players)` NumPy matrix and the ordered `player_ids` list.
- `get_player_stats()`: returns mean, std, min, max, p25, p75, p99 per player.
- `save_to_parquet(path)` / `to_dataframe()`: long-form persistence for auditing or re-running the optimizer without re-simulating.

---

## Bug Fixes Applied

| File | Bug | Fix |
|---|---|---|
| `src/ingestion/retrosheet_parser.py` | Used `df['K']` for strikeouts; `cwbox` outputs column `SO` | Changed to `df['SO']` |
| `tests/test_ingestion.py` | Pitching fixtures used `'K'` key, masking the above bug | Updated to `'SO'` |
| `src/simulation/__init__.py` | File missing; inconsistent with all other `src/` subpackages | Created empty `__init__.py` |

---

## Interface Contract for Phase 3

The optimizer receives a `SimulationResults` object. The key properties it needs:

```python
results.results_matrix   # np.ndarray, shape (n_sims, n_players)
results.player_ids       # List[int], column order matches results_matrix
results.n_sims           # int
results.n_players        # int
```

The player metadata (salary, position, team) lives in the `players_df` passed to `SimulationEngine`, which is derived from the DK slate via `DraftKingsSlateIngestor` (Phase 1). The optimizer will need both `results` and `players_df` to enforce salary/position constraints.

---

## Known Limitations / Phase 3 Inputs

- **Slot assignment**: `players_df` requires a `slot` column (batting order 1–9 or 10 for pitchers). In production this must be sourced from a lineup card or projection service; it is not present in the DK salary CSV.
- **Marginals are Gaussian only**: Phase 4 will replace these with the mixture model + PCA pipeline for batters.
- **Copula is unconditional**: The `context_filter` hook in `EmpiricalCopula.sample()` is a stub. Stratification by venue, park factor, or game total is a Phase 6 refinement.
