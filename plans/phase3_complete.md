# Phase 3 Complete: Basin-Hopping Optimizer

## Summary

Phase 3 (Basin-Hopping optimizer with salary/position constraints) is fully implemented and tested. All 39 unit tests pass (16 new + 23 from prior phases).

---

## Deliverables

### `src/ingestion/dk_slate.py` — `Player` dataclass + `DraftKingsSlateIngestor` (updated)
- Added `game: str = ""` field to `Player`.
- Ingestor now extracts a game ID from the "Game Info" column (e.g. `"LAD @ SD 03/20/2026 09:40PM ET"` → `"LAD@SD"`) and includes it in the returned DataFrame and `Player` objects.
- Fully backward-compatible: when "Game Info" is absent, `game` defaults to `""`.

### `src/optimization/lineup.py` — `Lineup`
- Dataclass holding `player_ids: List[int]` (exactly 10 players).
- `score(sim_matrix, col_map, target) -> float`: returns P(lineup total ≥ target) estimated over all simulation rows.
- `is_valid(player_meta: Dict[int, dict]) -> bool`: enforces all DraftKings Classic constraints:
  - Exact position counts: `{P: 2, C: 1, 1B: 1, 2B: 1, 3B: 1, SS: 1, OF: 3}`
  - Salary cap: $50,000
  - Max 5 hitters from any one team (pitchers excluded)
  - At least 2 different games (only checked when game info is present)

### `src/optimization/optimizer.py` — `BasinHoppingOptimizer` + `_ChainRunner`
- `BasinHoppingOptimizer`: public entry point. Constructor filters the player pool to those present in `SimulationResults`, pre-groups candidates by position, and delegates chain execution to `_ChainRunner`.
  - `optimize() -> Tuple[Lineup, float]`: runs `n_chains` independent chains, returns the best `(Lineup, score)` found.
  - Sequential by default (`n_workers=1`); parallel execution via `ProcessPoolExecutor` when `n_workers > 1`.
- `_ChainRunner`: self-contained, picklable class that runs one Basin-Hopping chain:
  1. **Initialization**: random valid lineup sampled by filling positions greedily.
  2. **Perturbation (mutation)**: 3 random position-preserving player swaps, retried until the result is valid.
  3. **Local search**: one greedy pass over all 10 slots. For each slot, the best improving single-player swap is accepted using a delta-update: `new_totals = totals − sim_matrix[:, col_out] + sim_matrix[:, col_in]` (avoids recomputing the full sum per candidate).
  4. **Metropolis acceptance**: always accept improvements; accept a worse result with probability `exp(Δ / T)`.
  5. Tracks and returns the best lineup seen across all steps.

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `target` | (required) | Score threshold for the objective function |
| `n_chains` | 250 | Independent Basin-Hopping chains |
| `temperature` | 0.1 | Metropolis temperature T |
| `n_steps` | 100 | Perturbation steps per chain |
| `n_workers` | 1 | Parallel worker processes (1 = sequential) |
| `rng_seed` | None | Base seed; chain i uses seed+i |

---

## Interface Contract for Phase 4

The optimizer's interface is stable. Phase 4 (mixture model + PCA marginals) only needs to replace how `SimulationResults` is produced — the `BasinHoppingOptimizer` signature is unchanged.

For reference, the optimizer consumes:

```python
sim_results.results_matrix   # np.ndarray, shape (n_sims, n_players)
sim_results.player_ids       # List[int], column order matches results_matrix

players_df                   # pd.DataFrame with columns:
                             #   player_id, position, salary, team, game
```

---

## Known Limitations / Phase 4 Inputs

- **Marginals are Gaussian only**: the `SimulationResults` fed to the optimizer still come from `GaussianMarginal`. Phase 4 replaces these with the mixture model + PCA pipeline for batters.
- **Parallelism is process-based**: each worker process receives a full copy of `sim_matrix`. For large simulation matrices, Phase 6 should migrate to shared-memory via Ray or `multiprocessing.shared_memory`.
- **No FLEX/DH slot**: the current roster template is fixed Classic format (`P×2 + C + 1B + 2B + 3B + SS + OF×3`). Slate formats with a FLEX or DH slot would require a roster requirements update.
