# Phase 5 Complete: Portfolio Construction

## Summary

Phase 5 (iterative greedy lineup selection with simulation row consumption) is fully implemented and tested. All 82 unit tests pass (11 new + 71 from prior phases).

---

## Deliverables

### `src/optimization/portfolio.py` — `PortfolioConstructor` (new)

**Constructor parameters**

| Parameter | Default | Description |
|---|---|---|
| `sim_results` | — | `SimulationResults` from `SimulationEngine.simulate()` |
| `players_df` | — | Player pool with columns: `player_id, position, salary, team, game` |
| `target` | — | DraftKings score threshold (same value used by the optimizer) |
| `portfolio_size` | — | Number of lineups to construct |
| `n_chains` | `250` | Basin-hopping chains per optimization round |
| `temperature` | `0.1` | Metropolis acceptance temperature |
| `n_steps` | `100` | Perturbation steps per chain |
| `n_workers` | `1` | Worker processes for parallel chains |
| `rng_seed` | `None` | Base seed; seed for round `i` is `rng_seed + i` |

**`construct() → List[Tuple[Lineup, float]]`**

Returns one `(lineup, score)` pair per selected lineup in selection order. `score` is `P(lineup_total ≥ target)` computed over the *full* simulation matrix so values are directly comparable across all lineups in the portfolio.

**Algorithm**

```
active_mask = all True  (shape: n_sims)

for i in range(portfolio_size):
    if active_mask is all False: stop early

    active_sim = SimulationResults(full_matrix[active_mask])
    lineup, _ = BasinHoppingOptimizer(active_sim, ...).optimize()

    # Score against full matrix for comparability
    full_score = P(full_matrix[:, lineup_cols].sum(axis=1) >= target)

    # Consume active rows where this lineup already hits the target
    active_mask[active_rows where lineup_total >= target] = False

    portfolio.append((lineup, full_score))

return portfolio
```

### `src/optimization/__init__.py` — updated to export `PortfolioConstructor`

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Active row mask on the full matrix | Avoids copying large arrays; indexing with a boolean mask is a cheap NumPy view |
| Score reported against the full matrix | Scores across rounds are directly comparable; using the active-subset score would inflate later lineups' apparent value |
| Seed per round = `base_seed + i` | Ensures each round is deterministic and independent while the overall sequence is reproducible |
| No changes to existing classes | `SimulationResults` and `BasinHoppingOptimizer` are consumed unchanged; Phase 5 is purely additive |
| Early stop when rows exhausted | Prevents a deadlock where the optimizer has zero simulation rows; portfolio may be smaller than `portfolio_size` when this occurs |

---

## Interface Contract for Phase 6

`PortfolioConstructor` is the terminal consumer of the pipeline. Phase 6 refinements plug into earlier stages:

- **Performance (Numba/Ray)**: accelerate `BasinHoppingOptimizer` internals; `PortfolioConstructor` calls `optimizer.optimize()` and is unaffected
- **Prop market projections**: swap the `ProjectionProvider` that populates `players_df.mean` / `players_df.std_dev`; `PortfolioConstructor` is unaffected

Usage example:

```python
from src.optimization.portfolio import PortfolioConstructor

pc = PortfolioConstructor(
    sim_results=sim_results,        # SimulationResults from SimulationEngine
    players_df=players_df,          # player_id, position, salary, team, game, ...
    target=150.0,
    portfolio_size=20,
    n_chains=250,
    n_workers=4,                    # parallel chains
    rng_seed=0,
)
portfolio = pc.construct()
# [(Lineup([...]), 0.312), (Lineup([...]), 0.287), ...]
```

---

## Known Limitations / Phase 6 Inputs

- **No overlap constraint**: the same player can appear in multiple lineups. A future enhancement could cap how many times a player appears across the portfolio.
- **Single-pass greedy**: lineups are selected in order; there is no backtracking. A beam-search or simulated-annealing variant over the full portfolio could improve aggregate expected value.
- **Row consumption is binary**: a row is either active or consumed. A soft-weighting scheme (down-weight rather than remove) could smooth out edge effects when the target is rarely hit.
