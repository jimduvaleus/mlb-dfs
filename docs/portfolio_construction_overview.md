# MLB DFS Portfolio Construction: How It Works

This document explains how the application builds a portfolio of DraftKings lineups from player projections and historical data, covering the simulation, player modeling, optimization, and diversity mechanics end-to-end.

---

## 1. The Simulation Matrix: What `n_sims` Means

Before any lineup is selected, the system runs a Monte Carlo simulation to generate a full picture of possible game outcomes. The result is an `n_sims × n_players` matrix — each row is one plausible "universe" of DK scores across all slated players.

### How a single simulation row is constructed

Players are grouped into 10-player **units**: the 9 batters of a team plus the opposing pitcher. For each unit, one row is sampled from the **empirical copula** — a precomputed `G × 10` table of historical rank-quantiles derived from Retrosheet game logs. That sampled row is a vector of 10 quantile values (each between 0 and 1), one per batting-order slot / pitcher slot.

Each quantile is then transformed into an actual DK score by passing it through the player's **marginal distribution** (the inverse CDF, or PPF):

- **Pitchers**: Gaussian marginal — `N(μ, σ)` with projections as parameters.
- **Batters (Phases 1–3)**: Also Gaussian, used as a placeholder.
- **Batters (Phase 4+)**: A mixture marginal — `w · Exp(λ) + (1-w) · N(μ, σ)`. The mixture parameters `(w, λ, μ, σ)` are resolved from the player's projection `(mean, std_dev)` via the `BatterPCAModel`, which finds the point on a fitted PCA manifold in 4-D parameter space that satisfies the projection constraints.

Because all players in a unit share the same bootstrapped copula row, their scores are **correlated** the way they were historically — a game where a team's cleanup hitter had a great game tends to be one where the rest of the lineup also did well.

### What `n_sims` controls

`n_sims` is the number of complete game universes generated. A larger `n_sims`:

- Provides a more stable estimate of the probability that any given lineup hits the target score — the optimizer's objective is `P(lineup_total ≥ target)`, estimated directly from this matrix.
- Gives the portfolio construction algorithm more resolution when identifying which simulation "scenarios" a lineup covers and which remain uncovered.
- Increases memory usage and run time roughly linearly.

In practice, `n_sims` is the primary fidelity knob. Reducing it speeds up the pipeline at the cost of noisier probability estimates and coarser coverage tracking in portfolio construction.

---

## 2. How Player Projections Are Factored In

Each player enters the simulation with two projection values: `mean` and `std_dev`. These come from an external source (e.g., a projections CSV or a provider like RotoWire) and represent the expected DK score and its uncertainty.

Projections shape the **marginal distribution** for each player, not the correlation structure:

- For **pitchers** and simple batter models, `(mean, std_dev)` directly parameterize a Gaussian: `N(mean, std_dev)`.
- For **batters with the Phase 4 model**, `(mean, std_dev)` are used as projection constraints. The `BatterPCAModel.project(mean, std_dev)` solves a 2×2 linear system to find which point on the PCA plane in `(w, λ, μ, σ)` space matches those constraints. This yields a richer, empirically-grounded shape — capturing the heavy right tail of DK batter scores — while still honoring the analyst's projection.

The copula determines **who moves together**; the marginal (parameterized by the projection) determines **how high or low** that player's score lands when they do move.

---

## 3. How the Optimizer Selects Lineups

Each lineup is found by the **Basin-Hopping Optimizer** (`BasinHoppingOptimizer`). It maximizes a single objective: `P(lineup_total ≥ target)`, estimated by summing the lineup's column scores across all active simulation rows and checking what fraction clear the target.

### The search process (per chain)

1. **Start**: Draw a random valid lineup (satisfying DK's position requirements, $50k salary cap, max 5 hitters from one team, min 2 games represented).
2. **Mutate**: Replace 3 randomly chosen players with position-compatible alternatives.
3. **Local search**: For each slot in random order, evaluate all valid single-player swaps using a delta-update — the new total is `old_total - old_player_score + new_player_score` — and accept the best improving swap.
4. **Metropolis acceptance**: Accept the mutated+searched candidate if it improved, or with probability `exp(Δ / T)` if it didn't (where `T` is the temperature parameter). This allows occasional downhill moves to escape local optima.
5. **Track best**: Record the best lineup seen across all steps; stop early if no improvement for `niter_success` consecutive steps.

### Multiple chains

The optimizer runs `n_chains` (default 250) independent chains from different random starting points. Running chains in parallel via `ProcessPoolExecutor` is supported. The best result across all chains is returned.

### DraftKings constraints

`Lineup.is_valid()` enforces the full DK Classic roster: `{P×2, C, 1B, 2B, 3B, SS, OF×3}`, $50k cap, ≤5 hitters from one team, ≥2 distinct games.

---

## 4. How Portfolio Diversity Is Achieved

Diversity is not enforced via explicit constraints (like exposure caps). Instead, it emerges naturally from the **simulation row consumption** mechanic built into the portfolio construction loop.

### The core algorithm (`PortfolioConstructor`)

1. Start with all `n_sims` rows **active**.
2. Run the optimizer on the active rows to find the best lineup `L₁`.
3. **Consume** all active rows where `L₁`'s total score already clears the target. These are the simulation scenarios that `L₁` "covers."
4. Repeat on the remaining active rows to find `L₂`, `L₃`, … until the desired portfolio size is reached.

Because later lineups are optimized over scenarios that earlier lineups *did not* cover, they are forced to find upside through different player combinations — different teams, different game stacks, different salary allocations. Lineups that would simply replicate the same "good game" scenarios as a prior lineup provide no incremental coverage and score poorly on the active rows.

**Coverage is the driving force for diversity**: two lineups that target the same set of high-upside scenarios will overlap heavily, so the algorithm naturally prefers lineups that spread coverage across different simulation branches.

### Addressing greedy lock-in (`BeamPortfolioConstructor`)

The greedy approach has a known failure mode: a dominant first lineup can consume a large block of scenarios, leaving a residual distribution that is hard for any single valid lineup to cover well. The `BeamPortfolioConstructor` mitigates this:

- Instead of committing to one best lineup per round, it maintains `beam_width` candidate portfolio paths simultaneously.
- At each depth, every path branches by running `optimize_top_k` and selecting up to `beam_width` distinct lineups from the active rows for that path.
- All candidate paths are pruned back to `beam_width` by ranking on **coverage** (fewest active rows remaining), breaking ties by total portfolio score.

This allows a path with a slightly weaker first lineup to survive if it enables much better subsequent lineups — directly trading off single-lineup optimality for portfolio-level coverage.

---

## Summary

| Stage | Key mechanism | Parameter handle |
|---|---|---|
| Simulation | Bootstrap copula rows → correlated quantiles → marginal PPF | `n_sims` |
| Player projections | `(mean, std_dev)` parameterize each player's marginal | External projection source |
| Lineup optimization | Basin-Hopping over `P(total ≥ target)` on active rows | `n_chains`, `n_steps`, `target` |
| Portfolio diversity | Row consumption forces successive lineups to cover new scenarios | `portfolio_size`, `beam_width` |
