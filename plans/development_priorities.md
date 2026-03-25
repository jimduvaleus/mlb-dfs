1. Real Copula Data (highest priority)
Right now the entire pipeline runs on dummy data — normal-distributed synthetic scores with no real correlation structure. This means every optimization result is meaningless regardless of how good the code is. The copula is the heart of this system; without real game-level correlations, the portfolio optimizer is just shuffling noise.

What's needed: Run scripts/process_historical.py against real Retrosheet .EVN files to produce historical_logs.parquet, then build_copula.py and fit_batter_pca.py. This also unblocks the missing batter_pca_model.npz and batter_score_grid.npy artifacts that Phase 4 requires.

I'd prioritize this above all code work — no code change will matter until the data is real.

2. End-to-End CLI (main.py)
There's no single entry point that orchestrates the full pipeline: load slate → load copula → simulate → optimize → output portfolio. Right now a user has to write the ~20-line script shown in the exploration manually. A main.py with YAML config (as spec'd in Section 8 of the architecture doc) would make the system actually usable on game day.

3. Phase 6 Performance (Numba, not Ray yet)
The Basin-Hopping objective function (_objective) is called thousands of times per chain × 250 chains. Profiling will likely show that the inner loop (summing simulation rows, comparing to target) is the bottleneck. @numba.njit on that hot path is a straightforward win — probably 10-50x speedup on the objective evaluation.

Ray is less urgent: ProcessPoolExecutor already parallelizes chains, and Ray adds deployment complexity. I'd defer Ray until you're running this on a multi-node setup or need distributed simulation.

4. Greedy vs Beam-Search vs SA for Portfolio Construction
Here's the tradeoff analysis for your specific setup:

Approach	Pros	Cons	Fit for this project
Current greedy	Simple, fast, deterministic	No backtracking; early lineup choices lock in suboptimal coverage	Good baseline
Beam search	Explores k portfolio paths in parallel; can recover from bad early picks	k× slower; memory scales with beam width × portfolio size	Best next step — moderate complexity, clear improvement
SA over full portfolio	Theoretically optimal coverage; can escape local minima	Very expensive (each "move" requires re-optimizing a lineup); hard to tune temperature schedule	Overkill for now
My recommendation: The greedy approach has a known weakness — if lineup 1 "consumes" rows that would have been better served by a different lineup 1 + lineup 2 combination, you're stuck. A beam search with width 3-5 is the sweet spot: you maintain a few candidate portfolio paths, and at each step you try the top-k lineups (not just the best), branching and pruning. The overhead is modest (3-5× the current cost per portfolio slot) and it directly addresses the greedy lock-in problem.

SA over the full portfolio is theoretically appealing but impractical — each perturbation means re-running Basin-Hopping, which is already your most expensive operation.

5. Other Ideas Worth Considering
Overlap constraints: The phase5_complete.md notes that the same player can appear in multiple lineups. For GPP contests, some overlap is fine, but you probably want a configurable max-exposure cap (e.g., no player in more than 60% of lineups). This is a small code change with big practical impact.

Integration test that runs the full pipeline: Currently test_simulation_pipeline.py is the only integration test and it's excluded from the default test run. A lightweight end-to-end test using the dummy data would catch regression at the seams between modules.

Target score auto-calibration: Right now target is a manual input. You could set it automatically as the p95 or p99 of the simulation distribution, which adapts to the slate.

Recommended Order
Real Retrosheet data → real copula + PCA artifacts
End-to-end CLI with YAML config
Numba on the objective function
Max-exposure constraint in portfolio construction
Beam-search portfolio (once the above are solid)
Documentation (the codebase is already well-structured; formal docs can wait)
The first two items make the system usable. Items 3-5 make it competitive. Documentation is lowest priority because the code is clean, well-tested, and the phase completion docs already capture the key decisions.