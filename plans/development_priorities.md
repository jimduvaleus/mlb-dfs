
Revised performance gain opportunities:
- Shared memory for the simulation matrix — replace the per-task pickle with multiprocessing.shared_memory so all workers read a single copy of the matrix from a shared memory block. This would allow the 8-worker parallelism to actually scale (~6–7×). Requires a moderate refactor of _run_chains in optimizer.py:324.

- Reduce chains/steps — for now, the practical knob without any code changes. Quick mode (25 chains × 20 steps) ran all 5k/10k scenarios in 0.7–7 min. A setting of ~75 chains × 40 steps (3× quick mode) would likely land in 2–21 min range for those scenarios with meaningfully better lineup quality.

- Pool reuse across lineup iterations — PortfolioConstructor creates a new optimizer (and therefore a new ProcessPoolExecutor) for each of the 20 lineups. The pool spawn overhead is paid 20 times unnecessarily.

How are we handling multi-position eligibility in the system and how SHOULD we handle multi-position eligibility?

Integration test that runs the full pipeline: Currently test_simulation_pipeline.py is the only integration test and it's excluded from the default test run. A lightweight end-to-end test using the dummy data would catch regression at the seams between modules.

