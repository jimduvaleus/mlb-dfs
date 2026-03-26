
- Reduce chains/steps — A default setting of ~75 chains × 50 steps (3× quick mode) would likely land in 2–21 min range for those scenarios with meaningfully better lineup quality.

- Pool reuse across lineup iterations — PortfolioConstructor creates a new optimizer (and therefore a new ProcessPoolExecutor) for each of the 20 lineups. The pool spawn overhead is paid 20 times unnecessarily.

How are we handling multi-position eligibility in the system and how SHOULD we handle multi-position eligibility?

Integration test that runs the full pipeline: Currently test_simulation_pipeline.py is the only integration test and it's excluded from the default test run. A lightweight end-to-end test using the dummy data would catch regression at the seams between modules.

