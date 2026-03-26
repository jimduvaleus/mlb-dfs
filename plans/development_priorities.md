
- Reduce chains/steps — A default setting of ~75 chains × 50 steps (3× quick mode) would likely land in 2–21 min range for those scenarios with meaningfully better lineup quality.

Recommendation: 10k sims is the right setting for the 150-lineup worst case. If 25k sim accuracy is important for the first 20–30 lineups (before rows thin out), a two-pass approach would work: run 25k sims to build the top 20 lineups, then either stop or switch to 10k sims for the tail of the portfolio where rows are exhausted anyway and simulation fidelity no longer matters.

How are we handling multi-position eligibility in the system and how SHOULD we handle multi-position eligibility?

Integration test that runs the full pipeline: Currently test_simulation_pipeline.py is the only integration test and it's excluded from the default test run. A lightweight end-to-end test using the dummy data would catch regression at the seams between modules.

