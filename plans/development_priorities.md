- Doc that walks through simulations (what they are doing), how player projections factor in, how the optimizer selects a lineup from within the set of simulations, and how diversity is naturally achieved as the portfolio is constructed

- Improvements to std dev estimates by player type?

- Need DK Entry parser and DK Upload file maker

- Reduce chains/steps — A default setting of ~75 chains × 50 steps (3× quick mode) would likely land in 2–21 min range for those scenarios with meaningfully better lineup quality.

- Web UI

- How are we handling multi-position eligibility in the system and how SHOULD we handle multi-position eligibility?

- Integration test that runs the full pipeline: Currently test_simulation_pipeline.py is the only integration test and it's excluded from the default test run. A lightweight end-to-end test using the dummy data would catch regression at the seams between modules.

