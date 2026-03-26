
- The negative-score clipping at engine.py:116 remains an open issue. Fixing that would require either allowing negative pitcher scores through to the optimizer (which actually helps — it makes 2-SP lineups look riskier, which is correct for tournaments) or using a truncated Gaussian for pitchers. Worth addressing as a separate change when you get to Phase 6 tuning.

- Parallelism: 
Run m = 250 independent chains using multiprocessing or Ray.?

- Add benchmark for 15000 and 20000 as opposed to 25000 and 50000

- Improvements to std dev estimates by player type?

- Need DK Entry parser and DK Upload file maker

- Reduce chains/steps — A default setting of ~75 chains × 50 steps (3× quick mode) would likely land in 2–21 min range for those scenarios with meaningfully better lineup quality.

- Two-pass approach, pros and cons?
Recommendation: 10k sims is the right setting for the 150-lineup worst case. If 25k sim accuracy is important for the first 20–30 lineups (before rows thin out), a two-pass approach would work: run 25k sims to build the top 20 lineups, then either stop or switch to 10k sims for the tail of the portfolio where rows are exhausted anyway and simulation fidelity no longer matters.

- Web UI

- How are we handling multi-position eligibility in the system and how SHOULD we handle multi-position eligibility?

- Integration test that runs the full pipeline: Currently test_simulation_pipeline.py is the only integration test and it's excluded from the default test run. A lightweight end-to-end test using the dummy data would catch regression at the seams between modules.

