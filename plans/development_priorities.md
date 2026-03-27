
- Is there anything strictly preventing an entry from having batters and an opposing pitcher, or is this expected to be a natural consequence of how lineups are chosen? 

- Improvements to std dev estimates by player type?

- Need DK Entry parser and DK Upload file maker

- Web UI should inform user of the state of the projections file (when last updated, etc.) and allow the user to fetch new projections via the rotowire script

- Web UI should display statistics for how long a portfolio took to create, among its metrics

- Show more detailed stacking information for each portfolio lineup (a lineup with 4 hitters from one team and 3 from another is a 4-3 stack, for example)

- fancy prediction market-based projections

- Integration test that runs the full pipeline: Currently test_simulation_pipeline.py is the only integration test and it's excluded from the default test run. A lightweight end-to-end test using dummy data would catch regression at the seams between modules.

