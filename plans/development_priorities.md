- get w_resid explainer
- config variables for top/bottom EVw
- config variable for portfolio $EV cutoff

source venv/bin/activate && python scripts/measure_pool_ceiling.py archive/07052026

The most interesting re-test is the tail bypass: sim-p99 was anti-selecting real ceiling precisely because of this defect. I'd leave tail_bypass_n: 0 for one or two slates, then re-enable at ~500 and let measure_pool_ceiling.py tell us whether the corrected sim-p99 now enriches — that's also the cleanest end-to-end validation of today's work.

Re-fit the overlay constants with diagnose_sim_tails.py as the archive grows (same pattern as env-sigma)