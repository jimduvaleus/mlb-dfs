"""
Regression contract for DraftKings scoring constants in src/utils/scoring.py.

These assertions act as a safety wire for the historical pipeline:
  retrosheet_parser.py → historical_logs.parquet → copula → simulation

If any constant changes, this test fails loudly before the corruption
reaches the parquet file.  See the HISTORICAL PIPELINE WARNING in scoring.py
for full context.

DO NOT update expected values here unless you are intentionally rebuilding
the copula from scratch (Phase 9 of the FanDuel Platform Plan).
"""

import src.utils.scoring as s


class TestDKScoringContract:
    # ----- Batter constants -----

    def test_batter_single(self):
        assert s.BATTER_SINGLE == 3

    def test_batter_double(self):
        assert s.BATTER_DOUBLE == 5

    def test_batter_triple(self):
        assert s.BATTER_TRIPLE == 8

    def test_batter_home_run(self):
        assert s.BATTER_HOME_RUN == 10

    def test_batter_rbi(self):
        assert s.BATTER_RBI == 2

    def test_batter_run(self):
        assert s.BATTER_RUN == 2

    def test_batter_walk(self):
        assert s.BATTER_WALK == 2

    def test_batter_hbp(self):
        assert s.BATTER_HBP == 2

    def test_batter_sb(self):
        assert s.BATTER_SB == 5

    # ----- Pitcher constants -----

    def test_pitcher_win(self):
        assert s.PITCHER_WIN == 4

    def test_pitcher_er(self):
        assert s.PITCHER_ER == -2

    def test_pitcher_so(self):
        assert s.PITCHER_SO == 2

    def test_pitcher_ip(self):
        assert s.PITCHER_IP == 2.25

    def test_pitcher_h(self):
        assert s.PITCHER_H == -0.6

    def test_pitcher_bb(self):
        assert s.PITCHER_BB == -0.6

    def test_pitcher_hb(self):
        assert s.PITCHER_HB == -0.6

    def test_pitcher_cg(self):
        assert s.PITCHER_CG == 2.5

    def test_pitcher_cgs(self):
        assert s.PITCHER_CGS == 2.5

    def test_pitcher_nh(self):
        assert s.PITCHER_NH == 5

    # ----- Smoke test: calculation functions still use the constants -----

    def test_calculate_batter_points_hr(self):
        pts = s.calculate_batter_points(hr=1, rbi=1, run=1)
        assert pts == s.BATTER_HOME_RUN + s.BATTER_RBI + s.BATTER_RUN

    def test_calculate_pitcher_points_qs(self):
        # 6 IP, 0 ER, 7 SO, no win
        pts = s.calculate_pitcher_points(ip=6, er=0, so=7)
        assert pts == 6 * s.PITCHER_IP + 7 * s.PITCHER_SO
