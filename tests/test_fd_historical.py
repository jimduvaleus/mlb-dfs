"""
Tests for Phase 9: FanDuel historical pipeline.

Covers:
- Quality Start detection (derived from IP outs and ER — not a Retrosheet column)
- FD pitcher point calculation via ScoringRules
- FD batter point calculation via ScoringRules
- Backward compatibility: DK scoring unchanged when rules=None
"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.retrosheet_parser import RetrosheetParser
from src.platforms.fanduel import FD_SCORING
from src.platforms.draftkings import DK_SCORING


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pitching_row(
    ip_outs: int,
    er: int,
    w: int = 0,
    so: int = 0,
    h: int = 0,
    bb: int = 0,
    hb: int = 0,
    cg: int = 0,
) -> pd.DataFrame:
    """Build a single-row pitching DataFrame with GS=1."""
    return pd.DataFrame({
        "GS": [1],
        "W":  [w],
        "IP": [ip_outs],
        "ER": [er],
        "SO": [so],
        "H":  [h],
        "BB": [bb],
        "HB": [hb],
        "CG": [cg],
    })


def _make_batting_row(
    h: int = 0,
    d: int = 0,
    t: int = 0,
    hr: int = 0,
    rbi: int = 0,
    r: int = 0,
    bb: int = 0,
    hbp: int = 0,
    sb: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame({
        "H":   [h],
        "D":   [d],
        "T":   [t],
        "HR":  [hr],
        "RBI": [rbi],
        "R":   [r],
        "BB":  [bb],
        "HBP": [hbp],
        "SB":  [sb],
    })


# ---------------------------------------------------------------------------
# Quality Start detection
# ---------------------------------------------------------------------------

class TestQualityStartDetection:
    """QS = IP >= 18 outs (6 inn) AND ER <= 3.  Applies regardless of W/L."""

    def test_classic_qs_win(self):
        """6 IP, 2 ER, win — canonical QS."""
        df = RetrosheetParser.process_pitching_stats(_make_pitching_row(18, 2, w=1))
        assert df.iloc[0]["QS"] == 1

    def test_qs_no_decision(self):
        """6 IP, 3 ER, no decision — still a QS."""
        df = RetrosheetParser.process_pitching_stats(_make_pitching_row(18, 3, w=0))
        assert df.iloc[0]["QS"] == 1

    def test_qs_loss(self):
        """7 IP, 3 ER, loss — QS because definition is outcome-neutral."""
        df = RetrosheetParser.process_pitching_stats(_make_pitching_row(21, 3, w=0))
        assert df.iloc[0]["QS"] == 1

    def test_not_qs_too_few_innings(self):
        """5.2 IP (17 outs), 2 ER — not a QS (short of 6 full innings)."""
        df = RetrosheetParser.process_pitching_stats(_make_pitching_row(17, 2))
        assert df.iloc[0]["QS"] == 0

    def test_not_qs_exactly_six_innings_four_er(self):
        """6 IP, 4 ER — not a QS (exceeded allowed ER)."""
        df = RetrosheetParser.process_pitching_stats(_make_pitching_row(18, 4))
        assert df.iloc[0]["QS"] == 0

    def test_qs_boundary_exactly_18_outs_3_er(self):
        """Exact boundary: 18 outs and 3 ER — qualifies."""
        df = RetrosheetParser.process_pitching_stats(_make_pitching_row(18, 3))
        assert df.iloc[0]["QS"] == 1

    def test_qs_boundary_17_outs_0_er(self):
        """5.2 IP, 0 ER — not a QS (innings threshold not met)."""
        df = RetrosheetParser.process_pitching_stats(_make_pitching_row(17, 0))
        assert df.iloc[0]["QS"] == 0

    def test_qs_column_present_for_dk_pipeline(self):
        """QS column is always computed even for the DK pipeline (rules=None)."""
        df = RetrosheetParser.process_pitching_stats(_make_pitching_row(18, 2))
        assert "QS" in df.columns


# ---------------------------------------------------------------------------
# FanDuel pitcher scoring
# ---------------------------------------------------------------------------

class TestFDPitcherScoring:
    """
    FD pitcher: W=6, QS=4, ER=-3, SO=3, IP=3.
    No bonuses for H, BB, HB, CG.
    """

    def test_qs_adds_four_points(self):
        """A QS (18 outs, <=3 ER) should add exactly 4 FD points."""
        # 6 IP (18 outs), 0 ER, 0 SO, no win → only QS points
        df = RetrosheetParser.process_pitching_stats(
            _make_pitching_row(18, 0), rules=FD_SCORING
        )
        # IP_dec = 6.0 → 6 * 3 = 18, QS = 4
        expected = 6.0 * 3 + 4  # ip + qs
        assert df.iloc[0]["dk_points"] == pytest.approx(expected)

    def test_win_and_qs(self):
        """6 IP, 2 ER, win → W=6, QS=4, IP=18, ER=-6."""
        df = RetrosheetParser.process_pitching_stats(
            _make_pitching_row(18, 2, w=1, so=6), rules=FD_SCORING
        )
        expected = 6 + 4 + (6 * 3) + (2 * -3) + (6 * 3)  # W + QS + IP + ER + SO
        assert df.iloc[0]["dk_points"] == pytest.approx(expected)

    def test_no_qs_bonus_when_not_qualified(self):
        """5 IP, 0 ER — no QS bonus despite shutting out opponents."""
        df = RetrosheetParser.process_pitching_stats(
            _make_pitching_row(15, 0), rules=FD_SCORING
        )
        expected = (15 / 3) * 3  # ip only, no QS
        assert df.iloc[0]["dk_points"] == pytest.approx(expected)

    def test_hits_walks_hbp_ignored_for_fd(self):
        """FD has no H/BB/HB penalties — should not affect score."""
        base = RetrosheetParser.process_pitching_stats(
            _make_pitching_row(18, 0), rules=FD_SCORING
        )
        with_extras = RetrosheetParser.process_pitching_stats(
            _make_pitching_row(18, 0, h=10, bb=5, hb=2), rules=FD_SCORING
        )
        assert base.iloc[0]["dk_points"] == pytest.approx(with_extras.iloc[0]["dk_points"])

    def test_cg_ignored_for_fd(self):
        """FD has no CG bonus."""
        no_cg = RetrosheetParser.process_pitching_stats(
            _make_pitching_row(27, 0, cg=0), rules=FD_SCORING
        )
        with_cg = RetrosheetParser.process_pitching_stats(
            _make_pitching_row(27, 0, cg=1), rules=FD_SCORING
        )
        assert no_cg.iloc[0]["dk_points"] == pytest.approx(with_cg.iloc[0]["dk_points"])


# ---------------------------------------------------------------------------
# FanDuel batter scoring
# ---------------------------------------------------------------------------

class TestFDBatterScoring:
    """
    FD batter: 1B=3, 2B=6, 3B=9, HR=12, RBI=3.5, R=3.2, BB=3, SB=6, HBP=3.
    """

    def test_home_run(self):
        df = RetrosheetParser.process_batting_stats(
            _make_batting_row(h=1, hr=1, rbi=1, r=1), rules=FD_SCORING
        )
        # HR=12, RBI=3.5, R=3.2 (no single because H-HR=0)
        expected = 12 + 3.5 + 3.2
        assert df.iloc[0]["dk_points"] == pytest.approx(expected)

    def test_single_walk_sb(self):
        df = RetrosheetParser.process_batting_stats(
            _make_batting_row(h=1, bb=1, sb=1, r=1), rules=FD_SCORING
        )
        expected = 3 + 3 + 6 + 3.2  # 1B + BB + SB + R
        assert df.iloc[0]["dk_points"] == pytest.approx(expected)

    def test_hbp_worth_three(self):
        df = RetrosheetParser.process_batting_stats(
            _make_batting_row(hbp=1), rules=FD_SCORING
        )
        assert df.iloc[0]["dk_points"] == pytest.approx(3.0)

    def test_double_triple(self):
        df = RetrosheetParser.process_batting_stats(
            _make_batting_row(h=2, d=1, t=1), rules=FD_SCORING
        )
        # H=2, D=1, T=1 → 1B = 2-1-1=0; double=6, triple=9
        expected = 6 + 9
        assert df.iloc[0]["dk_points"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# DK backward compatibility
# ---------------------------------------------------------------------------

class TestDKBackwardCompat:
    """Existing DK pipeline must be unchanged when rules=None."""

    def test_dk_pitcher_no_qs_points(self):
        """DK has no QS bonus; QS column present but contributes 0."""
        df_none = RetrosheetParser.process_pitching_stats(_make_pitching_row(18, 2, w=1, so=6))
        df_dk = RetrosheetParser.process_pitching_stats(
            _make_pitching_row(18, 2, w=1, so=6), rules=DK_SCORING
        )
        assert df_none.iloc[0]["dk_points"] == pytest.approx(df_dk.iloc[0]["dk_points"])

    def test_dk_batter_points_unchanged(self):
        """rules=None and rules=DK_SCORING must produce identical batter scores."""
        row = _make_batting_row(h=2, d=1, hr=1, rbi=2, r=1, bb=1, sb=1)
        df_none = RetrosheetParser.process_batting_stats(row)
        df_dk = RetrosheetParser.process_batting_stats(row, rules=DK_SCORING)
        assert df_none.iloc[0]["dk_points"] == pytest.approx(df_dk.iloc[0]["dk_points"])

    def test_starters_only_filter_still_works(self):
        """starters_only=True must still filter by GS regardless of rules."""
        df = pd.DataFrame({
            "GS": [1, 0],
            "W":  [1, 0],
            "IP": [18, 21],
            "ER": [2, 1],
            "SO": [6, 5],
            "H":  [5, 4],
            "BB": [2, 1],
            "HB": [0, 0],
            "CG": [0, 0],
        })
        result = RetrosheetParser.process_pitching_stats(df, starters_only=True, rules=FD_SCORING)
        assert len(result) == 1
        assert result.iloc[0]["QS"] == 1
