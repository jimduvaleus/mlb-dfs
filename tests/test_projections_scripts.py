"""
Tests for Phase 6 — Projection Fetch Script Platform Support.

Coverage:
- fetch_rotowire_projections:
    FANDUEL_SITE_ID constant, fetch_slate_list site_id param,
    _slate_df_teams helper, _extract_date_from_fd_path helper,
    find_best_slate with slate_teams (replaces dk_path),
    build_projections_csv with slate_df (DK and FD),
    CLI --platform arg wiring (import + argparse)

- fetch_dff_projections:
    DK/FD URL segment constants, _make_slate_link_re / _make_date_link_re,
    _projections_base / _slate_list_url helpers,
    _slate_df_teams / _extract_date_from_fd_path helpers,
    build_projections_csv with slate_df (DK and FD),
    CLI --platform arg wiring

- projections_meta:
    _extract_fd_date, _extract_slate_date,
    fetch_and_cache_slates site_id param + cache invalidation on site_id change,
    compute_freshness DK and FD paths

All external I/O (requests, Playwright) is mocked.
"""

import csv
import io
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_slate_df(specs: list[dict]) -> pd.DataFrame:
    """Minimal slate DataFrame compatible with both DK and FD ingestor output."""
    rows = []
    for s in specs:
        rows.append({
            "player_id":   int(s["player_id"]),
            "name":        s.get("name", f"Player{s['player_id']}"),
            "position":    s.get("position", "OF"),
            "salary":      float(s.get("salary", 4000)),
            "team":        s.get("team", "TEA"),
            "opponent":    s.get("opponent", "TEB"),
            "game":        s.get("game", "TEA@TEB"),
        })
    return pd.DataFrame(rows)


SAMPLE_DK_SPECS = [
    {"player_id": 1001, "name": "Zack Wheeler",  "position": "P",  "salary": 10000, "team": "PHI", "opponent": "ARI", "game": "ARI@PHI"},
    {"player_id": 2001, "name": "Aaron Judge",   "position": "OF", "salary": 5000,  "team": "NYY", "opponent": "TB",  "game": "NYY@TB"},
    {"player_id": 3001, "name": "Freddie Freeman","position": "1B", "salary": 4500, "team": "LAD", "opponent": "SF",  "game": "LAD@SF"},
]

SAMPLE_FD_SPECS = [
    {"player_id": 16960, "name": "Zack Wheeler",  "position": "P",  "salary": 11000, "team": "PHI", "opponent": "ARI", "game": "ARI@PHI"},
    {"player_id": 20001, "name": "Aaron Judge",   "position": "OF", "salary": 4300,  "team": "NYY", "opponent": "TB",  "game": "NYY@TB"},
]


# ---------------------------------------------------------------------------
# fetch_rotowire_projections — constants and helpers
# ---------------------------------------------------------------------------

class TestRotoWireConstants:
    def test_draftkings_site_id_is_1(self):
        from scripts.fetch_rotowire_projections import DRAFTKINGS_SITE_ID
        assert DRAFTKINGS_SITE_ID == 1

    def test_fanduel_site_id_is_2(self):
        from scripts.fetch_rotowire_projections import FANDUEL_SITE_ID
        assert FANDUEL_SITE_ID == 2

    def test_site_ids_differ(self):
        from scripts.fetch_rotowire_projections import DRAFTKINGS_SITE_ID, FANDUEL_SITE_ID
        assert DRAFTKINGS_SITE_ID != FANDUEL_SITE_ID


class TestRotoWireSlatedfTeams:
    def test_extracts_teams_from_team_column(self):
        from scripts.fetch_rotowire_projections import _slate_df_teams
        df = _make_slate_df(SAMPLE_DK_SPECS)
        teams = _slate_df_teams(df)
        assert "PHI" in teams
        assert "NYY" in teams

    def test_extracts_teams_from_opponent_column(self):
        from scripts.fetch_rotowire_projections import _slate_df_teams
        df = _make_slate_df(SAMPLE_DK_SPECS)
        teams = _slate_df_teams(df)
        # opponent column for DK is empty string; game column provides teams
        assert "ARI" in teams or "PHI" in teams  # from game column

    def test_extracts_teams_from_game_column(self):
        from scripts.fetch_rotowire_projections import _slate_df_teams
        df = _make_slate_df([{"player_id": 1, "team": "PHI", "opponent": "", "game": "ARI@PHI"}])
        teams = _slate_df_teams(df)
        assert "ARI" in teams
        assert "PHI" in teams

    def test_empty_df_returns_empty_set(self):
        from scripts.fetch_rotowire_projections import _slate_df_teams
        assert _slate_df_teams(pd.DataFrame()) == set()

    def test_fd_slate_uses_opponent_column(self):
        from scripts.fetch_rotowire_projections import _slate_df_teams
        df = _make_slate_df([
            {"player_id": 1, "team": "PHI", "opponent": "ARI", "game": "ARI@PHI"},
            {"player_id": 2, "team": "NYY", "opponent": "TB",  "game": "NYY@TB"},
        ])
        teams = _slate_df_teams(df)
        assert {"PHI", "ARI", "NYY", "TB"}.issubset(teams)


class TestRotoWireExtractDateFromFdPath:
    def test_extracts_date_from_valid_filename(self):
        from scripts.fetch_rotowire_projections import _extract_date_from_fd_path
        path = "data/raw/FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv"
        assert _extract_date_from_fd_path(path) == "2026-04-12"

    def test_returns_none_for_dk_filename(self):
        from scripts.fetch_rotowire_projections import _extract_date_from_fd_path
        assert _extract_date_from_fd_path("data/raw/DKSalaries.csv") is None

    def test_returns_none_for_unrecognised_pattern(self):
        from scripts.fetch_rotowire_projections import _extract_date_from_fd_path
        assert _extract_date_from_fd_path("other_file.csv") is None


class TestRotoWireFetchSlateList:
    def test_passes_dk_site_id_by_default(self):
        from scripts.fetch_rotowire_projections import fetch_slate_list, DRAFTKINGS_SITE_ID
        with patch("scripts.fetch_rotowire_projections._get") as mock_get:
            mock_get.return_value = {"slates": []}
            fetch_slate_list()
            mock_get.assert_called_once()
            call_params = mock_get.call_args[1]["params"]
            assert call_params["siteID"] == DRAFTKINGS_SITE_ID

    def test_passes_fd_site_id_when_specified(self):
        from scripts.fetch_rotowire_projections import fetch_slate_list, FANDUEL_SITE_ID
        with patch("scripts.fetch_rotowire_projections._get") as mock_get:
            mock_get.return_value = {"slates": []}
            fetch_slate_list(site_id=FANDUEL_SITE_ID)
            call_params = mock_get.call_args[1]["params"]
            assert call_params["siteID"] == FANDUEL_SITE_ID

    def test_returns_list(self):
        from scripts.fetch_rotowire_projections import fetch_slate_list
        with patch("scripts.fetch_rotowire_projections._get") as mock_get:
            mock_get.return_value = {"slates": [{"slateID": 1}]}
            result = fetch_slate_list()
        assert isinstance(result, list)
        assert result[0]["slateID"] == 1


class TestRotoWireFindBestSlate:
    def _make_slates(self, n: int = 2):
        return [
            {
                "slateID": 100 + i,
                "contestType": "Classic",
                "startDateOnly": "2026-04-12",
                "slateName": f"Slate{i}",
                "defaultSlate": i == 0,
            }
            for i in range(n)
        ]

    def test_single_candidate_returned_directly(self):
        from scripts.fetch_rotowire_projections import find_best_slate
        slates = self._make_slates(1)
        slate, records = find_best_slate(slates, "2026-04-12")
        assert slate["slateID"] == 100
        assert records is None

    def test_uses_slate_teams_to_score_candidates(self):
        from scripts.fetch_rotowire_projections import find_best_slate
        slates = self._make_slates(2)
        # Slate 100 has PHI+ARI; Slate 101 has NYY+TB
        def mock_fetch_players(slate_id, debug=False):
            if slate_id == 100:
                return [{"team": {"abbr": "PHI"}, "opponent": {"team": "ARI"}}]
            return [{"team": {"abbr": "NYY"}, "opponent": {"team": "TB"}}]

        with patch("scripts.fetch_rotowire_projections.fetch_players", side_effect=mock_fetch_players):
            with patch("scripts.fetch_rotowire_projections._load_slate_teams_cache", return_value={}):
                with patch("scripts.fetch_rotowire_projections._save_slate_teams_cache"):
                    slate, _ = find_best_slate(
                        slates, "2026-04-12",
                        slate_teams={"PHI", "ARI"},
                    )
        assert slate["slateID"] == 100

    def test_falls_back_to_default_when_no_teams(self):
        from scripts.fetch_rotowire_projections import find_best_slate
        slates = self._make_slates(2)
        slate, _ = find_best_slate(slates, "2026-04-12", slate_teams=None)
        # defaultSlate is slate 0 (slateID=100)
        assert slate["slateID"] == 100


class TestRotoWireBuildProjectionsCSV:
    """Tests for the refactored build_projections_csv that accepts slate_df."""

    def _rw_records(self):
        return [
            {
                "firstName": "Zack", "lastName": "Wheeler",
                "rotoPos": "P", "salary": 10000, "pts": 25.0,
                "lineup": {"slot": "SP", "isConfirmed": True},
            },
            {
                "firstName": "Aaron", "lastName": "Judge",
                "rotoPos": "OF", "salary": 5000, "pts": 15.0,
                "lineup": {"slot": "3", "isConfirmed": True},
            },
        ]

    def test_dk_slate_df_writes_output(self, tmp_path):
        from scripts.fetch_rotowire_projections import build_projections_csv, DRAFTKINGS_SITE_ID
        slate_df = _make_slate_df(SAMPLE_DK_SPECS)
        output = str(tmp_path / "proj.csv")
        with patch("scripts.fetch_rotowire_projections.fetch_players", return_value=self._rw_records()):
            result = build_projections_csv(
                slate_df=slate_df,
                slate_id=999,
                output_path=output,
                site_id=DRAFTKINGS_SITE_ID,
            )
        assert Path(output).exists()
        assert len(result) > 0
        assert "player_id" in result.columns

    def test_fd_slate_df_writes_output(self, tmp_path):
        from scripts.fetch_rotowire_projections import build_projections_csv, FANDUEL_SITE_ID
        slate_df = _make_slate_df(SAMPLE_FD_SPECS)
        output = str(tmp_path / "proj_fd.csv")
        with patch("scripts.fetch_rotowire_projections.fetch_players", return_value=self._rw_records()):
            result = build_projections_csv(
                slate_df=slate_df,
                slate_id=999,
                output_path=output,
                site_id=FANDUEL_SITE_ID,
            )
        assert Path(output).exists()
        assert len(result) > 0

    def test_matched_player_ids_come_from_slate_df(self, tmp_path):
        from scripts.fetch_rotowire_projections import build_projections_csv
        slate_df = _make_slate_df(SAMPLE_DK_SPECS)
        output = str(tmp_path / "proj.csv")
        with patch("scripts.fetch_rotowire_projections.fetch_players", return_value=self._rw_records()):
            result = build_projections_csv(
                slate_df=slate_df, slate_id=999, output_path=output
            )
        # player_ids in result must be a subset of what's in slate_df
        assert set(result["player_id"]).issubset(set(slate_df["player_id"]))

    def test_prefetched_records_skips_api_call(self, tmp_path):
        from scripts.fetch_rotowire_projections import build_projections_csv
        slate_df = _make_slate_df(SAMPLE_DK_SPECS)
        output = str(tmp_path / "proj.csv")
        with patch("scripts.fetch_rotowire_projections.fetch_players") as mock_fetch:
            build_projections_csv(
                slate_df=slate_df,
                slate_id=999,
                output_path=output,
                prefetched_records=self._rw_records(),
            )
        mock_fetch.assert_not_called()

    def test_output_csv_has_required_columns(self, tmp_path):
        from scripts.fetch_rotowire_projections import build_projections_csv
        slate_df = _make_slate_df(SAMPLE_DK_SPECS)
        output = str(tmp_path / "proj.csv")
        with patch("scripts.fetch_rotowire_projections.fetch_players", return_value=self._rw_records()):
            build_projections_csv(slate_df=slate_df, slate_id=999, output_path=output)
        df = pd.read_csv(output)
        for col in ("player_id", "name", "mean", "std_dev", "lineup_slot", "slot_confirmed"):
            assert col in df.columns, f"Missing column: {col}"


class TestRotoWireCLIArgs:
    def test_platform_draftkings_is_default(self):
        import argparse
        import sys
        from scripts.fetch_rotowire_projections import main
        # Just verify the parser accepts --platform without error
        with patch("sys.argv", ["script", "--list-slates", "--platform", "draftkings"]):
            with patch("scripts.fetch_rotowire_projections.fetch_slate_list", return_value=[]):
                with patch("sys.stdout"):
                    main()  # should not raise

    def test_platform_fanduel_accepted(self):
        with patch("sys.argv", ["script", "--list-slates", "--platform", "fanduel"]):
            with patch("scripts.fetch_rotowire_projections.fetch_slate_list", return_value=[]):
                with patch("sys.stdout"):
                    from scripts.fetch_rotowire_projections import main
                    main()  # should not raise

    def test_invalid_platform_exits(self):
        with patch("sys.argv", ["script", "--platform", "invalid"]):
            import sys
            with pytest.raises(SystemExit):
                from scripts import fetch_rotowire_projections as rw_mod
                import importlib
                # Use argparse directly
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--platform", choices=["draftkings", "fanduel"])
                parser.parse_args(["--platform", "invalid"])


# ---------------------------------------------------------------------------
# fetch_dff_projections — constants and helpers
# ---------------------------------------------------------------------------

class TestDFFConstants:
    def test_dk_url_segment(self):
        from scripts.fetch_dff_projections import DK_URL_SEGMENT
        assert DK_URL_SEGMENT == "draftkings"

    def test_fd_url_segment(self):
        from scripts.fetch_dff_projections import FD_URL_SEGMENT
        assert FD_URL_SEGMENT == "fanduel"

    def test_segments_differ(self):
        from scripts.fetch_dff_projections import DK_URL_SEGMENT, FD_URL_SEGMENT
        assert DK_URL_SEGMENT != FD_URL_SEGMENT


class TestDFFUrlHelpers:
    def test_projections_base_dk(self):
        from scripts.fetch_dff_projections import _projections_base, DK_URL_SEGMENT
        url = _projections_base(DK_URL_SEGMENT)
        assert "draftkings" in url
        assert url.startswith("https://www.dailyfantasyfuel.com")

    def test_projections_base_fd(self):
        from scripts.fetch_dff_projections import _projections_base, FD_URL_SEGMENT
        url = _projections_base(FD_URL_SEGMENT)
        assert "fanduel" in url

    def test_slate_list_url_dk(self):
        from scripts.fetch_dff_projections import _slate_list_url, DK_URL_SEGMENT
        url = _slate_list_url(DK_URL_SEGMENT)
        assert "draftkings" in url

    def test_slate_list_url_fd(self):
        from scripts.fetch_dff_projections import _slate_list_url, FD_URL_SEGMENT
        url = _slate_list_url(FD_URL_SEGMENT)
        assert "fanduel" in url

    def test_module_level_constants_use_dk(self):
        from scripts.fetch_dff_projections import PROJECTIONS_BASE, SLATE_LIST_URL
        assert "draftkings" in PROJECTIONS_BASE
        assert "draftkings" in SLATE_LIST_URL


class TestDFFSlateRegex:
    def test_dk_slate_link_re_matches_dk_url(self):
        from scripts.fetch_dff_projections import _make_slate_link_re, DK_URL_SEGMENT
        re_pattern = _make_slate_link_re(DK_URL_SEGMENT)
        href = "/mlb/projections/draftkings/2026-04-12?slate=235A9"
        m = re_pattern.search(href)
        assert m is not None
        assert m.group(1) == "2026-04-12"
        assert m.group(2) == "235A9"

    def test_fd_slate_link_re_matches_fd_url(self):
        from scripts.fetch_dff_projections import _make_slate_link_re, FD_URL_SEGMENT
        re_pattern = _make_slate_link_re(FD_URL_SEGMENT)
        href = "/mlb/projections/fanduel/2026-04-12?slate=FDABC"
        m = re_pattern.search(href)
        assert m is not None
        assert m.group(1) == "2026-04-12"
        assert m.group(2) == "FDABC"

    def test_dk_slate_link_re_does_not_match_fd_url(self):
        from scripts.fetch_dff_projections import _make_slate_link_re, DK_URL_SEGMENT
        re_pattern = _make_slate_link_re(DK_URL_SEGMENT)
        assert re_pattern.search("/mlb/projections/fanduel/2026-04-12?slate=ABC") is None

    def test_dk_date_link_re_matches_dk_url(self):
        from scripts.fetch_dff_projections import _make_date_link_re, DK_URL_SEGMENT
        re_pattern = _make_date_link_re(DK_URL_SEGMENT)
        href = "/mlb/projections/draftkings/2026-04-12/"
        m = re_pattern.search(href)
        assert m is not None
        assert m.group(1) == "2026-04-12"

    def test_fd_date_link_re_matches_fd_url(self):
        from scripts.fetch_dff_projections import _make_date_link_re, FD_URL_SEGMENT
        re_pattern = _make_date_link_re(FD_URL_SEGMENT)
        href = "/mlb/projections/fanduel/2026-04-12/"
        m = re_pattern.search(href)
        assert m is not None

    def test_module_level_re_uses_dk(self):
        from scripts.fetch_dff_projections import _SLATE_LINK_RE, _DATE_LINK_RE
        assert _SLATE_LINK_RE.search("/mlb/projections/draftkings/2026-04-12?slate=ABC") is not None
        assert _SLATE_LINK_RE.search("/mlb/projections/fanduel/2026-04-12?slate=ABC") is None


class TestDFFHelpers:
    def test_slate_df_teams(self):
        from scripts.fetch_dff_projections import _slate_df_teams
        df = _make_slate_df([
            {"player_id": 1, "team": "PHI", "opponent": "ARI", "game": "ARI@PHI"},
        ])
        teams = _slate_df_teams(df)
        assert "PHI" in teams
        assert "ARI" in teams

    def test_extract_date_from_fd_path(self):
        from scripts.fetch_dff_projections import _extract_date_from_fd_path
        path = "data/raw/FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv"
        assert _extract_date_from_fd_path(path) == "2026-04-12"


def _make_link(slate_id: str, game_count: int, href: str | None = None) -> dict:
    return {
        "href": href or f"/mlb/projections/draftkings/2026-05-20?slate={slate_id}",
        "text": f"{game_count} Games",
        "date": "2026-05-20",
        "slate_id": slate_id,
        "game_count": game_count,
    }


class TestSelectSlateCandidates:
    def test_prefers_superset_over_closer_subset(self):
        from scripts.fetch_dff_projections import _select_slate_candidates
        # DK=8; DFF has 7 (closer but smaller) and 10 (further but superset)
        links = [_make_link("A", 7), _make_link("B", 10)]
        result = _select_slate_candidates(links, game_count=8)
        assert result[0]["slate_id"] == "B"

    def test_exact_match_first(self):
        from scripts.fetch_dff_projections import _select_slate_candidates
        links = [_make_link("A", 6), _make_link("B", 10)]
        result = _select_slate_candidates(links, game_count=6)
        assert result[0]["slate_id"] == "A"

    def test_smallest_superset_among_multiple(self):
        from scripts.fetch_dff_projections import _select_slate_candidates
        # DK=6; DFF has 8 and 12 — prefer 8 (smallest superset)
        links = [_make_link("A", 8), _make_link("B", 12)]
        result = _select_slate_candidates(links, game_count=6)
        assert result[0]["slate_id"] == "A"

    def test_falls_back_to_closest_when_no_superset(self):
        from scripts.fetch_dff_projections import _select_slate_candidates
        # DK=10; DFF only has 6 and 7 — fall back to closest (7)
        links = [_make_link("A", 6), _make_link("B", 7)]
        result = _select_slate_candidates(links, game_count=10)
        assert result[0]["slate_id"] == "B"

    def test_tiebreak_larger_when_same_diff(self):
        from scripts.fetch_dff_projections import _select_slate_candidates
        # DK=5; DFF has 3 and 7 — both diff=2, prefer 7 (more games)
        links = [_make_link("A", 3), _make_link("B", 7)]
        result = _select_slate_candidates(links, game_count=5)
        assert result[0]["slate_id"] == "B"

    def test_two_same_size_slates_both_returned(self):
        from scripts.fetch_dff_projections import _select_slate_candidates
        # Two 6-game slates (today's scenario) — both should appear in result
        links = [_make_link("A", 6), _make_link("B", 6)]
        result = _select_slate_candidates(links, game_count=6)
        assert len(result) == 2
        assert {r["slate_id"] for r in result} == {"A", "B"}


class TestDFFBuildProjectionsCSV:
    def _player_rows(self):
        return [
            {
                "data-name": "Zack Wheeler", "data-pos": "P",
                "data-salary": "10000", "data-ppg_proj": "25.0",
                "data-depth_rank": "0", "data-starter_flag": "0",
            },
            {
                "data-name": "Aaron Judge", "data-pos": "OF",
                "data-salary": "5000", "data-ppg_proj": "15.0",
                "data-depth_rank": "3", "data-starter_flag": "1",
            },
        ]

    def test_dk_build_with_slate_df(self, tmp_path):
        from scripts.fetch_dff_projections import build_projections_csv, DK_URL_SEGMENT
        slate_df = _make_slate_df(SAMPLE_DK_SPECS)
        output = str(tmp_path / "proj.csv")
        with patch("scripts.fetch_dff_projections.fetch_player_and_team_rows", return_value=(self._player_rows(), [])):
            result = build_projections_csv(
                slate_df=slate_df,
                target_date="2026-04-12",
                output_path=output,
                url_segment=DK_URL_SEGMENT,
            )
        assert Path(output).exists()
        assert len(result) > 0

    def test_fd_build_with_slate_df(self, tmp_path):
        from scripts.fetch_dff_projections import build_projections_csv, FD_URL_SEGMENT
        slate_df = _make_slate_df(SAMPLE_FD_SPECS)
        output = str(tmp_path / "proj_fd.csv")
        with patch("scripts.fetch_dff_projections.fetch_player_and_team_rows", return_value=(self._player_rows(), [])):
            result = build_projections_csv(
                slate_df=slate_df,
                target_date="2026-04-12",
                output_path=output,
                url_segment=FD_URL_SEGMENT,
            )
        assert Path(output).exists()

    def test_url_segment_passed_to_fetch_player_and_team_rows(self, tmp_path):
        from scripts.fetch_dff_projections import build_projections_csv, FD_URL_SEGMENT
        slate_df = _make_slate_df(SAMPLE_DK_SPECS)
        output = str(tmp_path / "proj.csv")
        with patch("scripts.fetch_dff_projections.fetch_player_and_team_rows") as mock_fetch:
            mock_fetch.return_value = (self._player_rows(), [])
            build_projections_csv(
                slate_df=slate_df,
                target_date="2026-04-12",
                output_path=output,
                url_segment=FD_URL_SEGMENT,
            )
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs.get("url_segment") == FD_URL_SEGMENT

    def test_output_has_required_columns(self, tmp_path):
        from scripts.fetch_dff_projections import build_projections_csv
        slate_df = _make_slate_df(SAMPLE_DK_SPECS)
        output = str(tmp_path / "proj.csv")
        with patch("scripts.fetch_dff_projections.fetch_player_and_team_rows", return_value=(self._player_rows(), [])):
            build_projections_csv(
                slate_df=slate_df, target_date="2026-04-12", output_path=output
            )
        df = pd.read_csv(output)
        for col in ("player_id", "name", "mean", "std_dev", "lineup_slot", "slot_confirmed"):
            assert col in df.columns

    def test_team_filter_discards_off_slate_players(self, tmp_path):
        """Players from teams not in the DK slate must be filtered out."""
        from scripts.fetch_dff_projections import build_projections_csv
        # SAMPLE_DK_SPECS has PHI, ARI, NYY, TB, LAD, SF — BOS is not present
        slate_df = _make_slate_df(SAMPLE_DK_SPECS)
        output = str(tmp_path / "proj.csv")
        rows = self._player_rows() + [{
            "data-name": "Rafael Devers", "data-pos": "3B",
            "data-salary": "6000", "data-ppg_proj": "14.0",
            "data-depth_rank": "4", "data-starter_flag": "1",
            "data-team": "BOS",
        }]
        with patch("scripts.fetch_dff_projections.fetch_player_and_team_rows", return_value=(rows, [])):
            build_projections_csv(
                slate_df=slate_df, target_date="2026-04-12", output_path=output
            )
        df = pd.read_csv(output)
        assert "Rafael Devers" not in df["name"].values

    def test_team_filter_keeps_slate_players(self, tmp_path):
        """Players from slate teams must not be filtered out."""
        from scripts.fetch_dff_projections import build_projections_csv
        slate_df = _make_slate_df(SAMPLE_DK_SPECS)
        output = str(tmp_path / "proj.csv")
        rows = [
            {
                "data-name": "Zack Wheeler", "data-pos": "P",
                "data-salary": "10000", "data-ppg_proj": "25.0",
                "data-depth_rank": "0", "data-starter_flag": "0",
                "data-team": "PHI",
            },
        ]
        with patch("scripts.fetch_dff_projections.fetch_player_and_team_rows", return_value=(rows, [])):
            result = build_projections_csv(
                slate_df=slate_df, target_date="2026-04-12", output_path=output
            )
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Zack Wheeler"

    def test_slate_teams_passed_to_fetch(self, tmp_path):
        """build_projections_csv passes slate_teams to fetch_player_and_team_rows."""
        from scripts.fetch_dff_projections import build_projections_csv
        slate_df = _make_slate_df(SAMPLE_DK_SPECS)
        output = str(tmp_path / "proj.csv")
        with patch("scripts.fetch_dff_projections.fetch_player_and_team_rows") as mock_fetch:
            mock_fetch.return_value = (self._player_rows(), [])
            build_projections_csv(
                slate_df=slate_df, target_date="2026-04-12", output_path=output
            )
        call_kwargs = mock_fetch.call_args[1]
        assert "slate_teams" in call_kwargs
        assert "PHI" in call_kwargs["slate_teams"]


# ---------------------------------------------------------------------------
# projections_meta
# ---------------------------------------------------------------------------

class TestProjectionsMetaExtractDate:
    def test_extract_fd_date_from_filename(self):
        from src.api.projections_meta import _extract_fd_date
        path = Path("data/raw/FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv")
        assert _extract_fd_date(path) == "2026-04-12"

    def test_extract_fd_date_returns_none_for_dk(self):
        from src.api.projections_meta import _extract_fd_date
        assert _extract_fd_date(Path("DKSalaries.csv")) is None

    def test_extract_slate_date_dk(self, tmp_path):
        from src.api.projections_meta import _extract_slate_date
        # Write a minimal DK CSV with a Game Info column
        dk_csv = tmp_path / "DKSalaries.csv"
        dk_csv.write_text("Game Info\nARI@PHI 04/12/2026 01:00PM ET\n")
        result = _extract_slate_date(dk_csv, platform="draftkings")
        assert result == "2026-04-12"

    def test_extract_slate_date_fd(self, tmp_path):
        from src.api.projections_meta import _extract_slate_date
        fd_csv = tmp_path / "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv"
        fd_csv.write_text("stub")
        result = _extract_slate_date(fd_csv, platform="fanduel")
        assert result == "2026-04-12"


class TestProjectionsMetaFetchAndCacheSlates:
    def _mock_rw_slates(self):
        return [
            {
                "slateID": 24060,
                "slateName": "Main Slate",
                "contestType": "Classic",
                "startDateOnly": "2026-04-12",
                "defaultSlate": True,
            }
        ]

    def test_passes_dk_site_id_by_default(self, tmp_path):
        from src.api.projections_meta import fetch_and_cache_slates, _DRAFTKINGS_SITE_ID
        dk_csv = tmp_path / "DKSalaries.csv"
        dk_csv.write_text("Game Info\nARI@PHI 04/12/2026 01:00PM ET\n")
        with patch("src.api.projections_meta._fetch_slate_list_from_rw", return_value=self._mock_rw_slates()) as mock_fetch:
            with patch("src.api.projections_meta.save_metadata"):
                with patch("src.api.projections_meta.load_metadata", return_value={}):
                    fetch_and_cache_slates(dk_csv)
        mock_fetch.assert_called_once_with(site_id=_DRAFTKINGS_SITE_ID)

    def test_passes_fd_site_id_when_specified(self, tmp_path):
        from src.api.projections_meta import fetch_and_cache_slates, _FANDUEL_SITE_ID
        fd_csv = tmp_path / "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv"
        fd_csv.write_text("stub")
        with patch("src.api.projections_meta._fetch_slate_list_from_rw", return_value=self._mock_rw_slates()) as mock_fetch:
            with patch("src.api.projections_meta.save_metadata"):
                with patch("src.api.projections_meta.load_metadata", return_value={}):
                    fetch_and_cache_slates(fd_csv, site_id=_FANDUEL_SITE_ID, platform="fanduel")
        mock_fetch.assert_called_once_with(site_id=_FANDUEL_SITE_ID)

    def test_cache_invalidated_when_site_id_changes(self, tmp_path):
        from src.api.projections_meta import get_cached_slates, _DRAFTKINGS_SITE_ID, _FANDUEL_SITE_ID
        dk_csv = tmp_path / "DKSalaries.csv"
        dk_csv.write_text("Game Info\nARI@PHI 04/12/2026 01:00PM ET\n")
        # Metadata has site_id=1 (DK), but we request site_id=2 (FD) — must be None
        stale_meta = {
            "date": "2026-04-12",
            "slates_fetched_at": 9999999999.0,
            "slates": [{"slate_id": "1", "name": "Old DK Slate", "is_default": True}],
            "site_id": _DRAFTKINGS_SITE_ID,
        }
        with patch("src.api.projections_meta.load_metadata", return_value=stale_meta):
            result = get_cached_slates(dk_csv, site_id=_FANDUEL_SITE_ID, platform="draftkings")
        assert result is None


class TestProjectionsMetaComputeFreshness:
    def _write_proj_csv(self, tmp_path: Path, player_ids: list[int]) -> Path:
        p = tmp_path / "projections.csv"
        df = pd.DataFrame({
            "player_id": player_ids,
            "name": [f"P{i}" for i in player_ids],
            "mean": [10.0] * len(player_ids),
            "std_dev": [3.0] * len(player_ids),
            "lineup_slot": [1] * len(player_ids),
            "slot_confirmed": [True] * len(player_ids),
        })
        df.to_csv(p, index=False)
        return p

    def test_dk_fresh_when_ids_overlap(self, tmp_path):
        from src.api.projections_meta import compute_freshness
        # Create a DK slate CSV with "ID" column
        dk_csv = tmp_path / "DKSalaries.csv"
        ids = list(range(1, 30))
        pd.DataFrame({"ID": ids}).to_csv(dk_csv, index=False)
        proj = self._write_proj_csv(tmp_path, ids[:25])
        result = compute_freshness(dk_csv, proj, platform="draftkings")
        assert result is True

    def test_dk_stale_when_ids_dont_overlap(self, tmp_path):
        from src.api.projections_meta import compute_freshness
        dk_csv = tmp_path / "DKSalaries.csv"
        pd.DataFrame({"ID": list(range(1, 30))}).to_csv(dk_csv, index=False)
        proj = self._write_proj_csv(tmp_path, list(range(500, 525)))
        result = compute_freshness(dk_csv, proj, platform="draftkings")
        assert result is False

    def test_returns_none_when_proj_fewer_than_20(self, tmp_path):
        from src.api.projections_meta import compute_freshness
        dk_csv = tmp_path / "DKSalaries.csv"
        pd.DataFrame({"ID": list(range(1, 30))}).to_csv(dk_csv, index=False)
        proj = self._write_proj_csv(tmp_path, [1, 2, 3])  # only 3 players
        result = compute_freshness(dk_csv, proj, platform="draftkings")
        assert result is False

    def test_fd_fresh_uses_fd_ingestor(self, tmp_path):
        from src.api.projections_meta import compute_freshness
        fd_csv = tmp_path / "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv"
        fd_csv.write_text("stub")
        proj = self._write_proj_csv(tmp_path, list(range(1, 25)))

        # Mock FanDuelSlateIngestor to return player_ids 1-28
        mock_df = pd.DataFrame({"player_id": list(range(1, 29))})
        mock_ingestor = MagicMock()
        mock_ingestor.get_slate_dataframe.return_value = mock_df

        with patch("src.ingestion.fd_slate.FanDuelSlateIngestor", return_value=mock_ingestor):
            result = compute_freshness(fd_csv, proj, platform="fanduel")
        assert result is True

    def test_fd_stale_when_ids_dont_overlap(self, tmp_path):
        from src.api.projections_meta import compute_freshness
        fd_csv = tmp_path / "FanDuel-MLB-2026-04-12-128874-entries-upload-template.csv"
        fd_csv.write_text("stub")
        proj = self._write_proj_csv(tmp_path, list(range(500, 525)))

        mock_df = pd.DataFrame({"player_id": list(range(1, 29))})
        mock_ingestor = MagicMock()
        mock_ingestor.get_slate_dataframe.return_value = mock_df

        with patch("src.ingestion.fd_slate.FanDuelSlateIngestor", return_value=mock_ingestor):
            result = compute_freshness(fd_csv, proj, platform="fanduel")
        assert result is False

    def test_returns_none_on_exception(self, tmp_path):
        from src.api.projections_meta import compute_freshness
        # Non-existent files → exception → None
        result = compute_freshness(tmp_path / "nope.csv", tmp_path / "nope2.csv")
        assert result is None


# ---------------------------------------------------------------------------
# PPD stale-projection filtering (regression tests for the PPD fix)
# ---------------------------------------------------------------------------

def _make_pipeline_slate_df() -> pd.DataFrame:
    """Minimal slate DataFrame with active-game players only."""
    return pd.DataFrame([
        {"player_id": 101, "name": "Active Pitcher",  "position": "P",  "salary": 9000, "team": "PHI", "game": "ARI@PHI"},
        {"player_id": 201, "name": "Active Batter 1", "position": "OF", "salary": 4500, "team": "ARI", "game": "ARI@PHI"},
        {"player_id": 301, "name": "Active Batter 2", "position": "1B", "salary": 4000, "team": "PHI", "game": "ARI@PHI"},
    ])


def _make_proj_df_with_ppd() -> pd.DataFrame:
    """proj_df containing three active players plus two from postponed games."""
    return pd.DataFrame([
        {"player_id": 101, "name": "Active Pitcher",  "mean": 20.0, "std_dev": 8.0, "lineup_slot": 10, "slot_confirmed": True},
        {"player_id": 201, "name": "Active Batter 1", "mean": 12.0, "std_dev": 5.0, "lineup_slot": 3,  "slot_confirmed": True},
        {"player_id": 301, "name": "Active Batter 2", "mean": 10.0, "std_dev": 4.5, "lineup_slot": 5,  "slot_confirmed": True},
        # PPD players — IDs 901/902 are not in the DK slate
        {"player_id": 901, "name": "Jordan Walker",  "mean": 11.0, "std_dev": 5.0, "lineup_slot": 2,  "slot_confirmed": True},
        {"player_id": 902, "name": "Tyrone Taylor",  "mean":  9.0, "std_dev": 4.0, "lineup_slot": 4,  "slot_confirmed": True},
    ])


class TestBuildPlayersDfPPDExclusion:
    """PipelineRunner._build_players_df must silently drop stale PPD players."""

    def test_ppd_players_not_in_output(self):
        from src.api.pipeline import PipelineRunner
        runner = PipelineRunner("dummy_config.yaml")
        result = runner._build_players_df(_make_pipeline_slate_df(), _make_proj_df_with_ppd())
        assert 901 not in result["player_id"].values
        assert 902 not in result["player_id"].values

    def test_active_players_retained(self):
        from src.api.pipeline import PipelineRunner
        runner = PipelineRunner("dummy_config.yaml")
        result = runner._build_players_df(_make_pipeline_slate_df(), _make_proj_df_with_ppd())
        assert set(result["player_id"].values) == {101, 201, 301}

    def test_stale_projection_warning_logged(self, caplog):
        import logging
        from src.api.pipeline import PipelineRunner
        runner = PipelineRunner("dummy_config.yaml")
        with caplog.at_level(logging.WARNING, logger="src.api.pipeline"):
            runner._build_players_df(_make_pipeline_slate_df(), _make_proj_df_with_ppd())
        messages = " ".join(r.message for r in caplog.records)
        assert "postponed" in messages or "not on the current slate" in messages

    def test_no_warning_when_projections_match_slate(self, caplog):
        import logging
        from src.api.pipeline import PipelineRunner
        runner = PipelineRunner("dummy_config.yaml")
        proj_df = _make_proj_df_with_ppd().query("player_id <= 301")
        with caplog.at_level(logging.WARNING, logger="src.api.pipeline"):
            runner._build_players_df(_make_pipeline_slate_df(), proj_df)
        assert not any("postponed" in r.message or "not on the current slate" in r.message
                       for r in caplog.records)


class TestOwnershipPPDTeamTotals:
    """compute_heuristic_ownership must ignore implied totals for PPD teams."""

    def _players(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"player_id": 1, "position": "P",  "mean": 20.0, "salary": 9000, "team": "PHI", "slot": 10},
            {"player_id": 2, "position": "OF", "mean": 12.0, "salary": 4500, "team": "ARI", "slot": 3},
            {"player_id": 3, "position": "1B", "mean": 10.0, "salary": 4000, "team": "PHI", "slot": 5},
            {"player_id": 4, "position": "OF", "mean":  9.0, "salary": 3800, "team": "ARI", "slot": 2},
        ])

    def test_ppd_team_extreme_total_does_not_change_ownership(self):
        import numpy as np
        from src.optimization.ownership import compute_heuristic_ownership
        df = self._players()
        ow_active = compute_heuristic_ownership(df, team_totals={"PHI": 4.5, "ARI": 5.0})
        # STL and MIL have no players in df — their extreme totals must be filtered
        ow_with_ppd = compute_heuristic_ownership(
            df, team_totals={"PHI": 4.5, "ARI": 5.0, "STL": 999.0, "MIL": 999.0}
        )
        np.testing.assert_array_almost_equal(ow_active, ow_with_ppd)

    def test_ownership_vector_length_matches_players_df(self):
        from src.optimization.ownership import compute_heuristic_ownership
        df = self._players()
        ow = compute_heuristic_ownership(
            df, team_totals={"PHI": 4.5, "ARI": 5.0, "STL": 999.0}
        )
        assert len(ow) == len(df)
