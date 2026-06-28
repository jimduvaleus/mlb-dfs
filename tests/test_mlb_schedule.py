"""
Tests for doubleheader detection (src/api/mlb_schedule.py) and the gate it
applies to the Twitter-lineup auto-lock feature in src/api/server.py.

All external I/O (the MLB Stats API, the twitter_lineups.json data file) is
mocked/redirected to a tmp path — no network calls, no mutation of real app
state.
"""

import json
from unittest.mock import MagicMock, patch

import requests

from src.api import mlb_schedule
from src.api import twitter_lineups


def _schedule_payload(games: list[dict]) -> dict:
    """Build a minimal MLB Stats API schedule response from (away, home, game_number) tuples."""
    return {
        "dates": [{
            "games": [
                {
                    "teams": {
                        "away": {"team": {"name": g["away"]}},
                        "home": {"team": {"name": g["home"]}},
                    },
                    "gameNumber": g.get("game_number", 1),
                    "doubleHeader": g.get("double_header", "N"),
                    "gameDate": g.get("game_date", "2026-06-24T17:10:00Z"),
                }
                for g in games
            ]
        }]
    }


DOUBLEHEADER_PAYLOAD = _schedule_payload([
    {"away": "Chicago Cubs", "home": "New York Mets", "game_number": 1, "double_header": "S", "game_date": "2026-06-24T17:10:00Z"},
    {"away": "Chicago Cubs", "home": "New York Mets", "game_number": 2, "double_header": "S", "game_date": "2026-06-24T23:10:00Z"},
    {"away": "Texas Rangers", "home": "Miami Marlins", "game_number": 1, "double_header": "N", "game_date": "2026-06-24T16:10:00Z"},
])


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# fetch_schedule / get_doubleheader_teams_cached
# ---------------------------------------------------------------------------

class TestFetchSchedule:
    def test_detects_doubleheader_teams_via_full_name_map(self):
        with patch.object(mlb_schedule.requests, "get", return_value=_mock_response(DOUBLEHEADER_PAYLOAD)):
            data = mlb_schedule.fetch_schedule("2026-06-24")
        assert data["doubleheader_teams"] == ["CHC", "NYM"]
        assert data["date"] == "2026-06-24"
        assert len(data["games"]) == 3

    def test_no_doubleheaders_returns_empty_list(self):
        payload = _schedule_payload([{"away": "Texas Rangers", "home": "Miami Marlins"}])
        with patch.object(mlb_schedule.requests, "get", return_value=_mock_response(payload)):
            data = mlb_schedule.fetch_schedule("2026-06-24")
        assert data["doubleheader_teams"] == []


class TestGetDoubleheaderTeamsCached:
    def test_fetches_and_caches_on_first_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mlb_schedule, "SCHEDULE_PATH", tmp_path / "mlb_schedule.json")
        with patch.object(mlb_schedule.requests, "get", return_value=_mock_response(DOUBLEHEADER_PAYLOAD)) as mock_get:
            teams, is_fresh = mlb_schedule.get_doubleheader_teams_cached("2026-06-24")
        assert teams == {"CHC", "NYM"}
        assert is_fresh is True
        mock_get.assert_called_once()

        cached = json.loads((tmp_path / "mlb_schedule.json").read_text())
        assert cached["date"] == "2026-06-24"
        assert sorted(cached["doubleheader_teams"]) == ["CHC", "NYM"]

    def test_uses_cache_without_refetching_same_date(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "mlb_schedule.json"
        cache_path.write_text(json.dumps({
            "date": "2026-06-24",
            "fetched_at": 0,
            "games": [],
            "doubleheader_teams": ["CHC", "NYM"],
        }))
        monkeypatch.setattr(mlb_schedule, "SCHEDULE_PATH", cache_path)
        with patch.object(mlb_schedule.requests, "get", side_effect=AssertionError("should not refetch")):
            teams, is_fresh = mlb_schedule.get_doubleheader_teams_cached("2026-06-24")
        assert teams == {"CHC", "NYM"}
        assert is_fresh is True

    def test_refetches_when_cached_date_differs(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "mlb_schedule.json"
        cache_path.write_text(json.dumps({
            "date": "2026-06-23", "fetched_at": 0, "games": [], "doubleheader_teams": ["CHC", "NYM"],
        }))
        monkeypatch.setattr(mlb_schedule, "SCHEDULE_PATH", cache_path)
        payload = _schedule_payload([{"away": "Texas Rangers", "home": "Miami Marlins"}])
        with patch.object(mlb_schedule.requests, "get", return_value=_mock_response(payload)):
            teams, is_fresh = mlb_schedule.get_doubleheader_teams_cached("2026-06-24")
        assert teams == set()
        assert is_fresh is True

    def test_fails_open_on_fetch_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mlb_schedule, "SCHEDULE_PATH", tmp_path / "mlb_schedule.json")
        with patch.object(mlb_schedule.requests, "get", side_effect=requests.exceptions.ConnectionError("down")):
            teams, is_fresh = mlb_schedule.get_doubleheader_teams_cached("2026-06-24")
        assert teams == set()
        assert is_fresh is False


# ---------------------------------------------------------------------------
# Server-side gate: POST /api/twitter-lineups vetoes auto-lock for
# doubleheader teams. Calls the route function directly (it's a plain async
# function) rather than spinning up an HTTP client.
# ---------------------------------------------------------------------------

class TestTwitterLineupLockGate:
    def _save(self, team: str):
        from src.api import server
        from src.api.models import TwitterLineupSaveRequest, TwitterLineupSlot

        req = TwitterLineupSaveRequest(
            team=team,
            notification_id="test-notif",
            slots=[TwitterLineupSlot(slot=1, player_id=1, name="Test Player")],
            locked=True,
        )
        return server.save_twitter_lineup(req)

    def test_doubleheader_team_save_is_not_locked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(twitter_lineups, "_DATA_PATH", tmp_path / "twitter_lineups.json")
        with patch("src.api.server.get_doubleheader_teams_cached", return_value=({"NYM", "CHC"}, True)):
            record = self._save("NYM")
        assert record.locked is False
        assert record.needs_game_confirmation is True

    def test_non_doubleheader_team_save_locks_normally(self, tmp_path, monkeypatch):
        monkeypatch.setattr(twitter_lineups, "_DATA_PATH", tmp_path / "twitter_lineups.json")
        with patch("src.api.server.get_doubleheader_teams_cached", return_value=({"NYM", "CHC"}, True)):
            record = self._save("LAD")
        assert record.locked is True
        assert record.needs_game_confirmation is False
