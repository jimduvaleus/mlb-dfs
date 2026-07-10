"""Regression tests for the GPP candidate/field lineup disk cache.

Candidate pools are generated under a specific optimizer.salary_floor (the
floor is baked into which lineups make it into the pool via
CandidateGenerator). If a cached pool were reused after the configured
floor changed, the new floor would be silently ignored — see the
salary_floor mismatch handling in save_candidates/load_candidates.
"""
import numpy as np
import pytest

from src.api import lineup_cache as lc
from src.optimization.lineup import Lineup


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "CACHE_DIR", tmp_path)
    yield tmp_path


def _make_candidates(n=5):
    return [Lineup(player_ids=list(range(i, i + 10))) for i in range(n)]


def test_load_candidates_hits_with_matching_salary_floor():
    fp = "fp1:100"
    cands = _make_candidates()
    lc.save_candidates(fp, cands, salary_floor=48500.0)

    loaded = lc.load_candidates(fp, salary_floor=48500.0)

    assert loaded is not None
    assert len(loaded) == len(cands)


def test_load_candidates_misses_on_salary_floor_change():
    fp = "fp1:100"
    cands = _make_candidates()
    lc.save_candidates(fp, cands, salary_floor=48500.0)

    # Configured floor changed since the cache was written -> must not
    # silently reuse the pool generated under the old floor.
    loaded = lc.load_candidates(fp, salary_floor=46000.0)

    assert loaded is None


def test_load_candidates_misses_when_floor_removed_or_added():
    fp = "fp1:100"
    cands = _make_candidates()
    lc.save_candidates(fp, cands, salary_floor=48500.0)

    assert lc.load_candidates(fp, salary_floor=None) is None

    fp2 = "fp2:100"
    lc.save_candidates(fp2, cands, salary_floor=None)
    assert lc.load_candidates(fp2, salary_floor=48500.0) is None
    assert lc.load_candidates(fp2, salary_floor=None) is not None


def test_get_cache_status_reports_unavailable_on_floor_mismatch(tmp_path):
    slate = tmp_path / "slate.csv"
    slate.write_text("dummy")
    fp = lc.get_cache_status(slate)["fingerprint"]

    cands = _make_candidates()
    lc.save_candidates(fp, cands, salary_floor=48500.0)

    status_match = lc.get_cache_status(slate, salary_floor=48500.0)
    assert status_match["candidates"] == len(cands)

    status_mismatch = lc.get_cache_status(slate, salary_floor=40000.0)
    assert status_mismatch["candidates"] is None
