"""
Tests for the ownership evaluation/tuning improvements:

  - leverage-aware metrics (log_rmse, calibration slope, field-points bias)
  - paired statistical comparison (bootstrap CI, baseline deltas)
  - provenance helpers and append-mode summary
  - W_resid isotonic residual calibrator (PAVA)
  - walk-forward split/seeding/full-model scoring helpers
"""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

import src.optimization.ownership as own_mod
from scripts.evaluate_ownership import (
    _LOG_EPS,
    _append_summary,
    _bootstrap_delta_ci,
    _calibration_metrics,
    _collect_production_constants,
    _compute_model_w,
    _evaluate,
    _field_points_bias,
    _git_commit_hash,
    _paired_comparison_table,
    _pava,
    _slate_sort_key,
    fit_residual_calibrator,
)
from scripts.optimize_ownership_params import PARAM_BOUNDS, PARAM_NAMES, _current_params
from scripts.walk_forward_ownership import (
    _make_walkforward_splits,
    _score_full_model,
    _seeded_init,
)


# ---------------------------------------------------------------------------
# Piece 2 — metrics
# ---------------------------------------------------------------------------

class TestLogRmse:
    def test_log_rmse_known_values(self):
        actual = np.array([0.10, 0.20, 0.05, 0.30, 0.0])
        predicted = np.array([0.20, 0.10, 0.05, 0.30, 0.0])
        metrics = _evaluate(actual, predicted)
        expected = np.sqrt(np.mean(
            (np.log(predicted + _LOG_EPS) - np.log(actual + _LOG_EPS)) ** 2
        ))
        assert metrics["log_rmse"] == pytest.approx(expected, abs=1e-4)

    def test_log_rmse_zero_ownership_is_finite(self):
        actual = np.array([0.0, 0.0, 0.1, 0.2, 0.3])
        predicted = np.array([0.05, 0.0, 0.1, 0.2, 0.3])
        metrics = _evaluate(actual, predicted)
        assert np.isfinite(metrics["log_rmse"])

    def test_perfect_prediction_zero_error(self):
        actual = np.array([0.05, 0.10, 0.20, 0.30, 0.40])
        metrics = _evaluate(actual, actual.copy())
        assert metrics["log_rmse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["spearman_r"] == 1.0


class TestCalibrationMetrics:
    def _positions(self, n_p, n_b):
        return np.array(["P"] * n_p + ["OF"] * n_b)

    def test_identity_recovers_slope_one(self):
        rng = np.random.default_rng(0)
        actual = rng.uniform(0.01, 0.5, size=40)
        positions = self._positions(10, 30)
        out = _calibration_metrics(actual, actual.copy(), positions)
        assert out["calib_slope_P"] == pytest.approx(1.0, abs=1e-6)
        assert out["calib_slope_bat"] == pytest.approx(1.0, abs=1e-6)
        assert out["calib_int_P"] == pytest.approx(0.0, abs=1e-6)
        assert out["calib_int_bat"] == pytest.approx(0.0, abs=1e-6)

    def test_known_power_recovered(self):
        # actual = predicted^1.5 in (x + eps) log space → slope == 1.5 exactly.
        rng = np.random.default_rng(1)
        predicted = rng.uniform(0.02, 0.6, size=50)
        actual = (predicted + _LOG_EPS) ** 1.5 - _LOG_EPS
        positions = np.array(["OF"] * 50)
        out = _calibration_metrics(actual, predicted, positions)
        assert out["calib_slope_bat"] == pytest.approx(1.5, abs=0.01)
        assert np.isnan(out["calib_slope_P"])  # no pitchers

    def test_too_few_points_returns_nan(self):
        actual = np.array([0.1, 0.2, 0.3])
        positions = np.array(["P", "P", "P"])
        out = _calibration_metrics(actual, actual.copy(), positions)
        assert np.isnan(out["calib_slope_P"])


class TestFieldPointsBias:
    def test_exact_values(self):
        merged = pd.DataFrame({
            "pct_drafted": [0.5, 0.2, 0.1, 0.4],
            "actual_fpts": [10.0, 20.0, 5.0, np.nan],
        })
        predicted = np.array([0.6, 0.1, 0.1, 0.9])
        standings = pd.DataFrame({"points": [100.0, 120.0, 80.0]})

        out = _field_points_bias(merged, predicted, standings)
        # NaN-fpts player excluded from both sums.
        assert out["pred_field_pts"] == pytest.approx(0.6 * 10 + 0.1 * 20 + 0.1 * 5)
        assert out["matched_field_pts"] == pytest.approx(0.5 * 10 + 0.2 * 20 + 0.1 * 5)
        assert out["contest_mean_pts"] == pytest.approx(100.0)
        assert out["field_pts_bias"] == pytest.approx(
            out["pred_field_pts"] - out["matched_field_pts"]
        )
        assert out["field_pts_coverage"] == pytest.approx(
            out["matched_field_pts"] / 100.0, abs=1e-4
        )


class TestPairedStats:
    def test_bootstrap_ci_excludes_zero_for_constant_positive(self):
        deltas = np.full(10, 0.02)
        mean, lo, hi = _bootstrap_delta_ci(deltas)
        assert mean == pytest.approx(0.02)
        assert lo > 0

    def test_bootstrap_ci_straddles_zero_for_symmetric(self):
        deltas = np.array([-0.03, 0.03, -0.02, 0.02, -0.01, 0.01, 0.0, 0.0])
        mean, lo, hi = _bootstrap_delta_ci(deltas, seed=0)
        assert lo < 0 < hi

    def test_paired_table_baseline_deltas_zero(self):
        combined = pd.DataFrame({
            "slate": ["s1", "s1", "s2", "s2"],
            "model": ["E_production", "X", "E_production", "X"],
            "spearman_r": [0.8, 0.8, 0.7, 0.7],
            "rmse": [0.03, 0.03, 0.04, 0.04],
            "log_rmse": [0.5, 0.5, 0.6, 0.6],
        })
        table = _paired_comparison_table(combined)
        assert (table["model"] == "X").all()
        assert (table["mean_delta"] == 0).all()
        assert not table["sig"].any()

    def test_paired_table_sign_flip_for_rmse(self):
        # X has LOWER rmse (better) → positive delta after sign flip.
        combined = pd.DataFrame({
            "slate": ["s1", "s1", "s2", "s2"],
            "model": ["E_production", "X", "E_production", "X"],
            "spearman_r": [0.8, 0.8, 0.7, 0.7],
            "rmse": [0.04, 0.03, 0.05, 0.04],
            "log_rmse": [0.6, 0.5, 0.7, 0.6],
        })
        table = _paired_comparison_table(combined)
        rmse_rows = table[table["metric"] == "rmse"]
        assert (rmse_rows["mean_delta"] > 0).all()


# ---------------------------------------------------------------------------
# Piece 4 — provenance & append-mode summary
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_collect_constants_contains_tunables(self):
        constants = _collect_production_constants()
        for name in PARAM_NAMES:
            assert name in constants, f"missing tunable {name}"
        assert "_BATTER_CALIB_EXP" in constants
        assert all(isinstance(v, float) for v in constants.values())
        # Dict-valued tunables are excluded.
        assert "_SLOT_COUNTS" not in constants
        assert "_BATTING_ORDER_MULT" not in constants

    def test_git_hash_fallback(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise OSError("no git")
        monkeypatch.setattr(subprocess, "run", _boom)
        assert _git_commit_hash() == "unknown"

    def test_append_summary_grows(self, tmp_path):
        path = tmp_path / "summary.csv"
        combined = pd.DataFrame({
            "slate": ["s1"], "model": ["E_production"], "spearman_r": [0.8],
        })
        _append_summary(combined, path)
        _append_summary(combined, path)
        out = pd.read_csv(path)
        assert len(out) == 2
        assert {"run_ts", "git_commit", "constants_hash", "constants_json"} <= set(out.columns)
        constants = json.loads(out["constants_json"].iloc[0])
        assert "_PITCHER_MATCHUP_EXP" in constants

    def test_append_summary_legacy_schema(self, tmp_path):
        path = tmp_path / "summary.csv"
        legacy = pd.DataFrame({"slate": ["old"], "model": ["E_production"], "rmse": [0.03]})
        legacy.to_csv(path, index=False)
        combined = pd.DataFrame({
            "slate": ["s1"], "model": ["E_production"], "spearman_r": [0.8],
        })
        _append_summary(combined, path)
        out = pd.read_csv(path)
        assert len(out) == 2
        assert out["run_ts"].isna().iloc[0]       # legacy row NaN-filled
        assert not pd.isna(out["run_ts"].iloc[1])

    def test_slate_sort_key_year_boundary(self):
        # Lexical: "01022027" < "12312026" — date key must order correctly.
        assert _slate_sort_key("12312026") < _slate_sort_key("01022027")
        assert _slate_sort_key("05252026") < _slate_sort_key("05252026e")


# ---------------------------------------------------------------------------
# Piece 3 — PAVA / W_resid
# ---------------------------------------------------------------------------

def _make_synthetic_pool(seed: int = 7) -> pd.DataFrame:
    """4-team, 2-game pool: 1 P + 8 batters per team across DK positions."""
    rng = np.random.default_rng(seed)
    rows = []
    pid = 1
    games = {"NYY": "NYY@BOS", "BOS": "NYY@BOS", "LAD": "LAD@SF", "SF": "LAD@SF"}
    opps = {"NYY": "BOS", "BOS": "NYY", "LAD": "SF", "SF": "LAD"}
    batter_positions = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
    for team in ("NYY", "BOS", "LAD", "SF"):
        rows.append({
            "player_id": pid, "position": "P", "team": team, "opponent": opps[team],
            "mean": float(rng.uniform(12, 22)), "salary": int(rng.uniform(7000, 10000)),
            "game": games[team], "slot": np.nan,
        })
        pid += 1
        for i, pos in enumerate(batter_positions):
            rows.append({
                "player_id": pid, "position": pos, "team": team, "opponent": opps[team],
                "mean": float(rng.uniform(5, 12)), "salary": int(rng.uniform(2500, 6000)),
                "game": games[team], "slot": float(i + 1),
            })
            pid += 1
    return pd.DataFrame(rows)


class TestPava:
    def test_pools_violators(self):
        np.testing.assert_allclose(_pava(np.array([1.0, 3.0, 2.0])), [1.0, 2.5, 2.5])

    def test_monotone_on_random_input(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=200)
        fitted = _pava(y)
        assert np.all(np.diff(fitted) >= -1e-12)

    def test_already_monotone_unchanged(self):
        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(_pava(y), y)


class TestResidualCalibrator:
    def _write_fake_slate(self, d, n=60, seed=0):
        rng = np.random.default_rng(seed)
        pred = rng.uniform(0.01, 0.6, size=n)
        positions = np.array(["P"] * 15 + ["OF"] * (n - 15))
        df = pd.DataFrame({
            "position": positions,
            "pred_E_production": pred,
            "pct_drafted": np.clip(pred * 1.2 + rng.normal(0, 0.02, n), 0, 1),
        })
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / "ownership_eval.csv", index=False)

    def test_skips_below_min_slates(self, tmp_path):
        dirs = [tmp_path / f"slate{i}" for i in range(2)]
        for i, d in enumerate(dirs):
            self._write_fake_slate(d, seed=i)
        assert fit_residual_calibrator(dirs, min_slates=5) is None

    def test_fits_with_enough_slates(self, tmp_path):
        dirs = [tmp_path / f"slate{i}" for i in range(5)]
        for i, d in enumerate(dirs):
            self._write_fake_slate(d, seed=i)
        cal = fit_residual_calibrator(dirs, min_slates=5)
        assert cal is not None
        assert cal["n_slates"] == 5
        assert "P" in cal and "bat" in cal
        x, y = cal["bat"]
        assert np.all(np.diff(x) > 0)            # unique, sorted knots
        assert np.all(np.diff(y) >= -1e-12)      # monotone fit

    def test_model_w_preserves_slot_sums_and_ranks(self, tmp_path):
        dirs = [tmp_path / f"slate{i}" for i in range(5)]
        for i, d in enumerate(dirs):
            self._write_fake_slate(d, seed=i)
        cal = fit_residual_calibrator(dirs, min_slates=5)

        pool = _make_synthetic_pool()
        team_totals = {"NYY": 5.2, "BOS": 4.1, "LAD": 4.8, "SF": 3.9}
        result = _compute_model_w(pool, team_totals, cal)

        from src.optimization.ownership import compute_heuristic_ownership, _SLOT_COUNTS
        base = compute_heuristic_ownership(pool, team_totals)
        positions = pool["position"].values
        for pos, n_slots in _SLOT_COUNTS.items():
            mask = positions == pos
            if not mask.any():
                continue
            assert result[mask].sum() == pytest.approx(n_slots, abs=1e-6)
            # Monotone calibration never inverts within-group rank order.
            order = np.argsort(base[mask])
            assert np.all(np.diff(result[mask][order]) >= -1e-12)


# ---------------------------------------------------------------------------
# Production calibration (apply / load / staleness)
# ---------------------------------------------------------------------------

class TestProductionCalibration:
    def _artifact(self, constants_hash=None):
        from src.optimization.ownership import ownership_constants_hash
        return {
            "fitted_at": "2026-06-11T12:00:00",
            "git_commit": "abc1234",
            "constants_hash": constants_hash or ownership_constants_hash(),
            "n_slates": 7,
            "slates": ["s1"],
            "groups": {
                "P":   {"x": [0.05, 0.40, 0.80], "y": [0.03, 0.45, 0.85]},
                "bat": {"x": [0.01, 0.20, 0.60], "y": [0.02, 0.18, 0.65]},
            },
        }

    def test_loader_missing_file_returns_none(self, tmp_path):
        from src.optimization.ownership import load_ownership_calibrator
        assert load_ownership_calibrator(tmp_path / "nope.json") is None

    def test_loader_roundtrip(self, tmp_path):
        from src.optimization.ownership import load_ownership_calibrator
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(self._artifact()))
        cal = load_ownership_calibrator(path)
        assert cal is not None
        np.testing.assert_allclose(cal["P"][0], [0.05, 0.40, 0.80])
        np.testing.assert_allclose(cal["bat"][1], [0.02, 0.18, 0.65])
        assert cal["n_slates"] == 7

    def test_loader_rejects_stale_constants_hash(self, tmp_path):
        from src.optimization.ownership import load_ownership_calibrator
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(self._artifact(constants_hash="deadbeef00")))
        assert load_ownership_calibrator(path) is None
        assert load_ownership_calibrator(path, check_constants_hash=False) is not None

    def test_loader_rejects_nonmonotone_knots(self, tmp_path):
        from src.optimization.ownership import load_ownership_calibrator
        artifact = self._artifact()
        artifact["groups"]["bat"]["y"] = [0.30, 0.18, 0.65]  # not non-decreasing
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(artifact))
        assert load_ownership_calibrator(path) is None

    def test_apply_preserves_slot_sums_and_identity_without_groups(self):
        from src.optimization.ownership import (
            _SLOT_COUNTS,
            apply_ownership_calibration,
            compute_heuristic_ownership,
        )
        pool = _make_synthetic_pool()
        team_totals = {"NYY": 5.2, "BOS": 4.1, "LAD": 4.8, "SF": 3.9}
        base = compute_heuristic_ownership(pool, team_totals)
        positions = pool["position"].values

        # No fitted groups → renormalization only, which is a no-op on
        # already-normalized output.
        ident = apply_ownership_calibration(base, positions, {})
        np.testing.assert_allclose(ident, base, atol=1e-9)

        cal = {
            "P":   (np.array([0.0, 1.0]), np.array([0.0, 0.9])),
            "bat": (np.array([0.0, 1.0]), np.array([0.01, 0.8])),
        }
        result = apply_ownership_calibration(base, positions, cal)
        for pos, n_slots in _SLOT_COUNTS.items():
            mask = positions == pos
            if mask.any():
                assert result[mask].sum() == pytest.approx(n_slots, abs=1e-6)

    def test_constants_hash_matches_eval_summary_hash(self):
        # The summary's constants_hash gates calibrator staleness — the two
        # introspection paths must agree.
        from src.optimization.ownership import collect_ownership_constants
        assert _collect_production_constants() == collect_ownership_constants()


# ---------------------------------------------------------------------------
# Piece 1 — walk-forward helpers
# ---------------------------------------------------------------------------

class TestWalkForward:
    def test_split_generator(self):
        slates = list(range(8))
        splits = _make_walkforward_splits(slates, min_train=5)
        assert len(splits) == 3
        for train, test in splits:
            assert test not in train
            assert all(t < test for t in train)   # strictly older
        assert splits[0] == ([0, 1, 2, 3, 4], 5)

    def test_seeded_init_shape_and_bounds(self):
        popsize = 8
        init = _seeded_init(None, popsize, seed=0)
        n_params = len(PARAM_NAMES)
        assert init.shape == (popsize * n_params, n_params)
        np.testing.assert_allclose(init[0], _current_params())
        lo = np.array([b[0] for b in PARAM_BOUNDS])
        hi = np.array([b[1] for b in PARAM_BOUNDS])
        assert np.all(init >= lo - 1e-12) and np.all(init <= hi + 1e-12)

    def test_seeded_init_includes_prev_best(self):
        prev = _current_params() * 1.01
        init = _seeded_init(prev, 8, seed=0)
        lo = np.array([b[0] for b in PARAM_BOUNDS])
        hi = np.array([b[1] for b in PARAM_BOUNDS])
        np.testing.assert_allclose(init[1], np.clip(prev, lo, hi))

    def test_score_full_model_restores_constants(self):
        pool = _make_synthetic_pool()
        team_totals = {"NYY": 5.2, "BOS": 4.1, "LAD": 4.8, "SF": 3.9}
        matched = pd.DataFrame({
            "player_id": pool["player_id"].values,
            "pct_drafted": np.random.default_rng(5).uniform(0.01, 0.5, len(pool)),
        })
        before = {name: getattr(own_mod, name) for name in PARAM_NAMES}
        perturbed = _current_params() * 1.1
        metrics = _score_full_model(perturbed, pool, matched, team_totals)
        after = {name: getattr(own_mod, name) for name in PARAM_NAMES}
        assert before == after
        assert np.isfinite(metrics["spearman_r"])
        assert np.isfinite(metrics["log_rmse"])
