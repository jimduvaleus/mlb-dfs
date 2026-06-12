"""
Walk-forward (out-of-sample) validation of ownership model constants.

optimize_ownership_params.py tunes the 11 ownership constants by maximizing
mean Spearman over ALL given slates — purely in-sample.  With 11 free
parameters and ~20 slates that risks capturing noise: its latest run reported
+0.04 Spearman with _PITCHER_MATCHUP_EXP ≈ 0.95 (production: 0.40), and this
script exists to answer whether such gains survive out-of-sample.

For each fold k (expanding window): tune on slates strictly older than the
test slate via differential evolution, then score the held-out slate with
both the tuned parameters and the current production constants.  Tuning uses
the fast NumPy objective; the headline OOS comparison is scored through the
FULL compute_heuristic_ownership model (the fast objective omits calibration
exponents, ownership caps, and the pitcher-opposition pass).

Usage
-----
    # Default budget (~15 min for 21 slates): popsize 8, maxiter 80
    python scripts/walk_forward_ownership.py

    # Thorough confirmation run (~2 h)
    python scripts/walk_forward_ownership.py --full

    # Restrict slates / change fold start
    python scripts/walk_forward_ownership.py --recent 12 --min-train 5

Output
------
  Per-fold OOS table, parameter-stability table, paired-stats verdict.
  Writes archive/ownership_walkforward_results.json.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_ownership import (  # noqa: E402
    _bootstrap_delta_ci,
    _collect_production_constants,
    _evaluate,
    _find_recent_full_slates,
    _git_commit_hash,
)
from optimize_ownership_params import (  # noqa: E402
    PARAM_BOUNDS,
    PARAM_NAMES,
    _current_params,
    _load_slates_with_frames,
    _score_slates,
    run_de,
)
import src.optimization.ownership as _own_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------

def _parse_slate_date(name: str) -> datetime:
    """Parse an archive dir name like '05252026' or '05252026e' to a date.

    Date-based (not lexical) ordering keeps walk-forward splits correct
    across a year boundary.
    """
    return datetime.strptime(name[:8], "%m%d%Y")


def _make_walkforward_splits(slates: list, min_train: int = 5) -> list[tuple[list, object]]:
    """
    Expanding-window splits over slates already sorted oldest-first:
    fold k trains on slates[:k] and tests on slates[k], k >= min_train.
    """
    return [(slates[:k], slates[k]) for k in range(min_train, len(slates))]


def _seeded_init(prev_best: np.ndarray | None, popsize: int, seed: int) -> np.ndarray:
    """
    Build a DE init population (popsize × n_params rows): current production
    constants, the previous fold's optimum, the frozen full-sample DE optimum
    from archive/ownership_regression_results.json when available, and random
    uniform fill within PARAM_BOUNDS.  All rows clipped to bounds.
    """
    n_params = len(PARAM_NAMES)
    n_pop = max(popsize * n_params, 5)
    lo = np.array([b[0] for b in PARAM_BOUNDS])
    hi = np.array([b[1] for b in PARAM_BOUNDS])

    rng = np.random.default_rng(seed)
    init = lo + rng.random((n_pop, n_params)) * (hi - lo)

    seeds = [_current_params()]
    if prev_best is not None:
        seeds.append(np.asarray(prev_best, dtype=float))
    frozen_path = PROJECT_ROOT / "archive" / "ownership_regression_results.json"
    if frozen_path.exists():
        try:
            frozen = json.loads(frozen_path.read_text())["params"]
            seeds.append(np.array([frozen[name] for name in PARAM_NAMES]))
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    for i, vec in enumerate(seeds[:n_pop]):
        init[i] = vec
    return np.clip(init, lo, hi)


# ---------------------------------------------------------------------------
# Full-model scoring
# ---------------------------------------------------------------------------

def _score_full_model(
    params_vec: np.ndarray,
    pool_df,
    matched_df,
    team_totals: dict | None,
) -> dict:
    """
    Score a parameter vector through the full compute_heuristic_ownership
    model on one slate's matched players.

    Temporarily patches the tunable module constants (the _compute_model_k
    pattern from evaluate_ownership.py) and restores them in a finally block.
    Returns the _evaluate metric dict (spearman_r, rmse, log_rmse, ...).
    """
    old_vals = {name: getattr(_own_mod, name) for name in PARAM_NAMES}
    for name, val in zip(PARAM_NAMES, params_vec):
        setattr(_own_mod, name, float(val))
    try:
        ownership = _own_mod.compute_heuristic_ownership(pool_df, team_totals)
    finally:
        for name, old_val in old_vals.items():
            setattr(_own_mod, name, old_val)

    pid_map = dict(zip(pool_df["player_id"], ownership))
    predicted = matched_df["player_id"].map(pid_map).values.astype(float)
    actual = matched_df["pct_drafted"].values.astype(float)
    return _evaluate(actual, predicted)


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward OOS validation of ownership model constants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("archive_dirs", nargs="*", metavar="ARCHIVE_DIR",
                        help="Specific archive directories (default: all full slates).")
    parser.add_argument("--recent", type=int, default=0, metavar="N",
                        help="Use the N most recent full slates.")
    parser.add_argument("--min-train", type=int, default=5,
                        help="Minimum training slates before the first fold (default 5).")
    parser.add_argument("--popsize", type=int, default=8,
                        help="DE population multiplier per parameter (default 8).")
    parser.add_argument("--maxiter", type=int, default=80,
                        help="DE maximum iterations per fold (default 80).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument("--full", action="store_true",
                        help="Thorough run: popsize 15, maxiter 300 (~2 h).")
    args = parser.parse_args()

    if args.recent and args.archive_dirs:
        parser.error("--recent and positional ARCHIVE_DIR are mutually exclusive.")
    popsize = 15 if args.full else args.popsize
    maxiter = 300 if args.full else args.maxiter

    if args.recent:
        dirs = _find_recent_full_slates(args.recent)
    elif args.archive_dirs:
        dirs = [Path(d) for d in args.archive_dirs if Path(d).is_dir()]
    else:
        dirs = _find_recent_full_slates(999)
    if not dirs:
        print("No archive directories found.")
        sys.exit(1)

    print(f"Loading and pre-extracting {len(dirs)} slate(s)...", flush=True)
    loaded = _load_slates_with_frames(dirs)
    if len(loaded) <= args.min_train:
        print(f"Need more than {args.min_train} usable slates "
              f"(got {len(loaded)}) for walk-forward folds.")
        sys.exit(1)
    loaded.sort(key=lambda item: _parse_slate_date(item[0].label))

    splits = _make_walkforward_splits(loaded, min_train=args.min_train)
    prod_params = _current_params()
    print(f"\n{len(loaded)} slate(s) ready → {len(splits)} walk-forward fold(s); "
          f"DE popsize={popsize}, maxiter={maxiter}\n", flush=True)

    folds: list[dict] = []
    prev_best: np.ndarray | None = None
    for fold_i, (train_items, test_item) in enumerate(splits):
        train_slates = [item[0] for item in train_items]
        test_sa, test_pool, test_matched, test_totals = test_item

        print(f"[fold {fold_i + 1}/{len(splits)}] test={test_sa.label} "
              f"train={len(train_slates)} slate(s) ... ", end="", flush=True)

        tuned_params, train_score, _ = run_de(
            train_slates,
            popsize=popsize,
            maxiter=maxiter,
            seed=args.seed + fold_i,
            polish=True,
            init=_seeded_init(prev_best, popsize, seed=args.seed + fold_i),
            tol=1e-3,
            verbose=False,
        )
        prev_best = tuned_params

        # Fast-objective OOS scores (apples-to-apples with the DE objective)
        _, prod_fast = _score_slates(prod_params, [test_sa])
        _, tuned_fast = _score_slates(tuned_params, [test_sa])

        # Full-model OOS scores — the headline comparison
        prod_full  = _score_full_model(prod_params, test_pool, test_matched, test_totals)
        tuned_full = _score_full_model(tuned_params, test_pool, test_matched, test_totals)

        fold = {
            "test_slate": test_sa.label,
            "n_train": len(train_slates),
            "train_spearman_tuned": round(float(train_score), 4),
            "fast_prod": round(prod_fast.get(test_sa.label, float("nan")), 4),
            "fast_tuned": round(tuned_fast.get(test_sa.label, float("nan")), 4),
            "full_prod_spearman": prod_full["spearman_r"],
            "full_tuned_spearman": tuned_full["spearman_r"],
            "full_prod_log_rmse": prod_full["log_rmse"],
            "full_tuned_log_rmse": tuned_full["log_rmse"],
            "tuned_params": {n: round(float(v), 4) for n, v in zip(PARAM_NAMES, tuned_params)},
        }
        folds.append(fold)
        print(f"OOS spearman prod={fold['full_prod_spearman']:.4f} "
              f"tuned={fold['full_tuned_spearman']:.4f} "
              f"Δ={fold['full_tuned_spearman'] - fold['full_prod_spearman']:+.4f}", flush=True)

    # ── Per-fold table ────────────────────────────────────────────────────────
    W = 96
    print(f"\n{'=' * W}")
    print("PER-FOLD OUT-OF-SAMPLE RESULTS (full model)")
    print(f"{'=' * W}")
    print(f"{'Test slate':<12} {'n_tr':>4} {'train(t)':>8} "
          f"{'fast_p':>7} {'fast_t':>7} {'full_p':>7} {'full_t':>7} {'Δfull':>8} "
          f"{'logrmse_p':>9} {'logrmse_t':>9}")
    print("-" * W)
    for f in folds:
        d = f["full_tuned_spearman"] - f["full_prod_spearman"]
        print(f"{f['test_slate']:<12} {f['n_train']:>4} {f['train_spearman_tuned']:>8.4f} "
              f"{f['fast_prod']:>7.4f} {f['fast_tuned']:>7.4f} "
              f"{f['full_prod_spearman']:>7.4f} {f['full_tuned_spearman']:>7.4f} {d:>+8.4f} "
              f"{f['full_prod_log_rmse']:>9.4f} {f['full_tuned_log_rmse']:>9.4f}")

    # ── Paired stats on full-model deltas ─────────────────────────────────────
    deltas_sp = np.array(
        [f["full_tuned_spearman"] - f["full_prod_spearman"] for f in folds], dtype=float
    )
    deltas_lr = np.array(
        [f["full_prod_log_rmse"] - f["full_tuned_log_rmse"] for f in folds], dtype=float
    )  # sign-flipped: + = tuned better
    fast_deltas = np.array([f["fast_tuned"] - f["fast_prod"] for f in folds], dtype=float)

    mean_sp, lo_sp, hi_sp = _bootstrap_delta_ci(deltas_sp)
    mean_lr, lo_lr, hi_lr = _bootstrap_delta_ci(deltas_lr)
    p_sp = np.nan
    valid = deltas_sp[np.isfinite(deltas_sp)]
    if len(valid) >= 6 and np.any(valid != 0):
        try:
            p_sp = float(wilcoxon(valid).pvalue)
        except ValueError:
            pass

    print(f"\n{'=' * W}")
    print("AGGREGATE (paired per-fold deltas, + = walk-forward-tuned better)")
    print(f"{'=' * W}")
    print(f"spearman_r: mean Δ={mean_sp:+.4f}  95% CI [{lo_sp:+.4f}, {hi_sp:+.4f}]  "
          f"wilcoxon_p={p_sp:.4f}" if np.isfinite(p_sp) else
          f"spearman_r: mean Δ={mean_sp:+.4f}  95% CI [{lo_sp:+.4f}, {hi_sp:+.4f}]")
    print(f"log_rmse:   mean Δ={mean_lr:+.4f}  95% CI [{lo_lr:+.4f}, {hi_lr:+.4f}]")
    fast_vs_full_disagree = bool(
        np.isfinite(fast_deltas.mean()) and np.sign(fast_deltas.mean()) != np.sign(mean_sp)
    )
    if fast_vs_full_disagree:
        print("WARNING: fast-objective and full-model OOS deltas disagree in sign — "
              "the DE objective approximation may be misleading the tuner.")

    # ── Parameter-stability table ─────────────────────────────────────────────
    frozen: dict[str, float] = {}
    frozen_path = PROJECT_ROOT / "archive" / "ownership_regression_results.json"
    if frozen_path.exists():
        try:
            frozen = json.loads(frozen_path.read_text())["params"]
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    param_matrix = np.array(
        [[f["tuned_params"][n] for n in PARAM_NAMES] for f in folds], dtype=float
    )
    print(f"\n{'=' * W}")
    print("PARAMETER STABILITY ACROSS FOLDS (is the in-sample optimum rediscovered OOS?)")
    print(f"{'=' * W}")
    print(f"{'Parameter':<30} {'prod':>8} {'frozen':>8} {'median':>8} {'IQR_lo':>8} {'IQR_hi':>8}")
    print("-" * W)
    stability: dict[str, dict] = {}
    for j, name in enumerate(PARAM_NAMES):
        med = float(np.median(param_matrix[:, j]))
        q1, q3 = np.percentile(param_matrix[:, j], [25, 75])
        froz = frozen.get(name, float("nan"))
        print(f"{name:<30} {prod_params[j]:>8.4f} {froz:>8.4f} "
              f"{med:>8.4f} {q1:>8.4f} {q3:>8.4f}")
        stability[name] = {
            "production": float(prod_params[j]),
            "frozen_full_sample": float(froz) if np.isfinite(froz) else None,
            "median": med, "iqr": [float(q1), float(q3)],
        }

    # ── Verdict ───────────────────────────────────────────────────────────────
    significant = lo_sp > 0
    verdict = "GO — promote tuned constants" if significant else (
        "NO-GO — OOS gain not significant; keep production constants"
    )
    print(f"\nVerdict: walk-forward tuned vs production OOS Δspearman = {mean_sp:+.4f} "
          f"[{lo_sp:+.4f}, {hi_sp:+.4f}] → {verdict}")

    results = {
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit_hash(),
        "production_constants": _collect_production_constants(),
        "config": {
            "popsize": popsize, "maxiter": maxiter, "min_train": args.min_train,
            "seed": args.seed, "n_slates": len(loaded), "n_folds": len(folds),
        },
        "folds": folds,
        "aggregate": {
            "mean_delta_spearman": round(mean_sp, 4),
            "ci_spearman": [round(lo_sp, 4), round(hi_sp, 4)],
            "wilcoxon_p_spearman": round(p_sp, 4) if np.isfinite(p_sp) else None,
            "mean_delta_log_rmse": round(mean_lr, 4),
            "ci_log_rmse": [round(lo_lr, 4), round(hi_lr, 4)],
            "fast_vs_full_disagree": fast_vs_full_disagree,
            "verdict": verdict,
        },
        "parameter_stability": stability,
    }
    out_path = PROJECT_ROOT / "archive" / "ownership_walkforward_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
