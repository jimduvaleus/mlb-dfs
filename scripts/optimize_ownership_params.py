"""
Numerical optimization of ownership model constants.

Loads all archive slates with contest-standings data, pre-extracts player pool
data into NumPy arrays (once), then uses scipy.optimize.differential_evolution
to maximize mean Spearman correlation across all slates.

The fast NumPy objective (~5–10 ms per evaluation) makes a full 22,000-eval DE
run in ~3–5 min rather than hours.

Usage
-----
    # All available slates
    python scripts/optimize_ownership_params.py

    # N most recent full slates
    python scripts/optimize_ownership_params.py --recent 10

    # Specific slates
    python scripts/optimize_ownership_params.py archive/05092026 archive/05102026

    # Thorough run (larger population, more iterations)
    python scripts/optimize_ownership_params.py --popsize 15 --maxiter 500

Output
------
  Prints parameter comparison table and per-slate Spearman breakdown.
  Writes archive/ownership_regression_results.json consumed by K_regressed in
  evaluate_ownership.py.

Overfitting note
----------------
With ~11 free parameters and ~10 slates, in-sample optimization captures some
noise. Treat gains < 0.005 Spearman as within-variance; gains ≥ 0.01 sustained
across all slates are likely real.
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_ownership import (  # noqa: E402
    _build_player_pool,
    _find_recent_full_slates,
    _match_ownership,
    _parse_contest_zip,
)
import src.optimization.ownership as _own_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

# (module_attr_name, (lower_bound, upper_bound))
# Current values are read from the live module so they stay in sync with
# ownership.py even after manual edits.
_PARAM_DEFS: list[tuple[str, tuple[float, float]]] = [
    ("_BATTER_STD_FLOOR",            (0.30, 1.50)),
    ("_PITCHER_STD_FLOOR",           (0.15, 0.80)),
    ("_PITCHER_COMPRESS",            (0.00, 0.60)),
    ("_PITCHER_MATCHUP_EXP",         (0.10, 3.00)),
    ("_PITCHER_COSTACK_EXP",         (0.00, 1.50)),
    ("_BATTER_TOTAL_CAP",            (3.50, 8.00)),
    ("_STACK_VALUE_EXP",             (0.00, 0.60)),
    ("_TIME_FACTOR",                 (0.00, 0.40)),
    ("_PITCHER_MEAN_FRAC",           (0.50, 1.00)),
    ("_HR_PROB_EXP",                 (0.00, 0.50)),
    ("_SECONDARY_POSITION_DISCOUNT", (0.10, 0.90)),
]

PARAM_NAMES  = [d[0] for d in _PARAM_DEFS]
PARAM_BOUNDS = [d[1] for d in _PARAM_DEFS]

_SLOT_COUNTS = {"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}


def _current_params() -> np.ndarray:
    return np.array([getattr(_own_mod, name) for name in PARAM_NAMES])


# ---------------------------------------------------------------------------
# Pre-extracted slate representation (built once, used by fast objective)
# ---------------------------------------------------------------------------

@dataclass
class _SlateArrays:
    label: str
    n: int

    # Per-player arrays ── fixed signals (not parametric)
    is_pitcher:      np.ndarray   # [n] bool
    sqrt_mean:       np.ndarray   # [n] sqrt(mean.clip(0))
    sqrt_sv:         np.ndarray   # [n] sqrt(salary_value.clip(0))
    batter_base_raw: np.ndarray   # [n] sqrt(mean)*bo_mult*sal_cap for batters
    has_hr:          np.ndarray   # [n] bool — batter has hr_prob
    hr_factor:       np.ndarray   # [n] hr_prob / mean_hr_prob, else 1.0
    time_frac:       np.ndarray   # [n] fraction 0–1 (1 = earliest game)

    # Per-player team/opponent indices
    player_team:     np.ndarray   # [n] int team index, -1 if none
    player_opp:      np.ndarray   # [n] int opp team index, -1 if none

    # Per-team arrays
    n_teams:         int
    team_totals:     np.ndarray   # [n_teams] implied total, nan if missing
    slot5_sal:       np.ndarray   # [n_teams] avg top-5 batter salary
    mean_team_total: float
    mean_slot5_sal:  float
    has_team_totals: bool

    # Position groups: pos → (player_indices[k], is_primary[k])
    pos_groups: dict  # str → (np.ndarray[int], np.ndarray[bool])

    # Matched ownership
    matched_idx:    np.ndarray   # [m] indices into pool
    matched_actual: np.ndarray   # [m] actual pct_drafted


def _parse_game_minutes(game_str: str) -> float | None:
    m = re.search(r'(\d{1,2}):(\d{2})(AM|PM)', str(game_str))
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    if ap == "PM" and h != 12:
        h += 12
    elif ap == "AM" and h == 12:
        h = 0
    return float(h * 60 + mi)


def _preextract(
    label: str,
    pool_df,
    matched_df,
    team_totals: dict | None,
) -> _SlateArrays:
    """Convert a pool DataFrame + matched ownership into a _SlateArrays struct."""
    df   = pool_df.reset_index(drop=True)
    n    = len(df)

    mean_arr = df["mean"].values.astype(float)
    sal_arr  = df["salary"].values.astype(float)
    sv_arr   = (mean_arr / (sal_arr / 1000.0)).clip(0)
    sqrt_mean = np.sqrt(mean_arr.clip(0))
    sqrt_sv   = np.sqrt(sv_arr)

    is_pitcher  = (df["position"] == "P").values.astype(bool)
    batter_mask = ~is_pitcher

    # Batting-order multiplier (fixed — not a parameter)
    slot_col = (
        "lineup_slot" if "lineup_slot" in df.columns
        else ("slot" if "slot" in df.columns else None)
    )
    slots = df[slot_col].values.astype(float) if slot_col else np.full(n, np.nan)
    bo_mult = np.ones(n)
    for batting_slot, mult in _own_mod._BATTING_ORDER_MULT.items():
        mask = batter_mask & (slots == float(batting_slot))
        bo_mult[mask] = mult

    # Salary-cap pressure (fixed — _BATTER_SALARY_PRESSURE_BASE not a parameter)
    sal_cap = np.ones(n)
    sal_cap[batter_mask] = np.minimum(
        1.0, (_own_mod._BATTER_SALARY_PRESSURE_BASE / sal_arr[batter_mask]) ** 0.5
    )

    batter_base_raw = sqrt_mean * bo_mult * sal_cap

    # HR probability factor — store ratio; exponent is a parameter
    has_hr   = np.zeros(n, dtype=bool)
    hr_factor = np.ones(n)
    if "hr_prob" in df.columns:
        hr_prob = df["hr_prob"].values.astype(float)
        hr_valid = batter_mask & np.isfinite(hr_prob)
        if hr_valid.any():
            mean_hr = hr_prob[hr_valid].mean()
            if mean_hr > 0:
                has_hr[hr_valid] = True
                hr_factor[hr_valid] = hr_prob[hr_valid] / mean_hr

    # Game start-time fraction: 1 = earliest bucket, 0 = latest bucket.
    # _TIME_NEUTRAL_WINDOW is not a parameter so use current module value.
    time_frac = np.zeros(n)
    if "game" in df.columns:
        tw = int(_own_mod._TIME_NEUTRAL_WINDOW)
        game_mins = {g: _parse_game_minutes(g) for g in df["game"].unique()}
        valid_mins = [v for v in game_mins.values() if v is not None]
        if len(valid_mins) > 1:
            min_t = min(valid_mins)
            buckets = {
                g: min_t + tw * int((t - min_t) // tw)
                for g, t in game_mins.items() if t is not None
            }
            bvals     = list(buckets.values())
            min_b, max_b = min(bvals), max(bvals)
            rng = max_b - min_b
            if rng > 0:
                frac_map = {g: (max_b - b) / rng for g, b in buckets.items()}
                for i, g in enumerate(df["game"].values):
                    if batter_mask[i] and g in frac_map:
                        time_frac[i] = frac_map[g]

    # Team/opponent index mapping
    unique_teams = sorted(set(df["team"].dropna().unique()))
    team_to_idx  = {t: i for i, t in enumerate(unique_teams)}
    n_teams      = len(unique_teams)

    player_team = np.array(
        [team_to_idx.get(str(t), -1) for t in df["team"].values], dtype=int
    )
    if "opponent" in df.columns:
        player_opp = np.array(
            [team_to_idx.get(str(t), -1) for t in df["opponent"].values], dtype=int
        )
    else:
        player_opp = np.full(n, -1, dtype=int)

    # Per-team implied totals and slot5 salary
    team_totals_arr = np.full(n_teams, np.nan)
    slot5_sal_arr   = np.full(n_teams, 4500.0)
    mean_team_total = 1.0
    mean_slot5_sal  = 4500.0
    has_team_totals = False

    if team_totals:
        active_teams = set(df["team"].dropna().unique())
        active = {t: v for t, v in team_totals.items() if t in active_teams and v and v > 0}
        if active:
            has_team_totals = True
            for t, v in active.items():
                if t in team_to_idx:
                    team_totals_arr[team_to_idx[t]] = v
            mean_team_total = float(np.nanmean(team_totals_arr))

            batter_idx = np.where(batter_mask)[0]
            for t, ti in team_to_idx.items():
                t_idx = batter_idx[player_team[batter_idx] == ti]
                if len(t_idx) == 0:
                    continue
                t_slots = slots[t_idx]
                valid   = np.isfinite(t_slots)
                if valid.sum() >= 3:
                    order    = np.argsort(t_slots[valid])
                    top5_idx = t_idx[valid][order[:5]]
                else:
                    order    = np.argsort(-mean_arr[t_idx])
                    top5_idx = t_idx[order[:5]]
                slot5_sal_arr[ti] = float(sal_arr[top5_idx].mean())

            valid_s5 = slot5_sal_arr[np.isfinite(slot5_sal_arr)]
            mean_slot5_sal = float(valid_s5.mean()) if len(valid_s5) else 4500.0

    # Position groups
    use_eligible = "eligible_positions" in df.columns
    pos_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if use_eligible:
        eligible = df["eligible_positions"].tolist()
        primary  = df["position"].tolist()
        all_pos: set[str] = set()
        for ep in eligible:
            all_pos.update(ep if isinstance(ep, list) else [str(ep)])
        for pos in all_pos:
            idxs, is_prim = [], []
            for i, (ep, pp) in enumerate(zip(eligible, primary)):
                in_ep = (pos in ep) if isinstance(ep, list) else (str(ep) == pos)
                if in_ep:
                    idxs.append(i)
                    is_prim.append(pp == pos)
            if idxs:
                pos_groups[pos] = (
                    np.array(idxs, dtype=int),
                    np.array(is_prim, dtype=bool),
                )
    else:
        prim_pos = df["position"].values
        for pos in np.unique(prim_pos):
            idxs = np.where(prim_pos == pos)[0]
            pos_groups[pos] = (idxs, np.ones(len(idxs), dtype=bool))

    # Matched player indices into pool
    pid_to_idx = {int(pid): i for i, pid in enumerate(df["player_id"].values)}
    m_idx, m_actual = [], []
    for _, row in matched_df.iterrows():
        pid = int(row["player_id"])
        if pid in pid_to_idx:
            m_idx.append(pid_to_idx[pid])
            m_actual.append(float(row["pct_drafted"]))
    matched_idx    = np.array(m_idx,    dtype=int)
    matched_actual = np.array(m_actual, dtype=float)

    return _SlateArrays(
        label=label, n=n, is_pitcher=is_pitcher,
        sqrt_mean=sqrt_mean, sqrt_sv=sqrt_sv,
        batter_base_raw=batter_base_raw,
        has_hr=has_hr, hr_factor=hr_factor, time_frac=time_frac,
        player_team=player_team, player_opp=player_opp,
        n_teams=n_teams, team_totals=team_totals_arr,
        slot5_sal=slot5_sal_arr, mean_team_total=mean_team_total,
        mean_slot5_sal=mean_slot5_sal, has_team_totals=has_team_totals,
        pos_groups=pos_groups,
        matched_idx=matched_idx, matched_actual=matched_actual,
    )


# ---------------------------------------------------------------------------
# Fast NumPy objective  (~5–10 ms per call, no pandas in the hot loop)
# ---------------------------------------------------------------------------

def _score_slates(
    params_vec: np.ndarray,
    slates: list[_SlateArrays],
) -> tuple[float, dict[str, float]]:
    """
    Return (negative_mean_spearman, per_slate_dict).

    Unpacks params_vec into the 11 named parameters and evaluates the
    ownership model on each pre-extracted slate using pure NumPy.
    """
    (p_bsf, p_psf, p_pc, p_pme, p_pce,
     p_btc, p_sve, p_tf, p_pmf, p_hpe, p_spd) = params_vec

    scores: dict[str, float] = {}

    for s in slates:
        raw         = np.empty(s.n)
        batter_mask = ~s.is_pitcher

        # Raw scores
        raw[s.is_pitcher] = (
            p_pmf * s.sqrt_mean[s.is_pitcher]
            + (1.0 - p_pmf) * s.sqrt_sv[s.is_pitcher]
        )
        raw[batter_mask] = s.batter_base_raw[batter_mask].copy()

        # HR-probability batter boost
        if s.has_hr.any():
            raw[s.has_hr] *= s.hr_factor[s.has_hr] ** p_hpe

        # Game start-time boost (batters only)
        raw[batter_mask] *= 1.0 + p_tf * s.time_frac[batter_mask]

        # Implied-total boosts
        if s.has_team_totals:
            boost = np.ones(s.n)
            mt = s.mean_team_total

            for ti in range(s.n_teams):
                total = s.team_totals[ti]
                if not (np.isfinite(total) and total > 0 and mt > 0):
                    continue
                capped = min(total, p_btc)

                b_mask = batter_mask & (s.player_team == ti)
                if b_mask.any():
                    s5 = s.slot5_sal[ti]
                    sm = (s.mean_slot5_sal / s5) ** p_sve if (total >= 4.0 and s5 > 0) else 1.0
                    boost[b_mask] = (capped / mt) * sm

                po_mask = s.is_pitcher & (s.player_opp == ti)
                if po_mask.any():
                    boost[po_mask] = (mt / capped) ** p_pme

            for ti in range(s.n_teams):
                total = s.team_totals[ti]
                if not (np.isfinite(total) and total > 0 and mt > 0):
                    continue
                capped = min(total, p_btc)
                po_own = s.is_pitcher & (s.player_team == ti)
                if po_own.any():
                    boost[po_own] *= (capped / mt) ** p_pce

            raw *= boost

        # Per-position softmax
        ownership = np.zeros(s.n)
        for pos, (idxs, is_prim) in s.pos_groups.items():
            vals    = raw[idxs]
            n_slots = _SLOT_COUNTS.get(pos, 1)
            is_p    = pos == "P"
            sf      = p_psf if is_p else p_bsf

            std     = vals.std()
            shifted = (vals - vals.mean()) / max(float(std), sf)
            exp_v   = np.exp(shifted)
            sfmx    = exp_v / exp_v.sum()

            if is_p:
                uniform = np.ones(len(vals)) / len(vals)
                share   = (1.0 - p_pc) * sfmx + p_pc * uniform
            else:
                share = sfmx

            discount = np.where(is_prim, 1.0, p_spd)
            np.add.at(ownership, idxs, share * n_slots * discount)

        # Spearman on matched players
        pred   = ownership[s.matched_idx]
        actual = s.matched_actual
        mask   = np.isfinite(pred) & np.isfinite(actual)
        if mask.sum() >= 5:
            r, _ = spearmanr(actual[mask], pred[mask])
            scores[s.label] = float(r)

    if not scores:
        return 0.0, {}
    return -float(np.mean(list(scores.values()))), scores


def _objective(params_vec: np.ndarray, slates: list[_SlateArrays]) -> float:
    neg_mean, _ = _score_slates(params_vec, slates)
    return neg_mean


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_slates_with_frames(
    archive_dirs: list[Path],
) -> list[tuple[_SlateArrays, "object", "object", dict | None]]:
    """
    Load and pre-extract all slates, keeping the source frames.

    Returns a list of (slate_arrays, pool_df, matched_df, team_totals) so
    callers (e.g. walk_forward_ownership.py) can score candidate parameters
    through the full compute_heuristic_ownership model without re-reading
    the archives.
    """
    slates = []
    for d in archive_dirs:
        label = d.name
        zips  = sorted(d.glob("contest-standings-*.zip"))
        if not zips:
            print(f"  [{label}] No contest standings zip — skipping.", flush=True)
            continue
        print(f"  [{label}] Loading ...", end=" ", flush=True)
        try:
            _, ownership_df = _parse_contest_zip(zips[0])
            pool_df         = _build_player_pool(d)
            matched_df      = _match_ownership(ownership_df, pool_df)
            if len(matched_df) < 10:
                print(f"only {len(matched_df)} matched — skipping.", flush=True)
                continue
            team_totals = None
            if pool_df["implied_total"].notna().any():
                team_totals = (
                    pool_df[["team", "implied_total"]]
                    .dropna(subset=["implied_total"])
                    .drop_duplicates("team")
                    .set_index("team")["implied_total"]
                    .to_dict()
                )
            sa = _preextract(label, pool_df, matched_df, team_totals)
            slates.append((sa, pool_df, matched_df, team_totals))
            print(f"{len(matched_df)} players matched.", flush=True)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
    return slates


def _load_slates(archive_dirs: list[Path]) -> list[_SlateArrays]:
    """Load and pre-extract all slates. Returns list of _SlateArrays."""
    return [sa for sa, _, _, _ in _load_slates_with_frames(archive_dirs)]


# ---------------------------------------------------------------------------
# Optimization driver (importable — also used by walk_forward_ownership.py)
# ---------------------------------------------------------------------------

def run_de(
    slates: list[_SlateArrays],
    popsize: int = 15,
    maxiter: int = 300,
    seed: int = 42,
    polish: bool = True,
    init: "str | np.ndarray" = "latinhypercube",
    tol: float = 1e-4,
    verbose: bool = True,
) -> tuple[np.ndarray, float, dict]:
    """
    Differential evolution + optional Nelder-Mead polish over PARAM_BOUNDS.

    init may be an (n_candidates, n_params) array to warm-start the DE
    population (e.g. seeding with the current production constants and a
    previous fold's optimum).

    Returns (optimal_params, optimal_mean_spearman, info) where info carries
    de_nit / de_nfev / de_converged for reporting.
    """
    callback = None
    if verbose:
        iter_log: list[tuple[int, float]] = []

        def callback(xk: np.ndarray, convergence: float) -> bool:
            neg, _ = _score_slates(xk, slates)
            n = len(iter_log) + 1
            iter_log.append((n, -neg))
            if n % 25 == 0 or n == 1:
                print(
                    f"  iter {n:4d}  Spearman={-neg:.4f}"
                    f"  convergence={convergence:.5f}", flush=True
                )
            return False

    de_result = differential_evolution(
        _objective,
        bounds=PARAM_BOUNDS,
        args=(slates,),
        strategy="best1bin",
        popsize=popsize,
        maxiter=maxiter,
        tol=tol,
        seed=seed,
        workers=1,
        callback=callback,
        disp=False,
        init=init,
    )

    optimal_params = de_result.x.copy()
    optimal_score  = -de_result.fun
    if verbose:
        print(f"\nDE complete: {de_result.nit} iterations, {de_result.nfev} evals, "
              f"converged={de_result.success}", flush=True)

    if polish:
        if verbose:
            print(f"Polishing with Nelder-Mead (DE best: {optimal_score:.4f})...", flush=True)
        polish_res = minimize(
            _objective,
            optimal_params,
            args=(slates,),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6, "adaptive": True},
        )
        polished_score = -polish_res.fun
        if polished_score > optimal_score:
            optimal_params = polish_res.x.copy()
            optimal_score  = polished_score
            if verbose:
                print(f"  Polish improved: {polished_score:.4f}", flush=True)
        elif verbose:
            print(f"  No improvement ({polished_score:.4f} vs DE {optimal_score:.4f}); "
                  f"keeping DE result.", flush=True)

    info = {
        "de_nit":       int(de_result.nit),
        "de_nfev":      int(de_result.nfev),
        "de_converged": bool(de_result.success),
    }
    return optimal_params, optimal_score, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize ownership model constants via differential evolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "archive_dirs", nargs="*", metavar="ARCHIVE_DIR",
        help="Specific archive directories to evaluate.",
    )
    parser.add_argument(
        "--recent", type=int, default=0, metavar="N",
        help="Use the N most recent full slates.",
    )
    parser.add_argument(
        "--popsize", type=int, default=15,
        help="DE population multiplier per parameter (default 15).",
    )
    parser.add_argument(
        "--maxiter", type=int, default=300,
        help="DE maximum iterations (default 300).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default 42).",
    )
    parser.add_argument(
        "--no-polish", action="store_true",
        help="Skip Nelder-Mead local polish after DE.",
    )
    args = parser.parse_args()

    if args.recent and args.archive_dirs:
        parser.error("--recent and positional ARCHIVE_DIR are mutually exclusive.")

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
    slates = _load_slates(dirs)
    if not slates:
        print("No usable slates found.")
        sys.exit(1)
    print(f"\n{len(slates)} slate(s) ready.\n", flush=True)

    current_params = _current_params()

    # Time one evaluation to estimate runtime
    t0 = time.perf_counter()
    baseline_neg, _ = _score_slates(current_params, slates)
    ms_per_eval = (time.perf_counter() - t0) * 1000

    n_pop        = args.popsize * len(PARAM_NAMES)
    n_evals_est  = n_pop * args.maxiter
    est_min      = n_evals_est * ms_per_eval / 60_000

    print(f"Baseline Spearman (current constants): {-baseline_neg:.4f}", flush=True)
    print(f"Fast-objective timing: {ms_per_eval:.1f} ms per call", flush=True)
    print(
        f"DE config: {len(PARAM_NAMES)} params, popsize={args.popsize} "
        f"({n_pop} candidates), maxiter={args.maxiter}", flush=True
    )
    print(f"Estimated runtime: ~{est_min:.1f} min ({n_evals_est:,} evals)\n", flush=True)

    optimal_params, optimal_score, de_info = run_de(
        slates,
        popsize=args.popsize,
        maxiter=args.maxiter,
        seed=args.seed,
        polish=not args.no_polish,
    )

    # Per-slate breakdown
    _, current_per_slate = _score_slates(current_params, slates)
    _, optimal_per_slate = _score_slates(optimal_params, slates)

    # ── Report ────────────────────────────────────────────────────────────────
    W = 64
    print(f"\n{'='*W}")
    print("PARAMETER COMPARISON")
    print(f"{'='*W}")
    print(f"{'Parameter':<30} {'Current':>8} {'Optimal':>8} {'Δ':>8}")
    print(f"{'-'*W}")
    for name, cur, opt in zip(PARAM_NAMES, current_params, optimal_params):
        delta = opt - cur
        flag  = " *" if abs(delta) > max(0.05 * abs(cur), 0.01) else ""
        print(f"{name:<30} {cur:>8.4f} {float(opt):>8.4f} {delta:>+8.4f}{flag}")

    print(f"\n{'='*W}")
    print("PER-SLATE SPEARMAN")
    print(f"{'='*W}")
    print(f"{'Slate':<12} {'Current':>8} {'Optimal':>8} {'Δ':>8}")
    print(f"{'-'*W}")
    slates_sorted = sorted(current_per_slate)
    for label in slates_sorted:
        cur   = current_per_slate[label]
        opt   = optimal_per_slate.get(label, float("nan"))
        delta = opt - cur
        flag  = " +" if delta >= 0.005 else (" -" if delta <= -0.005 else "")
        print(f"{label:<12} {cur:>8.4f} {opt:>8.4f} {delta:>+8.4f}{flag}")

    cur_mean = float(np.mean(list(current_per_slate.values())))
    opt_mean = float(np.mean([optimal_per_slate[s] for s in slates_sorted
                               if s in optimal_per_slate]))
    gain = opt_mean - cur_mean
    print(f"{'-'*W}")
    print(f"{'MEAN':<12} {cur_mean:>8.4f} {opt_mean:>8.4f} {gain:>+8.4f}")

    if gain < 0.005:
        print("\n  NOTE: Gain < 0.005 Spearman — likely within noise. "
              "Collect more slates before applying changes.")
    elif gain >= 0.01:
        print(f"\n  Gain {gain:.4f} Spearman ≥ 0.01 — likely a real improvement.")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "params": {name: float(val) for name, val in zip(PARAM_NAMES, optimal_params)},
        "mean_spearman_optimal":  opt_mean,
        "mean_spearman_baseline": cur_mean,
        "gain":       gain,
        "per_slate":  {
            lbl: {
                "current": current_per_slate.get(lbl),
                "optimal": optimal_per_slate.get(lbl),
                "delta":   (optimal_per_slate.get(lbl) or 0.0)
                           - (current_per_slate.get(lbl) or 0.0),
            }
            for lbl in slates_sorted
        },
        "de_nit":       de_info["de_nit"],
        "de_nfev":      de_info["de_nfev"],
        "de_converged": de_info["de_converged"],
        "slates_used":  [s.label for s in slates],
        "param_bounds": {n: list(b) for n, b in zip(PARAM_NAMES, PARAM_BOUNDS)},
    }

    out_path = PROJECT_ROOT / "archive" / "ownership_regression_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}", flush=True)
    print("Review * flagged parameters (>5% change or >0.01 absolute) before "
          "applying to ownership.py.", flush=True)


if __name__ == "__main__":
    main()
