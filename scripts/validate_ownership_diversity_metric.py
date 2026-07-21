"""
Validates the "ownership-weighted overlap" hypothesis for portfolio
diversification against real DraftKings contest archives.

Theory being tested: for two lineups, how much sharing a player should count
against "diversity" ought to depend on how rare that player is (owned %) —
sharing a 45%-owned stud barely means anything (the whole field has him too,
so his outcome moves everyone together and cancels out in relative
standing), while sharing a 3%-owned piece is a real shared bet that should
move two lineups' *relative* field position together far more. This script
checks that directly against real contest fields rather than assuming it:
does an ownership/IDF-weighted overlap between two REAL entrants' rosters
predict their realized co-movement (z-scored actual points, product z_i*z_j)
better than plain (unweighted) roster overlap?

Data per archived slate (from contest-standings-*.zip, via
evaluate_ownership._parse_contest_zip):
  - standings_df: one row per real contest entry — entry_id, points,
    lineup_str (parsed to a 10-player roster by analyze_contest_lineups.
    _parse_lineup_string).
  - ownership_df: one row per player — pct_drafted (real field ownership %).

Method
------
Per slate: sample --sample-size real entrants, build a binary
entrant x player roster matrix M, and compute for every pair (i, j) in the
sample:
  - raw_overlap    = |L_i ∩ L_j|                      (0-10)
  - jaccard_raw    = raw_overlap / |L_i ∪ L_j|
  - idf_weighted_jaccard (two weight forms: -log(own_p), 1-own_p) — a
    weighted Jaccard where shared low-owned players count more toward
    "similarity" than shared chalk (TF-IDF-style, ownership as an inverse-
    document-frequency signal)
  - z_i * z_j      — product of each entrant's realized points, z-scored
    within the slate's full field. This is a pointwise estimator of the
    local covariance contribution: pooling z_i*z_j within similarity
    buckets approximates that bucket's implied correlation of realized
    outcomes.

Pairs are pooled across every slate (z-scoring makes them comparable across
slates with different scoring environments) and reported as: bucketed
mean(z_i*z_j) by raw overlap count and by weighted-Jaccard decile, headline
Pearson correlations, a partial correlation of each weighted metric
controlling for raw_overlap_count (isolates whether ownership-weighting adds
signal *beyond* simple shared-player count), and a per-slate sign test
(does the weighted metric out-predict raw overlap more often than not,
slate-by-slate, guarding against one slate driving the pooled result).

Usage
-----
    python scripts/validate_ownership_diversity_metric.py
    python scripts/validate_ownership_diversity_metric.py --recent 10
    python scripts/validate_ownership_diversity_metric.py --sample-size 500 --seed 1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_ownership import _parse_contest_zip  # noqa: E402
from analyze_contest_lineups import _parse_lineup_string  # noqa: E402
from analyze_candidate_pool import _slate_sort_key  # noqa: E402

_OWN_FLOOR = 0.001  # matches evaluate_ownership._LOG_EPS's resolution-floor reasoning
_DEFAULT_SAMPLE_SIZE = 400
_MIN_VALID_ENTRANTS = 20


def _find_slates(n: int | None) -> list[Path]:
    archive_root = PROJECT_ROOT / "archive"
    full = sorted(
        (d for d in archive_root.iterdir() if d.is_dir() and list(d.glob("contest-standings-*.zip"))),
        key=lambda d: _slate_sort_key(d.name),
    )
    return full[-n:] if n else full


def _load_slate(slate_dir: Path, sample_size: int, rng: np.random.Generator):
    """Returns (raw_overlap, jaccard_raw, wjac_log, wjac_lin, zz) arrays for
    all sampled pairs in this slate, or None if the slate can't be used."""
    zips = sorted(slate_dir.glob("contest-standings-*.zip"))
    standings_df, ownership_df = _parse_contest_zip(zips[0])
    if standings_df.empty or ownership_df.empty:
        return None

    rosters: list[frozenset] = []
    points: list[float] = []
    for _, row in standings_df.iterrows():
        parsed = _parse_lineup_string(str(row.get("lineup_str", "")))
        if not parsed:
            continue
        rosters.append(frozenset(name for _, name in parsed))
        points.append(row["points"])

    if len(rosters) < _MIN_VALID_ENTRANTS:
        return None

    points_arr = np.array(points, dtype=np.float64)
    mean_all, std_all = points_arr.mean(), points_arr.std()
    if std_all == 0:
        return None

    n = len(rosters)
    size = min(sample_size, n)
    idx = rng.choice(n, size=size, replace=False)
    sub_rosters = [rosters[i] for i in idx]
    z = (points_arr[idx] - mean_all) / std_all

    own_map = ownership_df.groupby("player_name")["pct_drafted"].mean().to_dict()

    universe = sorted(set().union(*sub_rosters))
    p_index = {p: k for k, p in enumerate(universe)}
    own_vec = np.clip(
        np.array([own_map.get(p, _OWN_FLOOR) for p in universe], dtype=np.float64),
        _OWN_FLOOR, 1.0,
    )

    m, P = len(sub_rosters), len(universe)
    M = np.zeros((m, P), dtype=np.float64)
    for i, roster in enumerate(sub_rosters):
        for p in roster:
            M[i, p_index[p]] = 1.0

    w_log = -np.log(own_vec)
    w_lin = 1.0 - own_vec

    overlap_mat = M @ M.T
    wsum_log = M @ w_log
    wsum_lin = M @ w_lin
    wov_log_mat = (M * w_log[None, :]) @ M.T
    wov_lin_mat = (M * w_lin[None, :]) @ M.T

    iu = np.triu_indices(m, k=1)
    overlap = overlap_mat[iu]
    union = 20.0 - overlap
    jaccard_raw = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)

    wo_log = wov_log_mat[iu]
    wu_log = wsum_log[iu[0]] + wsum_log[iu[1]] - wo_log
    wjac_log = np.divide(wo_log, wu_log, out=np.zeros_like(wo_log), where=wu_log > 0)

    wo_lin = wov_lin_mat[iu]
    wu_lin = wsum_lin[iu[0]] + wsum_lin[iu[1]] - wo_lin
    wjac_lin = np.divide(wo_lin, wu_lin, out=np.zeros_like(wo_lin), where=wu_lin > 0)

    zz = z[iu[0]] * z[iu[1]]

    return overlap, jaccard_raw, wjac_log, wjac_lin, zz


def _decile_report(x: np.ndarray, zz: np.ndarray) -> None:
    edges = np.quantile(x, np.linspace(0, 1, 11))
    edges[0] -= 1e-9
    bins = np.digitize(x, edges[1:-1], right=True)
    for b in range(10):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        print(
            f"  decile {b + 1:2d}  range=[{edges[b]:.4f},{edges[b + 1]:.4f}]  "
            f"n={n:9,}  mean(z_i*z_j)={zz[mask].mean():+.4f}"
        )


def _report(pooled: dict, per_slate: list[dict]) -> None:
    overlap, jac_raw, wjac_log, wjac_lin, zz = (
        pooled["overlap"], pooled["jac_raw"], pooled["wjac_log"], pooled["wjac_lin"], pooled["zz"],
    )
    print(f"\nPooled pairs: {len(zz):,} across {len(per_slate)} slates\n")

    print("=== Raw overlap-count buckets (0-10 shared players) ===")
    for k in range(0, 11):
        mask = overlap == k
        n = int(mask.sum())
        if n == 0:
            continue
        se = zz[mask].std() / np.sqrt(n)
        print(f"  overlap={k:2d}  n={n:9,}  mean(z_i*z_j)={zz[mask].mean():+.4f}  se={se:.4f}")

    print("\n=== IDF-weighted-Jaccard (log form: -log(own)) deciles ===")
    _decile_report(wjac_log, zz)

    print("\n=== IDF-weighted-Jaccard (linear form: 1-own) deciles ===")
    _decile_report(wjac_lin, zz)

    print("\n=== Headline correlations vs. realized co-movement (z_i * z_j) ===")
    metrics = [
        ("raw_overlap_count", overlap),
        ("jaccard_raw", jac_raw),
        ("idf_weighted_jaccard_log", wjac_log),
        ("idf_weighted_jaccard_lin", wjac_lin),
    ]
    for name, x in metrics:
        r, p = pearsonr(x, zz)
        print(f"  {name:28s}  r={r:+.4f}  p={p:.2e}")

    print("\n=== Partial correlation: does IDF-weighting add signal beyond raw overlap count? ===")
    r_zx, _ = pearsonr(overlap, zz)
    for name, x in metrics[2:]:
        r_zw, _ = pearsonr(x, zz)
        r_xw, _ = pearsonr(overlap, x)
        denom = np.sqrt((1 - r_zx ** 2) * (1 - r_xw ** 2))
        partial = (r_zw - r_zx * r_xw) / denom if denom > 0 else float("nan")
        print(f"  partial_r({name} ; controlling raw_overlap_count) = {partial:+.4f}")

    print("\n=== Per-slate sign test: does the weighted metric out-predict raw overlap? ===")
    for name in ("wjac_log", "wjac_lin"):
        diffs = np.array([s[f"r_{name}"] - s["r_overlap"] for s in per_slate])
        wins = int((diffs > 0).sum())
        n_slates = len(diffs)
        test = binomtest(wins, n_slates, 0.5)
        print(
            f"  {name}: wins {wins}/{n_slates} slates (mean diff={diffs.mean():+.4f}, "
            f"binomial p={test.pvalue:.3f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the ownership-weighted overlap diversity metric against real DK contest archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--recent", type=int, default=0, metavar="N",
                         help="Only use the N most recent slates (default: all slates with a standings zip).")
    parser.add_argument("--sample-size", type=int, default=_DEFAULT_SAMPLE_SIZE, metavar="K",
                         help=f"Real entrants sampled per slate (default: {_DEFAULT_SAMPLE_SIZE}).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for entrant sampling (default: 0).")
    args = parser.parse_args()

    slates = _find_slates(args.recent or None)
    if not slates:
        print("No archive slates with a contest-standings zip found.")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    pooled_arrays = {"overlap": [], "jac_raw": [], "wjac_log": [], "wjac_lin": [], "zz": []}
    per_slate = []

    for d in slates:
        try:
            result = _load_slate(d, args.sample_size, rng)
        except Exception as exc:
            print(f"Skipping {d.name}: {exc}")
            continue
        if result is None:
            print(f"Skipping {d.name}: not enough valid parsed entrants.")
            continue

        overlap, jac_raw, wjac_log, wjac_lin, zz = result
        pooled_arrays["overlap"].append(overlap)
        pooled_arrays["jac_raw"].append(jac_raw)
        pooled_arrays["wjac_log"].append(wjac_log)
        pooled_arrays["wjac_lin"].append(wjac_lin)
        pooled_arrays["zz"].append(zz)

        r_overlap, _ = pearsonr(overlap, zz)
        r_wjac_log, _ = pearsonr(wjac_log, zz)
        r_wjac_lin, _ = pearsonr(wjac_lin, zz)
        per_slate.append({
            "slate": d.name, "n_pairs": len(zz),
            "r_overlap": r_overlap, "r_wjac_log": r_wjac_log, "r_wjac_lin": r_wjac_lin,
        })
        print(f"{d.name}: n_pairs={len(zz):,}  r_overlap={r_overlap:+.3f}  "
              f"r_wjac_log={r_wjac_log:+.3f}  r_wjac_lin={r_wjac_lin:+.3f}")

    if not per_slate:
        print("No usable slates.")
        sys.exit(1)

    pooled = {k: np.concatenate(v) for k, v in pooled_arrays.items()}
    _report(pooled, per_slate)


if __name__ == "__main__":
    main()
