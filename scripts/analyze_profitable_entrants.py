"""Flip the rival-analysis question around: instead of picking one named
entrant and asking "how do we compare to them" (`analyze_rival_portfolio.py`,
`analyze_rival_roi.py`), find WHO in the archived field was actually
profitable, then look for portfolio-construction features that correlate
with that -- projection-chasing, chalk/contrarian tilt (and whether it
adapts to contest field size), portfolio diversity, and how a winning
lineup's projected ownership compared to what it realized.

Two views of "profitable" are reported side by side, never collapsed into
one score: the entry-weighted rate ladder (cash%/top1%/top0.1% vs the
field baseline -- proven far more stable at this sample size than dollars)
and a trimmed dollar ROI (drop_max, LOSO, bootstrap CI -- the same
robustness conventions `tests/backtest_lab.py::report` already uses,
because a single huge-field lottery-ticket win otherwise dominates any
raw ROI ranking).

Reuses, unmodified: `tests.bt_core.load_real_contests`/`ZIP_TO_CONTEST` for
zip discovery and payout wiring, `scripts.analyze_rival_roi.collect`/
`payout_for` for the entrant x contest money table, and
`scripts.analyze_rival_portfolio.parse_standings_rows`/`overlap_profile`/
`exposure_profile`/`team_map` for portfolio structure. New here: looping
every NAMED zip per slate (not just the UI-dupe zip `analyze_rival_portfolio`
reads) to get an entrant's full cross-contest lineup set, and joining each
lineup to SaberSim's per-player projected score/ownership.

Usage
-----
    source venv/bin/activate
    python scripts/analyze_profitable_entrants.py
    python scripts/analyze_profitable_entrants.py --min-contest-instances 8 --tolerance 0.85
"""
import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tests.bt_core import BACKTEST_SLATES, load_real_contests  # noqa: E402
from scripts.analyze_rival_roi import collect, payout_for  # noqa: E402
from scripts.analyze_rival_portfolio import (  # noqa: E402
    parse_standings_rows, overlap_profile, exposure_profile, team_map,
    pitcher_names,
)
from src.api.external_pool import discover_external_files  # noqa: E402

TENTH_SLATE = "08032026"  # full named zips present, not yet in BACKTEST_SLATES


# ---------------------------------------------------------------------------
# Stage B: max-entry cohort
# ---------------------------------------------------------------------------

def scan_contest_instances(slates: list) -> pd.DataFrame:
    """(slate, contest, zip_stem, handle, n_entries) -- one row per handle
    per NAMED zip instance, counted directly off each zip's own entry rows
    (after `parse_standings_rows`, so it reuses the already-validated
    header-driven parse). Ground truth for cap computation -- deliberately
    NOT derived from `analyze_rival_roi.collect()`'s aggregated `contest`
    (display-name) column, for two reasons found by inspection:

    1. Several DISTINCT single-entry contests share one display name via
       `ZIP_TO_CONTEST` (Base Hit's 4k/6k/10k/5k guarantee-size variants).
       Aggregating by display name misreads "entered two different $4k and
       $6k Base Hit contests once each" as "multi-entered Base Hit twice".
    2. `archive/07252026/four-seamer.zip` and `sour-seamer.zip` are
       byte-identical duplicates of the same contest under two names --
       `ZIP_TO_CONTEST`'s own comment assumes this can't happen ("no
       four-seamer.zip exists otherwise") but it does in the current
       archive state, and `load_real_contests` doesn't dedupe, so anything
       built on its per-zip iteration (`collect()` included) double-counts
       that one contest's entries/fees/winnings on that one slate. Content-
       hash dedup below sidesteps it for cap/cohort purposes; Stage A/C's
       dollar figures still inherit it for Four-Seamer entrants on 07/25
       specifically -- flagged in this script's printed output, not
       silently patched here since `tests/bt_core.py` is shared by other
       historical analyses this script isn't chartered to touch.
    """
    from tests.bt_core import ZIP_TO_CONTEST
    import hashlib
    rows = []
    for slate in slates:
        d = PROJECT_ROOT / "archive" / slate
        if not d.exists():
            continue
        seen_hashes = set()
        for z in sorted(d.glob("*.zip")):
            if z.name.startswith("contest-standings"):
                continue
            stem = z.stem.lower()
            contest = ZIP_TO_CONTEST.get(stem)
            if contest is None:
                continue
            with zipfile.ZipFile(z) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".csv"))
                data = zf.read(name)
            h = hashlib.sha256(data).hexdigest()
            if h in seen_hashes:
                print(f"  note: {slate}/{z.name} is a byte-identical duplicate of "
                      "another named zip this slate -- skipped for cap/cohort purposes")
                continue
            seen_hashes.add(h)
            csv_rows = list(csv.reader(io.StringIO(
                data.decode("utf-8-sig", errors="replace"))))
            e, _ = parse_standings_rows(csv_rows)
            if e.empty:
                continue
            counts = e.groupby("handle").size()
            for handle, n in counts.items():
                rows.append({"slate": slate, "contest": contest, "zip_stem": stem,
                             "handle": handle, "n_entries": int(n)})
    return pd.DataFrame(rows)


def compute_contest_caps(instances_df: pd.DataFrame) -> dict:
    """contest display name -> empirical per-contest-instance entry cap,
    from `scan_contest_instances`'s ground-truth per-zip handle counts. The
    cap is the largest count that recurs across >=2 distinct handles -- a
    single outlier (e.g. one handle firing a one-off burst) shouldn't set
    the tier. A contest where no handle ever exceeds 1 entry has no
    portfolio concept."""
    caps = {}
    for contest, g in instances_df.groupby("contest"):
        multi = g[g.n_entries > 1]
        if multi.empty:
            caps[contest] = 1
            continue
        recurring = multi.groupby("n_entries").handle.nunique()
        recurring = recurring[recurring >= 2]
        caps[contest] = (int(recurring.index.max()) if len(recurring)
                          else int(multi.n_entries.max()))
    return caps


def cohort(instances_df: pd.DataFrame, caps: dict, min_contest_instances: int = 5,
           tolerance: float = 0.9) -> pd.DataFrame:
    """Per-handle qualifying-instance counts, overall and per cap tier.
    A handle qualifies in a (slate, contest, zip_stem) instance when its
    entry count there is >= tolerance * that contest's cap, and the cap is
    > 1 (single-entry contests never count toward the cohort)."""
    df = instances_df.copy()
    df["cap"] = df.contest.map(caps)
    df = df[df.cap > 1]
    df["qualifies"] = df.n_entries >= tolerance * df.cap
    q = df[df.qualifies]
    per_handle = q.groupby("handle").agg(
        n_instances=("cap", "size"),
        tiers=("cap", lambda s: tuple(sorted(set(s)))),
    ).reset_index()
    for tier in sorted(set(caps.values()) - {1}):
        per_handle[f"n_tier_{tier}"] = per_handle["tiers"].map(
            lambda t, tier=tier: int(tier in t))
    per_handle["n_field_tiers_covered"] = per_handle["tiers"].map(len)
    members = per_handle[per_handle.n_instances >= min_contest_instances].copy()
    return members.sort_values("n_instances", ascending=False)


# ---------------------------------------------------------------------------
# Stage C: profitability, both views
# ---------------------------------------------------------------------------

def rate_ladder(entries_df: pd.DataFrame, handle: str) -> dict:
    g = entries_df[entries_df.entrant == handle]
    if g.empty:
        return {}
    base = {"cash%": 100 * (entries_df.won > 0).mean(),
            "top1%": 100 * entries_df.top1.mean(),
            "top01%": 100 * entries_df.top01.mean()}
    mine = {"cash%": 100 * (g.won > 0).mean(),
            "top1%": 100 * g.top1.mean(),
            "top01%": 100 * g.top01.mean()}
    per_slate_top1 = g.groupby("slate").top1.mean()
    field_top1 = entries_df.groupby("slate").top1.mean()
    lift = (per_slate_top1 - field_top1.reindex(per_slate_top1.index)).dropna()
    out = {"entries": len(g), "slates": g.slate.nunique()}
    for k in base:
        out[k] = mine[k]
        out[f"{k}_field"] = base[k]
        out[f"{k}_lift"] = mine[k] - base[k]
    if len(lift):
        out["top1_n_pos_slates"] = f"{int((lift > 0).sum())}/{len(lift)}"
        out["top1_LOSO_min"] = min(
            (lift.sum() - x) / (len(lift) - 1) for x in lift) if len(lift) > 1 else np.nan
        out["top1_LOSO_max"] = max(
            (lift.sum() - x) / (len(lift) - 1) for x in lift) if len(lift) > 1 else np.nan
    return out


def dollar_view(entries_df: pd.DataFrame, handle: str, n_boot: int = 20000,
                 rng: np.random.Generator | None = None) -> dict:
    g = entries_df[entries_df.entrant == handle]
    if g.empty:
        return {}
    rng = rng or np.random.default_rng(0)
    fees, won = g.fee.sum(), g.won.sum()
    per = g.groupby("slate").apply(
        lambda x: (x.won.sum() - x.fee.sum()) / len(x), include_groups=False)
    bs = per.to_numpy()[rng.integers(0, len(per), size=(n_boot, len(per)))].mean(axis=1)
    gmax = g.won.max()
    return {
        "fees": fees, "won": won, "net": won - fees,
        "ROI%": 100 * (won - fees) / fees if fees else np.nan,
        "$/entry": (won - fees) / len(g),
        "drop_max_$/entry": (won - gmax - fees) / len(g),
        "lo95": np.percentile(bs, 2.5), "hi95": np.percentile(bs, 97.5),
        "pos_slates": f"{int((per > 0).sum())}/{len(per)}",
        "LOSO_min": min((per.sum() - x) / (len(per) - 1) for x in per) if len(per) > 1 else np.nan,
        "LOSO_max": max((per.sum() - x) / (len(per) - 1) for x in per) if len(per) > 1 else np.nan,
        "best_rank": int(g["rank"].min()),
    }


# ---------------------------------------------------------------------------
# Stage D: full per-contest lineups + SaberSim / realized ownership join
# ---------------------------------------------------------------------------

def parse_all_named_contests(archive_dir: Path, cohort_handles: set) -> list[dict]:
    """[{slate, contest, contest_id, n_field, entries_df, field_players_df}]
    -- entries_df restricted to cohort_handles, with won/rank attached via
    the SAME payout_for used by Stage A, so dollar figures must match
    exactly for identical (slate, contest, entrant) rows (see verification)."""
    out = []
    for c in load_real_contests(archive_dir):
        z = archive_dir / f"{c['contest_id'].split(':', 1)[1]}.zip"
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            rows = list(csv.reader(io.StringIO(
                zf.read(name).decode("utf-8-sig", errors="replace"))))
        e, fp = parse_standings_rows(rows)
        e = e[e.handle.isin(cohort_handles)].copy()
        if e.empty:
            continue
        won, rank = zip(*[payout_for(p, c["sorted_scores"], c["payout_arr"])
                           for p in e["points"]])
        e["won"] = won
        e["rank_check"] = rank
        e["n_field"] = c["n_field"]
        out.append({"slate": archive_dir.name, "contest": c["contest"],
                     "contest_id": c["contest_id"], "n_field": c["n_field"],
                     "entries_df": e, "field_players_df": fp})
    return out


def join_sabersim(entries_df: pd.DataFrame, field_players_df: pd.DataFrame,
                   archive_dir: Path) -> tuple[pd.DataFrame, int]:
    """Attach proj_total/proj_own_mean/realized_own_mean per lineup. Reads
    the RAW SaberSim projections CSV directly (not
    external_pool.parse_sabersim_projections, which drops unconfirmed
    batters and non-starting pitchers -- exactly the players a contrarian
    lineup is more likely to roster, which would bias this analysis against
    finding a contrarian signal)."""
    sal = archive_dir / "DKSalaries.csv"
    if not sal.exists():
        return entries_df.assign(proj_total=np.nan, proj_own_mean=np.nan,
                                  realized_own_mean=np.nan), len(entries_df)
    sal_df = pd.read_csv(sal)
    sal_df["Name"] = sal_df["Name"].astype(str).str.strip()
    name_to_id = sal_df.set_index("Name")["ID"].to_dict()
    found = discover_external_files(str(archive_dir))
    proj_path = found.get("projections_path")
    if proj_path is None:
        return entries_df.assign(proj_total=np.nan, proj_own_mean=np.nan,
                                  realized_own_mean=np.nan), len(entries_df)
    raw = pd.read_csv(proj_path)
    raw = raw.dropna(subset=["DFS ID"]).drop_duplicates(subset=["DFS ID"], keep="first")
    id_to_proj = pd.to_numeric(raw["dk_points"], errors="coerce").set_axis(
        raw["DFS ID"].astype(int)).to_dict()
    id_to_own = pd.to_numeric(raw["Adj Own"], errors="coerce").set_axis(
        raw["DFS ID"].astype(int)).to_dict()
    field_own = field_players_df.set_index("player")["pct_drafted"].to_dict()

    proj_totals, proj_owns, realized_owns = [], [], []
    n_dropped = 0
    for names in entries_df["names"]:
        ids = [name_to_id.get(n) for n in names]
        if any(i is None for i in ids):
            n_dropped += 1
            proj_totals.append(np.nan); proj_owns.append(np.nan); realized_owns.append(np.nan)
            continue
        ids = [int(i) for i in ids]
        pts = [id_to_proj.get(i) for i in ids]
        owns = [id_to_own.get(i) for i in ids]
        if any(p is None or (isinstance(p, float) and np.isnan(p)) for p in pts):
            n_dropped += 1
            proj_totals.append(np.nan); proj_owns.append(np.nan)
        else:
            proj_totals.append(float(np.sum(pts)))
            proj_owns.append(float(np.nanmean(owns)))
        realized_owns.append(float(np.nanmean([field_own.get(n) for n in names])))

    out = entries_df.copy()
    out["proj_total"] = proj_totals
    out["proj_own_mean"] = proj_owns
    out["realized_own_mean"] = realized_owns
    return out, n_dropped


# ---------------------------------------------------------------------------
# Stage E: per-handle features
# ---------------------------------------------------------------------------

def proj_corr(lineups_df: pd.DataFrame, handle: str) -> dict:
    g = lineups_df[(lineups_df.handle == handle) & lineups_df.proj_total.notna()]
    field = lineups_df[lineups_df.proj_total.notna()]
    if len(g) < 3:
        return {"proj_corr": np.nan, "proj_corr_field": np.nan, "n_lineups": len(g)}
    return {
        "proj_corr": float(g["proj_total"].corr(g["points"], method="spearman")),
        "proj_corr_field": float(field["proj_total"].corr(field["points"], method="spearman")),
        "n_lineups": len(g),
    }


def chalk_profile(lineups_df: pd.DataFrame, handle: str, cohort_row: pd.Series) -> dict:
    g = lineups_df[(lineups_df.handle == handle) & lineups_df.realized_own_mean.notna()]
    field = lineups_df[lineups_df.realized_own_mean.notna()]
    if g.empty:
        return {}
    field_baseline = field["realized_own_mean"].mean()
    chalk_index = g["realized_own_mean"].mean() - field_baseline
    own_std = g["realized_own_mean"].std()
    out = {"chalk_index": chalk_index, "own_std": own_std,
           "field_baseline_own": field_baseline}
    if cohort_row.get("n_field_tiers_covered", 1) >= 2 and g["n_field"].nunique() >= 2:
        x = np.log(g["n_field"].to_numpy(dtype=float))
        y = g["realized_own_mean"].to_numpy(dtype=float)
        r = float(np.corrcoef(x, y)[0, 1])
        slope = float(np.polyfit(x, y, 1)[0])
        out["field_size_slope"] = slope
        out["field_size_r"] = r
    else:
        out["field_size_slope"] = np.nan
        out["field_size_r"] = np.nan
    if chalk_index > 2 and own_std < g["realized_own_mean"].mean() * 0.3:
        label = "chalky"
    elif chalk_index < -2 and own_std < abs(chalk_index) * 1.5:
        label = "contrarian"
    elif not np.isnan(out["field_size_slope"]) and abs(out["field_size_slope"]) > 1:
        label = "field_adaptive"
    elif own_std > abs(chalk_index) * 1.5:
        label = "spread"
    else:
        label = "static"
    out["label"] = label
    return out


def diversity_profile(lineups_df: pd.DataFrame, handle: str, large_field_cutoff: int) -> dict:
    g = lineups_df[(lineups_df.handle == handle) & (lineups_df.n_field >= large_field_cutoff)]
    if g.empty:
        return {"mean_overlap": np.nan, "pct_ge5": np.nan, "pct_ge7": np.nan,
                "exposure_herfindahl": np.nan, "n_large_field_instances": 0}
    ovs, hhis, n_inst = [], [], 0
    for (slate, contest), sub in g.groupby(["slate", "contest"]):
        lus = list(sub["names"])
        if len(lus) < 2:
            continue
        ov = overlap_profile(lus)
        exp = exposure_profile(lus)
        ovs.append((ov["mean_overlap"], ov["pct_ge5"], ov["pct_ge7"]))
        hhis.append(float((exp ** 2).sum()))
        n_inst += 1
    if not ovs:
        return {"mean_overlap": np.nan, "pct_ge5": np.nan, "pct_ge7": np.nan,
                "exposure_herfindahl": np.nan, "n_large_field_instances": 0}
    arr = np.array(ovs)
    return {"mean_overlap": float(arr[:, 0].mean()), "pct_ge5": float(arr[:, 1].mean()),
            "pct_ge7": float(arr[:, 2].mean()), "exposure_herfindahl": float(np.mean(hhis)),
            "n_large_field_instances": n_inst}


def own_delta(lineups_df: pd.DataFrame, handle: str) -> dict:
    g = lineups_df[(lineups_df.handle == handle) & lineups_df.proj_own_mean.notna()
                    & lineups_df.realized_own_mean.notna()]
    field = lineups_df[lineups_df.proj_own_mean.notna() & lineups_df.realized_own_mean.notna()]
    if g.empty:
        return {}
    g = g.assign(delta=g.realized_own_mean - g.proj_own_mean)
    cashed = g[g.won > 0]
    field_delta = float((field.realized_own_mean - field.proj_own_mean).mean())
    return {
        "own_delta_all": float(g["delta"].mean()),
        "own_delta_cashed": float(cashed["delta"].mean()) if len(cashed) else np.nan,
        "own_delta_field": field_delta,
        "n_cashed": len(cashed),
    }


# ---------------------------------------------------------------------------
# Stage F: correlate
# ---------------------------------------------------------------------------

def correlate_features(profitability_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    merged = profitability_df.merge(features_df, on="handle", how="inner")
    prof_cols = ["top1%_lift", "top01%_lift", "cash%_lift", "$/entry", "drop_max_$/entry"]
    feat_cols = ["proj_corr", "chalk_index", "own_std", "field_size_slope",
                 "mean_overlap", "exposure_herfindahl", "own_delta_all", "own_delta_cashed"]
    print("\n===== Spearman correlation: feature x profitability (n = cohort handles with both non-NaN) =====")
    for f in feat_cols:
        if f not in merged.columns:
            continue
        row = []
        for p in prof_cols:
            if p not in merged.columns:
                continue
            sub = merged[[f, p]].dropna()
            n = len(sub)
            if n < 4:
                row.append(f"{p}: n={n} (too few)")
                continue
            r = sub[f].corr(sub[p], method="spearman")
            row.append(f"{p}: r={r:+.3f} n={n}")
        print(f"  {f:20s} " + "   ".join(row))
    return merged


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-contest-instances", type=int, default=5)
    p.add_argument("--tolerance", type=float, default=0.9)
    p.add_argument("--slates", default=None,
                    help="comma-separated MMDDYYYY override (default: BACKTEST_SLATES + 08032026)")
    p.add_argument("--large-field-cutoff", type=int, default=1000,
                    help="min n_field for a contest instance to count toward diversity")
    args = p.parse_args()

    slates = (args.slates.split(",") if args.slates
              else list(BACKTEST_SLATES) + [TENTH_SLATE])
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"Stage A: collecting entrant x contest table over {len(slates)} slates: {slates}")
    entries_df = collect(slates)
    entries_df.to_csv(out_dir / "profitable_entrants_entries.csv", index=False)
    print(f"  {len(entries_df):,} entries across {entries_df.contest.nunique()} contest types")

    print("\nStage B: computing contest caps and max-entry cohort (ground-truth zip scan)")
    instances_df = scan_contest_instances(slates)
    caps = compute_contest_caps(instances_df)
    for c, cap in sorted(caps.items()):
        print(f"  {c:20s} cap={cap}")
    coh = cohort(instances_df, caps, args.min_contest_instances, args.tolerance)
    coh.to_csv(out_dir / "profitable_entrants_cohort.csv", index=False)
    print(f"  cohort size: {len(coh)}")
    if coh.empty:
        print("  no cohort members found -- lower --min-contest-instances or --tolerance")
        return
    cohort_handles = set(coh["handle"])

    print("\nStage C: profitability (rate ladder + trimmed dollar ROI)")
    prof_rows = []
    rng = np.random.default_rng(0)
    for h in cohort_handles:
        row = {"handle": h}
        row.update(rate_ladder(entries_df, h))
        row.update(dollar_view(entries_df, h, rng=rng))
        prof_rows.append(row)
    profitability_df = pd.DataFrame(prof_rows)
    profitability_df.to_csv(out_dir / "profitable_entrants_profitability.csv", index=False)

    print("\nStage D: parsing full per-contest lineups for cohort handles (this is slow -- "
          f"{len(slates)} slates x ~9 named zips each)")
    all_lineups = []
    for slate in slates:
        d = PROJECT_ROOT / "archive" / slate
        if not d.exists():
            continue
        try:
            contests = parse_all_named_contests(d, cohort_handles)
        except SystemExit as exc:
            print(f"  skipping {slate}: {exc}")
            continue
        for c in contests:
            joined, n_dropped = join_sabersim(c["entries_df"], c["field_players_df"], d)
            if n_dropped:
                print(f"    {slate}/{c['contest']}: {n_dropped}/{len(joined)} lineups "
                      "unresolved against SaberSim projections")
            joined["slate"] = slate
            joined["contest"] = c["contest"]
            all_lineups.append(joined)
    if not all_lineups:
        print("  no cohort lineups parsed -- aborting before Stage E/F")
        return
    lineups_df = pd.concat(all_lineups, ignore_index=True)
    lineups_df.to_csv(out_dir / "profitable_entrants_lineups.csv", index=False)
    print(f"  {len(lineups_df):,} cohort lineups parsed")

    print("\nStage E: per-handle features")
    feat_rows = []
    for _, crow in coh.iterrows():
        h = crow["handle"]
        row = {"handle": h}
        row.update(proj_corr(lineups_df, h))
        row.update(chalk_profile(lineups_df, h, crow))
        row.update(diversity_profile(lineups_df, h, args.large_field_cutoff))
        row.update(own_delta(lineups_df, h))
        feat_rows.append(row)
    features_df = pd.DataFrame(feat_rows)
    features_df.to_csv(out_dir / "profitable_entrants_features.csv", index=False)

    print("\nStage F: correlating features against profitability")
    report = correlate_features(profitability_df, features_df)
    report = report.merge(coh, on="handle", how="left", suffixes=("", "_cohort"))
    report.to_csv(out_dir / "profitable_entrants_report.csv", index=False)
    print(f"\nwrote outputs/profitable_entrants_report.csv ({len(report)} cohort handles)")


if __name__ == "__main__":
    main()
