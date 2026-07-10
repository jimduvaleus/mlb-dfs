"""Fit the duplicate-entry model against real DK contest standings.

The GPP dupe penalty (ContestScorer, gpp.dupe_* config) predicts how many
field entries duplicate a candidate lineup from a log-linear model:

    log E[dupes] = intercept
                   + log_own_coef * SUM log(ownership_i)
                   - salary_coef  * (unused salary / $100)
                   + stack_coef   * (primary stack size - 4)

This script fits those coefficients on the contest-standings archive
(archive/<MMDDYYYY>/contest-standings-*.zip). For every contest:

  - each entry's lineup string is parsed into 10 players,
  - identical lineups are grouped -> observed copy count c per unique lineup,
  - SUM log(own) uses the contest's real %Drafted column,
  - salary and team (for the primary stack) come from the slate's
    DKSalaries.csv archived alongside the standings.

Counts are modelled as zero-truncated Poisson: a lineup only enters the data
when someone played it (c >= 1), but the production quantity is E[copies] of
an arbitrary candidate we might submit, i.e. the untruncated mean mu(x).
Naive regression on (c - 1) would underestimate mu for contrarian lineups by
~2x (E[c - 1 | c >= 1] -> mu/2 as mu -> 0). Contest size enters as an offset
log(N / N_REF), so the fitted intercept is calibrated to the reference
14,863-entry DK Classic GPP the payout structure models — matching the
production model, whose intercept absorbs field size.

Usage:
    python scripts/fit_dupe_model.py [--archive-root archive] [--min-entries 1000]
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

N_REF = 14_863          # entries in the reference DK Classic GPP structure
SALARY_CAP = 50_000.0
OWN_CLIP = (1e-4, 0.95)  # must match ContestScorer._compute_dupe_scale

_LINEUP_RE = re.compile(r"(?:^|\s)(1B|2B|3B|C|OF|P|SS)\s+")


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def parse_lineup(s: str) -> list[tuple[str, str]] | None:
    """'1B Kody Clemens 2B ... SS Otto Lopez' -> [(slot, name), ...] or None."""
    parts = _LINEUP_RE.split(s.strip())
    # parts = ['', slot, name, slot, name, ...]
    if len(parts) < 21 or parts[0].strip():
        return None
    out = []
    for i in range(1, len(parts) - 1, 2):
        name = parts[i + 1].strip()
        if not name:
            return None
        out.append((parts[i], name))
    return out if len(out) == 10 else None


def load_contest(zip_path: str) -> tuple[pd.DataFrame, dict[str, float]] | None:
    """Return (entries df with lineup strings, %drafted map) for one contest."""
    z = zipfile.ZipFile(zip_path)
    with z.open(z.namelist()[0]) as f:
        rows = list(csv.reader(io.TextIOWrapper(f, "utf-8-sig")))
    header = rows[0]
    try:
        i_lineup = header.index("Lineup")
        i_player = header.index("Player")
        i_drafted = header.index("%Drafted")
    except ValueError:
        return None

    lineups: list[str] = []
    drafted: dict[str, float] = {}
    for r in rows[1:]:
        if len(r) > i_lineup and r[0].strip() and r[i_lineup].strip():
            lineups.append(r[i_lineup].strip())
        if len(r) > i_drafted and r[i_player].strip() and r[i_drafted].strip():
            drafted[_norm_name(r[i_player])] = (
                float(r[i_drafted].rstrip("%")) / 100.0
            )
    if not lineups or not drafted:
        return None
    return pd.DataFrame({"lineup": lineups}), drafted


def load_salaries(slate_dir: Path) -> tuple[dict[str, float], dict[str, str]]:
    """name -> salary, name -> team from the archived DKSalaries.csv."""
    path = slate_dir / "DKSalaries.csv"
    if not path.exists():
        return {}, {}
    df = pd.read_csv(path)
    keys = df["Name"].map(_norm_name)
    return (
        dict(zip(keys, df["Salary"].astype(float))),
        dict(zip(keys, df["TeamAbbrev"])),
    )


def build_design(archive_root: Path, min_entries: int) -> pd.DataFrame:
    """One row per unique lineup per contest: count, features, offset."""
    frames = []
    seen_contest_ids: set[str] = set()
    for zp in sorted(glob.glob(str(archive_root / "*" / "contest-standings-*.zip"))):
        slate_dir = Path(zp).parent
        # Re-run archives (e.g. 07042026r) hold a copy of the base slate's
        # standings zip — count each contest once.
        contest_id = Path(zp).stem.rsplit("-", 1)[-1]
        if contest_id in seen_contest_ids:
            print(f"  {slate_dir.name}: contest {contest_id} already counted — skipped")
            continue
        seen_contest_ids.add(contest_id)
        loaded = load_contest(zp)
        if loaded is None:
            print(f"  {slate_dir.name}: unparseable standings — skipped")
            continue
        entries, drafted = loaded
        n_entries = len(entries)
        if n_entries < min_entries:
            print(f"  {slate_dir.name}: only {n_entries} entries — skipped")
            continue
        sal_map, team_map = load_salaries(slate_dir)
        if not sal_map:
            print(f"  {slate_dir.name}: no DKSalaries.csv — skipped")
            continue

        counts = entries["lineup"].value_counts()
        rows = []
        n_unmatched = 0
        for lineup_str, c in counts.items():
            parsed = parse_lineup(lineup_str)
            if parsed is None:
                continue
            slo = 0.0
            salary = 0.0
            team_counts: dict[str, int] = {}
            ok = True
            for slot, name in parsed:
                key = _norm_name(name)
                own = drafted.get(key)
                sal = sal_map.get(key)
                if own is None or sal is None:
                    ok = False
                    break
                slo += np.log(np.clip(own, *OWN_CLIP))
                salary += sal
                if slot != "P":
                    t = team_map.get(key, "")
                    if t:
                        team_counts[t] = team_counts.get(t, 0) + 1
            if not ok:
                n_unmatched += 1
                continue
            rows.append({
                "count": int(c),
                "sum_log_own": slo,
                "unused_hundreds": max(SALARY_CAP - salary, 0.0) / 100.0,
                "stack_minus4": (max(team_counts.values()) if team_counts else 0) - 4,
                "offset": np.log(n_entries / N_REF),
                "slate": slate_dir.name,
            })
        df = pd.DataFrame(rows)
        frames.append(df)
        n_dupes = int((df["count"] - 1).sum())
        print(
            f"  {slate_dir.name}: {n_entries} entries, {len(df)} unique lineups, "
            f"{n_dupes} duplicate copies"
            + (f", {n_unmatched} lineups dropped (name mismatch)" if n_unmatched else "")
        )
    if not frames:
        raise SystemExit("No usable contests found.")
    return pd.concat(frames, ignore_index=True)


def fit_ztp(df: pd.DataFrame) -> np.ndarray:
    """MLE of the zero-truncated Poisson GLM. Returns beta (4,)."""
    X = np.column_stack([
        np.ones(len(df)),
        df["sum_log_own"].to_numpy(),
        df["unused_hundreds"].to_numpy(),
        df["stack_minus4"].to_numpy(np.float64),
    ])
    c = df["count"].to_numpy(np.float64)
    off = df["offset"].to_numpy()

    def negll(beta):
        eta = np.clip(X @ beta + off, -15.0, 8.0)
        mu = np.exp(eta)
        # log P(c | c>=1) = c log mu - mu - log(1 - e^-mu)   (+ const)
        ll = c * eta - mu - np.log(-np.expm1(-mu))
        return -ll.sum()

    def grad(beta):
        eta = np.clip(X @ beta + off, -15.0, 8.0)
        mu = np.exp(eta)
        # d/d_eta: c - mu - mu e^-mu / (1 - e^-mu)
        w = c - mu - mu * np.exp(-mu) / (-np.expm1(-mu))
        return -(X * w[:, None]).sum(axis=0)

    beta0 = np.array([9.0, 0.42, -0.15, 0.0])
    res = minimize(negll, beta0, jac=grad, method="BFGS")
    if not res.success:
        print(f"WARNING: optimizer did not fully converge: {res.message}")
    return res.x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive-root", default=str(PROJECT_ROOT / "archive"))
    ap.add_argument("--min-entries", type=int, default=1000)
    args = ap.parse_args()

    print("Parsing contest standings...")
    df = build_design(Path(args.archive_root), args.min_entries)
    n_contests = df["slate"].nunique()
    print(
        f"\nDesign: {len(df):,} unique lineups from {n_contests} contests, "
        f"{int((df['count'] - 1).sum()):,} duplicate copies "
        f"({(df['count'] > 1).mean() * 100:.1f}% of unique lineups duped)"
    )

    beta = fit_ztp(df)
    intercept, c_own, b_sal, c_stack = beta
    c_sal = -b_sal  # production formula subtracts salary_coef * unused_hundreds

    print("\nFitted coefficients (reference contest = "
          f"{N_REF:,} entries):")
    print(f"  dupe_intercept        = {intercept:.4f}")
    print(f"  dupe_log_own_coef     = {c_own:.4f}")
    print(f"  dupe_salary_coef      = {c_sal:.4f}   (per $100 unused)")
    print(f"  dupe_stack_coef       = {c_stack:.4f}")

    # Diagnostics: observed vs predicted mean copies by predicted-mu decile.
    X = np.column_stack([
        np.ones(len(df)), df["sum_log_own"], df["unused_hundreds"],
        df["stack_minus4"].astype(float),
    ])
    mu = np.exp(np.clip(X @ beta + df["offset"].to_numpy(), -15.0, 8.0))
    # Comparison must be on the truncated scale (data only contains c >= 1).
    mu_trunc = mu / -np.expm1(-mu)
    dec = pd.qcut(mu, 10, labels=False, duplicates="drop")
    diag = pd.DataFrame({
        "pred_mu": mu, "pred_copies_trunc": mu_trunc, "obs_copies": df["count"],
    }).groupby(dec).mean()
    print("\nCalibration by predicted-mu decile "
          "(pred_copies_trunc vs obs_copies should track):")
    print(diag.round(3).to_string())

    # What the penalty implies at the reference contest size.
    print("\nImplied E[dupes] at reference size:")
    for label, slo, unused, stack in [
        ("max chalk (20% avg own, full salary, 5-stack)",
         10 * np.log(0.20), 0.0, 1),
        ("mid (10% avg own, $300 unused, 4-stack)", 10 * np.log(0.10), 3.0, 0),
        ("contrarian (5% avg own, $700 unused, 4-stack)",
         10 * np.log(0.05), 7.0, 0),
    ]:
        log_d = intercept + c_own * slo - c_sal * unused + c_stack * stack
        e_d = np.exp(np.clip(log_d, -20, 10))
        print(f"  {label:<50} E[dupes]={e_d:8.3f}  scale={1/(1+e_d):.3f}")

    print("\nConfig snippet (gpp:):")
    print(f"  dupe_intercept: {intercept:.3f}")
    print(f"  dupe_log_own_coef: {c_own:.3f}")
    print(f"  dupe_salary_coef: {c_sal:.3f}")
    print(f"  dupe_stack_coef: {c_stack:.3f}")


if __name__ == "__main__":
    main()
