"""
Reverse-engineer a named entrant's whole portfolio out of the archived
contest-standings zips, and measure how it is structured.

The standings CSV carries every entrant's full lineup, so for a max-entry
professional this is their entire portfolio for that contest — the finished
product of whatever construction process they run. That makes it a
reference to measure our own portfolios against: not "did they beat us"
(one contest is noise) but "how is their 150 shaped, and how does that
differ from ours".

Metrics, all portfolio-level rather than lineup-level:

  overlap      distribution of pairwise player overlap (0-10) across the
               portfolio. The direct measurement of how hard the entries
               are spread apart.
  exposure     per-player share of the portfolio, vs the field's %Drafted
               from the standings sidebar — the leverage profile.
  stacks       primary/secondary team-stack shape (5-3, 4-4, ...) and how
               many distinct primary stack teams the portfolio spans.
  result       realized rank/points distribution and hit99 rate.

Comparison baselines on the same slate: a uniformly random draw of the same
size from the real field, and (when the SaberSim export is present) the same
size drawn from our own pool by value and by greedy max-diversity.

Usage
-----
    python scripts/analyze_rival_portfolio.py --entrant ShaidyAdvice
    python scripts/analyze_rival_portfolio.py --entrant ShaidyAdvice --slate 07262026
    python scripts/analyze_rival_portfolio.py --entrant ShaidyAdvice --mode exposure
"""
import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.api.external_pool import discover_external_files  # noqa: E402

# DK writes the lineup as "<POS> <Name> <POS> <Name> ..." with positions in a
# fixed alphabetical order, so splitting on the position tokens recovers the
# names without needing to know where one ends and the next begins.
_POS = ("1B", "2B", "3B", "C", "OF", "P", "SS")
_SPLIT = re.compile(r"\s*\b(" + "|".join(_POS) + r")\b\s+")
_ENTRY_SUFFIX = re.compile(r"\s*\(\d+/\d+\)\s*$")


def parse_standings_rows(rows: list[list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entries, field_players) from an already-decoded standings CSV's rows.
    The standings CSV holds two tables side by side separated by an empty
    column: per-entry results on the left, the player %Drafted/FPTS sidebar
    on the right. Shared by parse_standings (the UI-dupe zip, one contest)
    and any caller that loops the named per-contest zips instead."""
    hdr = rows[0]
    ci = {c: i for i, c in enumerate(hdr) if c}
    entries, players = [], []
    for r in rows[1:]:
        if len(r) > ci["Lineup"] and r[ci["Lineup"]].strip():
            entries.append({
                "rank": int(r[ci["Rank"]]) if r[ci["Rank"]].strip() else np.nan,
                "entry_name": r[ci["EntryName"]],
                "points": float(r[ci["Points"]]) if r[ci["Points"]].strip() else np.nan,
                "lineup_raw": r[ci["Lineup"]],
            })
        if "Player" in ci and len(r) > ci["Player"] and r[ci["Player"]].strip():
            pct = r[ci["%Drafted"]].replace("%", "").strip() if len(r) > ci["%Drafted"] else ""
            players.append({
                "player": r[ci["Player"]].strip(),
                "roster_position": r[ci["Roster Position"]].strip(),
                "pct_drafted": float(pct) if pct else np.nan,
                "fpts": float(r[ci["FPTS"]]) if len(r) > ci["FPTS"] and r[ci["FPTS"]].strip() else np.nan,
            })
    e = pd.DataFrame(entries)
    e["names"] = e["lineup_raw"].map(lambda s: tuple(sorted(x for x in _SPLIT.split(s)[1:][1::2])))
    e["handle"] = e["entry_name"].map(lambda s: _ENTRY_SUFFIX.sub("", s))
    return e, pd.DataFrame(players)


def parse_standings(archive_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entries, field_players) for the UI-dupe contest-standings zip (one
    contest per slate -- whichever the operator last exported from DK)."""
    zips = sorted(archive_dir.glob("contest-standings-*.zip"))
    if not zips:
        raise FileNotFoundError(f"no contest-standings zip in {archive_dir}")
    with zipfile.ZipFile(zips[0]) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        rows = list(csv.reader(io.StringIO(zf.read(name).decode("utf-8-sig"))))
    return parse_standings_rows(rows)


def team_map(archive_dir: Path) -> dict:
    """name -> team, from the slate's DKSalaries so stacks can be measured."""
    f = archive_dir / "DKSalaries.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    return dict(zip(df["Name"].astype(str).str.strip(), df["TeamAbbrev"].astype(str)))


def indicator(lineups: list[tuple]) -> tuple[np.ndarray, list]:
    ids = sorted({p for lu in lineups for p in lu})
    pos = {p: i for i, p in enumerate(ids)}
    m = np.zeros((len(lineups), len(ids)), dtype=np.float32)
    for r, lu in enumerate(lineups):
        for p in lu:
            m[r, pos[p]] = 1.0
    return m, ids


def overlap_profile(lineups: list[tuple]) -> dict:
    m, _ = indicator(lineups)
    ov = m @ m.T
    iu = np.triu_indices(len(lineups), k=1)
    v = ov[iu]
    counts = np.bincount(v.astype(int), minlength=11)[:11]
    return {"mean_overlap": float(v.mean()), "max_overlap": float(v.max()),
            "pct_ge7": float((v >= 7).mean()), "pct_ge5": float((v >= 5).mean()),
            "hist": counts / max(counts.sum(), 1)}


def exposure_profile(lineups: list[tuple]) -> pd.Series:
    n = len(lineups)
    s = pd.Series([p for lu in lineups for p in lu]).value_counts() / n
    return s


def cluster_decomposition(lineups: list[tuple], primary: list) -> dict:
    """Split pairwise overlap into pairs sharing a primary stack team and
    pairs that don't. This is the structural fingerprint that separates
    "150 spread-apart entries" from "~20 correlated clusters, one per
    plausible boom team": a clustered portfolio has high within-team overlap
    (several variants of the same bet) and low between-team overlap."""
    m, _ = indicator(lineups)
    ov = m @ m.T
    g = np.array([p if p is not None else "?" for p in primary])
    iu = np.triu_indices(len(lineups), k=1)
    same = g[iu[0]] == g[iu[1]]
    v = ov[iu]
    within = float(v[same].mean()) if same.any() else np.nan
    between = float(v[~same].mean()) if (~same).any() else np.nan
    return {"within": within, "between": between,
            "ratio": within / between if between else np.nan}


def primary_teams(lineups: list[tuple], tmap: dict, pitchers: set) -> list:
    """Each lineup's most-represented HITTER team (pitchers excluded, so a
    5-man stack isn't confounded by a same-team pitcher)."""
    out = []
    for lu in lineups:
        h = [tmap.get(p) for p in lu if p not in pitchers and tmap.get(p)]
        out.append(pd.Series(h).value_counts().index[0] if h else None)
    return out


def pitcher_names(archive_dir: Path) -> set:
    f = archive_dir / "DKSalaries.csv"
    if not f.exists():
        return set()
    df = pd.read_csv(f)
    return set(df.loc[df["Position"].astype(str).str.contains("P"), "Name"]
               .astype(str).str.strip())


def stack_profile(lineups: list[tuple], tmap: dict, pitchers: set) -> dict:
    """Stack shape over HITTERS only. DK MLB Classic caps hitters from one
    team at 5, so a "6" only ever appears if a same-team pitcher is counted
    alongside a 5-man hitter stack — which is a pitcher/stack pairing, not a
    6-man stack. The 8 hitter slots are what actually partition here."""
    shapes, primaries = [], []
    for lu in lineups:
        teams = [tmap.get(p) for p in lu if p not in pitchers and tmap.get(p)]
        c = pd.Series(teams).value_counts()
        if c.empty:
            continue
        top = c.iloc[:2].tolist()
        shapes.append("-".join(str(x) for x in top))
        primaries.append(c.index[0])
    return {"shape_counts": pd.Series(shapes).value_counts(normalize=True),
            "n_primary_teams": len(set(primaries)),
            "primary_counts": pd.Series(primaries).value_counts(normalize=True)}


def _our_pool(archive_dir: Path):
    """(lineups as name tuples, value score) from the SaberSim export, or None."""
    found = discover_external_files(str(archive_dir))
    if not found["lineups_paths"]:
        return None
    sal = archive_dir / "DKSalaries.csv"
    if not sal.exists():
        return None
    id_to_name = pd.read_csv(sal).set_index("ID")["Name"].astype(str).str.strip().to_dict()
    frames = [pd.read_csv(p) for p in found["lineups_paths"]]
    df = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]
    lus, keep = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        pids = [int(row[j]) for j in range(10)]
        names = [id_to_name.get(p) for p in pids]
        if any(n is None for n in names):
            continue
        lus.append(tuple(sorted(names)))
        keep.append(i)
    sub = df.iloc[keep]
    ev = (pd.to_numeric(sub["Proj Score"], errors="coerce")
          - pd.to_numeric(sub["Ownership"], errors="coerce") / 3.0).to_numpy()
    return lus, ev


def greedy_diverse(lineups: list[tuple], n: int, seed_row: int) -> list[int]:
    m, _ = indicator(lineups)
    picked = [seed_row]
    max_ov = m @ m[seed_row]
    tot_ov = max_ov.copy()
    for _ in range(n - 1):
        key = max_ov * 1000.0 + tot_ov
        key[picked] = np.inf
        j = int(np.argmin(key))
        picked.append(j)
        ov = m @ m[j]
        max_ov = np.maximum(max_ov, ov)
        tot_ov += ov
    return picked


def analyse_slate(d: Path, entrant: str, mode: str, seed: int) -> dict | None:
    try:
        entries, field_players = parse_standings(d)
    except (FileNotFoundError, ValueError, StopIteration) as exc:
        print(f"Skipping {d.name}: {exc}")
        return None
    mine = entries[entries["handle"].str.lower() == entrant.lower()]
    if mine.empty:
        return None
    lus = list(mine["names"])
    n = len(lus)
    tmap = team_map(d)
    rng = np.random.default_rng(seed)

    pitchers = pitcher_names(d)
    ov = overlap_profile(lus)
    exp = exposure_profile(lus)
    stk = stack_profile(lus, tmap, pitchers)
    field_pts = np.sort(entries["points"].dropna().to_numpy())
    pct = np.searchsorted(field_pts, mine["points"].to_numpy(), side="right") / len(field_pts)

    print(f"\n=== {d.name} === {entrant}: {n} entries of {len(entries):,} in the field")
    print(f"  best rank {int(mine['rank'].min())}   median rank {int(mine['rank'].median())}"
          f"   best pts {mine['points'].max():.1f}   field p99 {field_pts[int(len(field_pts)*.99)]:.1f}")
    print(f"  hit99 {np.mean(pct >= 0.99):.3f}   hit95 {np.mean(pct >= 0.95):.3f}"
          f"   mean pctile {pct.mean():.3f}")
    print(f"  overlap: mean {ov['mean_overlap']:.2f}  max {ov['max_overlap']:.0f}"
          f"  share of pairs >=5 {ov['pct_ge5']:.3f}  >=7 {ov['pct_ge7']:.3f}")
    print(f"  unique players {len(exp)}   max exposure {exp.max():.3f}"
          f"   n players over 50% {int((exp > 0.5).sum())}   over 25% {int((exp > 0.25).sum())}")
    print(f"  distinct primary stack teams {stk['n_primary_teams']}")

    # baselines
    rows = [("them", ov, exp, pct)]
    fl = list(entries["names"])
    ridx = rng.choice(len(fl), n, replace=False)
    rl = [fl[i] for i in ridx]
    rov, rexp = overlap_profile(rl), exposure_profile(rl)
    rpct = np.searchsorted(field_pts, entries["points"].to_numpy()[ridx], side="right") / len(field_pts)
    rows.append(("random field", rov, rexp, rpct))

    pool = _our_pool(d)
    if pool is not None:
        plus, ev = pool
        top = list(np.argsort(-ev)[:n])
        rows.append(("our pool: value", overlap_profile([plus[i] for i in top]),
                     exposure_profile([plus[i] for i in top]), None))
        div = greedy_diverse(plus, n, int(np.argmax(ev)))
        rows.append(("our pool: diverse", overlap_profile([plus[i] for i in div]),
                     exposure_profile([plus[i] for i in div]), None))

    # Structural fingerprint: overlap split by whether two entries share a
    # primary stack team. Needs DKSalaries for team/position lookup.
    lineup_sets = {"them": lus, "random field": rl}
    if pool is not None:
        lineup_sets["our pool: value"] = [plus[i] for i in top]
        lineup_sets["our pool: diverse"] = [plus[i] for i in div]

    print(f"\n  {'portfolio':20s} {'mean ovl':>9s} {'max':>4s} {'pairs>=5':>9s} "
          f"{'uniq':>5s} {'maxexp':>7s} {'>25%':>5s} {'within':>7s} {'betw':>6s} "
          f"{'ratio':>6s} {'teams':>6s} {'max/tm':>7s}")
    for label, o, e, _ in rows:
        ls = lineup_sets.get(label)
        cd, nt, mx = {"within": np.nan, "between": np.nan, "ratio": np.nan}, np.nan, np.nan
        if ls and tmap and pitchers:
            pr = primary_teams(ls, tmap, pitchers)
            cd = cluster_decomposition(ls, pr)
            vc = pd.Series([x for x in pr if x]).value_counts()
            nt, mx = len(vc), vc.max()
        print(f"  {label:20s} {o['mean_overlap']:9.2f} {o['max_overlap']:4.0f} "
              f"{o['pct_ge5']:9.3f} {len(e):5d} {e.max():7.3f} {int((e > 0.25).sum()):5d} "
              f"{cd['within']:7.2f} {cd['between']:6.2f} {cd['ratio']:6.2f} "
              f"{nt:6.0f} {mx:7.0f}")

    if mode == "overlap":
        print("\n  pairwise overlap histogram (share of pairs at each overlap 0..10)")
        print(f"  {'portfolio':20s} " + " ".join(f"{i:>6d}" for i in range(11)))
        for label, o, _, _ in rows:
            print(f"  {label:20s} " + " ".join(f"{v:6.3f}" for v in o["hist"]))

    if mode == "exposure":
        fp = field_players.set_index("player")["pct_drafted"].to_dict()
        t = pd.DataFrame({"exposure": exp * 100})
        t["field_pct"] = [fp.get(p, np.nan) for p in t.index]
        t["leverage"] = t["exposure"] - t["field_pct"]
        t["fpts"] = [field_players.set_index("player")["fpts"].to_dict().get(p, np.nan) for p in t.index]
        print("\n  top 20 exposures vs the field")
        print(t.sort_values("exposure", ascending=False).head(20)
              .to_string(float_format=lambda x: f"{x:.1f}"))
        print(f"\n  players they used that the field had under 5%: "
              f"{int(((t['field_pct'] < 5) & (t['exposure'] > 0)).sum())}")
        print(f"  correlation(exposure, field %drafted) = "
              f"{t[['exposure','field_pct']].corr().iloc[0,1]:+.3f}")

    if mode == "stacks":
        print("\n  primary-secondary shape distribution")
        print(stk["shape_counts"].head(10).to_string(float_format=lambda x: f"{x:.3f}"))
        print("\n  primary stack team distribution")
        print(stk["primary_counts"].head(12).to_string(float_format=lambda x: f"{x:.3f}"))

    return {"slate": d.name, "n": n, "hit99": float(np.mean(pct >= 0.99)),
            "hit95": float(np.mean(pct >= 0.95)), "mean_pct": float(pct.mean()),
            "best_rank": int(mine["rank"].min()), "mean_overlap": ov["mean_overlap"],
            "uniq_players": len(exp), "max_exposure": float(exp.max()),
            "n_primary_teams": stk["n_primary_teams"]}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Reverse-engineer a named entrant's portfolio from contest-standings zips.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--entrant", required=True, help="DK handle, case-insensitive.")
    p.add_argument("--slate", default=None, help="Single archive dir name (default: all).")
    p.add_argument("--mode", choices=["summary", "overlap", "exposure", "stacks"], default="summary")
    p.add_argument("--min-entries", type=int, default=2,
                   help="Ignore slates where they fired fewer than this many entries.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = PROJECT_ROOT / "archive"
    dirs = [root / args.slate] if args.slate else sorted(d for d in root.iterdir() if d.is_dir())
    out = []
    for d in dirs:
        if not list(d.glob("contest-standings-*.zip")):
            continue
        r = analyse_slate(d, args.entrant, args.mode, args.seed)
        if r and r["n"] >= args.min_entries:
            out.append(r)
    if len(out) > 1:
        t = pd.DataFrame(out).set_index("slate")
        print(f"\n\n=== {args.entrant} across {len(t)} slates ===")
        print(t.to_string(float_format=lambda x: f"{x:.3f}"))
        print("\nmeans: " + "  ".join(
            f"{c}={t[c].mean():.3f}" for c in
            ("hit99", "hit95", "mean_pct", "mean_overlap", "uniq_players", "max_exposure")))


if __name__ == "__main__":
    main()
