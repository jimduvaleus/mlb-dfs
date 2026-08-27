"""Post-contest sim ROI of every real entry in one archived contest.

The standings tell you what happened in the single world that occurred. This
replaces that one draw with `--n-sims` draws: rebuild the slate's simulation
from the SaberSim export (the same `build_external_players_df` +
`build_quantile_grids` path external-pool mode uses), score EVERY real entry
in the contest against EVERY other real entry in each simulated world, apply
the contest's real payout ladder with tie splitting, and report expected ROI
against the entry fee.

Nothing is inserted into the field -- every lineup graded is already in it, so
each world is a genuine re-run of the actual contest.

Two views come out:
  * top ROI lineups   -- unique lineups ranked by E[net $] / entry fee
  * top ROI by player -- entry-weighted mean ROI over every entry rostering
                         that player, next to exposure and realized FPTS

Usage
-----
    source venv/bin/activate
    python scripts/analyze_contest_sim_roi.py \
        --standings archive/08252026/main-event-warm-up.zip \
        --slate archive/08252026 --entry-fee 333 \
        --payout-table outputs/me_warmup_payouts.txt --n-sims 50000
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.api import external_pool as ep  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from sim_evaluate_portfolios import _derive_opponent  # noqa: E402
from analyze_rival_portfolio import _SPLIT, _ENTRY_SUFFIX  # noqa: E402


# ---------------------------------------------------------------------------
# Slate
# ---------------------------------------------------------------------------

def build_slate(archive_dir: Path, gpp_cfg: dict) -> tuple[pd.DataFrame, dict, dict]:
    """(players_df, quantile_grids, name->player_id).

    Same construction as sim_evaluate_portfolios.build_slate, plus a union
    with the SaberSim export: an archived DKSalaries.csv can predate a late
    roster add (08/25: Dustin Harris, rostered by 22 real entries), and a
    player missing from players_df would force those entries out of the
    field. The export carries id/name/pos/team/salary for exactly those
    players; the game string is inherited from a teammate so the copula still
    pairs the unit with its opponent.
    """
    sal = pd.read_csv(archive_dir / "DKSalaries.csv")
    slate_df = pd.DataFrame({
        "player_id": sal["ID"].astype(int),
        "name": sal["Name"].astype(str).str.strip(),
        "position": sal["Position"].astype(str).str.split("/").str[0],
        "team": sal["TeamAbbrev"].astype(str),
        "game": sal["Game Info"].astype(str),
        "salary": sal["Salary"].astype(int),
    })
    slate_df.loc[sal["Position"].astype(str).str.contains("P"), "position"] = "P"
    slate_df["eligible_positions"] = sal["Position"].astype(str)

    found = ep.discover_external_files(str(archive_dir))
    proj_path = found["projections_path"]
    if proj_path is None:
        # discover_external_files pairs a projections CSV to a lineups_*.csv and
        # returns nothing when there is no lineups file at all. This path never
        # reads the lineup pool -- only projections -- so an archived slate that
        # has the SaberSim export but no optimizer export is perfectly usable.
        cands = sorted(Path(archive_dir).glob("MLB_*_DK_*.csv"),
                       key=lambda q: q.stat().st_mtime)
        if not cands:
            raise FileNotFoundError(f"no SaberSim projections CSV in {archive_dir}")
        proj_path = cands[-1]
    proj_ext = ep.parse_player_projections(proj_path)

    raw = pd.read_csv(proj_path)
    raw["DFS ID"] = pd.to_numeric(raw["DFS ID"], errors="coerce")
    raw = raw.dropna(subset=["DFS ID"])
    known = set(slate_df["player_id"])
    team_game = (slate_df.groupby("team")["game"]
                 .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "").to_dict())
    extra = []
    for r in raw.itertuples(index=False):
        pid = int(r._0)
        if pid in known:
            continue
        team = str(r.Team)
        pos_raw = str(r.Pos)
        extra.append({
            "player_id": pid,
            "name": str(r.Name).strip(),
            "position": "P" if "P" in pos_raw else pos_raw.split("/")[0],
            "team": team,
            "game": team_game.get(team, f"{team}@{r.Opp}"),
            "salary": int(pd.to_numeric(r.Salary, errors="coerce") or 0),
            "eligible_positions": pos_raw,
        })
    if extra:
        print(f"      +{len(extra)} export-only players not in DKSalaries "
              f"(e.g. {[e['name'] for e in extra[:5]]})")
        slate_df = pd.concat([slate_df, pd.DataFrame(extra)], ignore_index=True)

    players_df = ep.build_external_players_df(
        slate_df, proj_ext, set(slate_df["player_id"]), _derive_opponent,
    )

    # Standings lineups carry bare names, so a slate with two same-named
    # players (08/25: Max Muncy LAD/ATH, Jose Fermin STL/LAA) is ambiguous.
    # Resolve to the higher-projected id: in every observed case the twin is
    # an out-of-lineup 0.00-projection row nobody could have rostered on
    # purpose, and a naive dict(zip(...)) silently keeps whichever came last.
    proj_mean = dict(zip(proj_ext["player_id"], proj_ext["mean"].fillna(0.0)))
    name_to_id: dict[str, int] = {}
    for nm, pid in zip(slate_df["name"], slate_df["player_id"]):
        pid = int(pid)
        prev = name_to_id.get(nm)
        if prev is None or proj_mean.get(pid, 0.0) > proj_mean.get(prev, 0.0):
            name_to_id[nm] = pid
    dupes = slate_df["name"][slate_df["name"].duplicated(keep=False)].unique()
    if len(dupes):
        print(f"      {len(dupes)} duplicated name(s) resolved by projection: "
              f"{list(dupes)}")
    # Grid construction mirrors the live pipeline exactly (pipeline.py:2424),
    # reading the same config keys rather than the library defaults. The one
    # that matters here is `external_pool_grid_mean_rescale` (true in
    # config.yaml): every grid is scaled so its mean equals the file's
    # `My Proj`, which is what makes a hand-edited projection an actual dial
    # on the simulated distribution instead of a decorative column. It also
    # widens admission from the +-20% band to +-2x, so 383 of 1,108 players
    # get a grid instead of 281 -- the rest fall back to Gaussian.
    grids = ep.build_quantile_grids(
        proj_ext,
        zero_inflate=bool(gpp_cfg.get("external_pool_zero_inflate", False)),
        scratch_prob=float(gpp_cfg.get("external_pool_scratch_prob", 0.02)),
        mean_calib_batter=float(gpp_cfg.get("external_pool_mean_calib_batter", 1.0)),
        mean_calib_pitcher=float(gpp_cfg.get("external_pool_mean_calib_pitcher", 1.0)),
        rescale_to_file_mean=bool(gpp_cfg.get("external_pool_grid_mean_rescale", False)),
    )
    return players_df, grids, name_to_id


# ---------------------------------------------------------------------------
# Payout ladder
# ---------------------------------------------------------------------------

# Rank numbers carry thousands separators past 1,000th ("1,281st - 2,480th"),
# so commas are stripped before the int() below.
_RANGE = re.compile(r"^\s*([\d,]+)(?:\s*(?:st|nd|rd|th))?\s*(?:-\s*([\d,]+)(?:\s*(?:st|nd|rd|th))?)?\s*$")
_MONEY = re.compile(r"^\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$")


def parse_payout_table(text: str, n_entries: int) -> np.ndarray:
    """Two-line-per-tier DK-style text ('7th - 8th' / '$12,500') -> a
    length-n_entries array of GROSS dollars indexed by rank-1."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    arr = np.zeros(n_entries, dtype=np.float64)
    i = 0
    tiers = []
    while i < len(lines) - 1:
        rm, mm = _RANGE.match(lines[i]), _MONEY.match(lines[i + 1])
        if rm and mm:
            lo = int(rm.group(1).replace(",", ""))
            hi = int(rm.group(2).replace(",", "")) if rm.group(2) else lo
            amt = float(mm.group(1).replace(",", ""))
            tiers.append((lo, hi, amt))
            i += 2
        else:
            i += 1
    if not tiers:
        raise ValueError("no payout tiers parsed")
    for lo, hi, amt in tiers:
        if lo > n_entries:
            continue
        arr[lo - 1:min(hi, n_entries)] = amt
    return arr


# ---------------------------------------------------------------------------
# Standings
# ---------------------------------------------------------------------------

def parse_standings_zip(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        rows = list(csv.reader(io.StringIO(zf.read(name).decode("utf-8-sig"))))
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
            pct = r[ci["%Drafted"]].replace("%", "").strip()
            players.append({
                "player": r[ci["Player"]].strip(),
                "roster_position": r[ci["Roster Position"]].strip(),
                "pct_drafted": float(pct) if pct else np.nan,
                "fpts": float(r[ci["FPTS"]]) if r[ci["FPTS"]].strip() else np.nan,
            })
    e = pd.DataFrame(entries)
    e["names"] = e["lineup_raw"].map(lambda s: tuple(x for x in _SPLIT.split(s)[1:][1::2]))
    e["handle"] = e["entry_name"].map(lambda s: _ENTRY_SUFFIX.sub("", s))
    return e, pd.DataFrame(players)


# ---------------------------------------------------------------------------
# Sim + payout accumulation
# ---------------------------------------------------------------------------

def ladder_bands(n_paid: int) -> dict:
    """Ladder segments as 0-based [lo, hi) rank slices, cut to THIS contest.

    Hardcoding a 670-payer tail silently drops the bottom of a deeper ladder:
    on the 10,170-entry Bat Flip (2,480 paid) it left $4.99/entry of the
    $17.21 gross EV unattributed, so the band table did not sum to the total.
    """
    edges = [(0, 1, "rank1"), (1, 10, "rank2_10"), (10, 100, "rank11_100")]
    out = {name: (lo, hi) for lo, hi, name in edges if lo < n_paid}
    if n_paid > 100:
        out[f"rank101_{n_paid}"] = (100, n_paid)
    return out


def accumulate_payouts(scores: np.ndarray, F: np.ndarray, payout: np.ndarray,
                       chunk: int, gross: np.ndarray, tally: dict,
                       bands: dict, bands_spec: dict) -> None:
    """Add each sim world's payout ladder into the running per-entry totals.

    Chunked over sim worlds: the full (n_sims x n_field) score matrix is
    ~670MB at 50k x 3335 float32 and the ranking step needs two more of them
    (CLAUDE.md: chunk whichever axis is large). Only the paying ranks are
    sorted -- argpartition splits off the top `n_paid` of the field and the sort
    runs on that slice alone, which is where nearly all the time went.
    """
    n_sims, n_field = scores.shape[0], F.shape[0]
    n_paid = int((payout > 0).sum())
    pay_head = payout[:n_paid]
    for start in range(0, n_sims, chunk):
        s = scores[start:start + chunk]
        FS = s @ F.T                                       # (c, n_field)
        top = np.argpartition(-FS, n_paid - 1, axis=1)[:, :n_paid]
        # Order the paying block itself; everything below it pays 0.
        top = np.take_along_axis(
            top, np.argsort(-np.take_along_axis(FS, top, axis=1), axis=1), axis=1)
        np.add.at(gross, top.ravel(), np.broadcast_to(pay_head, top.shape).ravel())
        # Same dollars, split by ladder segment, so EV can be attributed to
        # the steep top vs the min-cash plateau.
        for label, (lo, hi) in bands_spec.items():
            hi = min(hi, n_paid)
            if lo >= hi:
                continue
            blk = top[:, lo:hi]
            np.add.at(bands[label], blk.ravel(),
                      np.broadcast_to(payout[lo:hi], blk.shape).ravel())
        tally["win"] += np.bincount(top[:, 0], minlength=n_field)
        tally["top10"] += np.bincount(top[:, :10].ravel(), minlength=n_field)
        tally["top100"] += np.bincount(top[:, :100].ravel(), minlength=n_field)
        tally["cash"] += np.bincount(top.ravel(), minlength=n_field)
        del FS, top


def _norm_cdf(z: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standings", required=True)
    ap.add_argument("--slate", required=True, help="archive dir with DKSalaries + SaberSim export")
    ap.add_argument("--payout-table", required=True)
    ap.add_argument("--entry-fee", type=float, required=True)
    ap.add_argument("--n-sims", type=int, default=50_000)
    ap.add_argument("--sim-batch", type=int, default=25_000)
    ap.add_argument("--chunk", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs/contest_sim_roi")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    archive = Path(args.slate)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    gpp_cfg = cfg.get("gpp", {})
    print(f"[1/5] slate {archive} -> players_df + quantile grids "
          f"(grid_mean_rescale="
          f"{bool(gpp_cfg.get('external_pool_grid_mean_rescale', False))})")
    players_df, grids, name_to_id = build_slate(archive, gpp_cfg)
    print(f"      {len(players_df)} players, {len(grids)} quantile grids")

    print(f"[2/5] standings {args.standings}")
    entries, field_players = parse_standings_zip(Path(args.standings))
    n_entries = len(entries)
    payout_full = parse_payout_table(Path(args.payout_table).read_text(), n_entries)
    print(f"      {n_entries:,} entries; {int((payout_full>0).sum()):,} paid; "
          f"pool ${payout_full.sum():,.0f}; fee ${args.entry_fee:.0f}; "
          f"rake {1 - payout_full.sum()/(n_entries*args.entry_fee):.1%}")

    print(f"[3/5] simulating {args.n_sims:,} worlds "
          f"in batches of {args.sim_batch:,}")
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)

    # SimulationResults' columns are engine.players_df['player_id'] in order
    # (engine.py:181), so the entry indicator matrix can be built up front and
    # reused across every sim batch.
    pid_index = {int(p): i for i, p in enumerate(engine.players_df["player_id"])}
    missing = sorted({n for names in entries["names"] for n in names
                      if name_to_id.get(n) is None
                      or pid_index.get(name_to_id.get(n)) is None})
    if missing:
        print(f"      WARNING {len(missing)} unmodelled names: {missing[:15]}")
    ok = entries["names"].map(
        lambda ns: all(pid_index.get(name_to_id.get(n), None) is not None for n in ns)
        and len(ns) == 10)
    entries = entries[ok].reset_index(drop=True)
    print(f"      {len(entries):,}/{n_entries:,} entries modelled")
    # The ladder is re-cut to the modelled field: every paying rank and its
    # dollars are unchanged, only the unpaid tail shortens.
    payout = payout_full[:len(entries)]

    # Only players some entry actually rosters need a column: 280 of 1,108
    # here, which shrinks both the score matrix and the F @ scores product
    # by ~4x.
    used = sorted({pid_index[name_to_id[n]]
                   for names in entries["names"] for n in names})
    col_of = {j: k for k, j in enumerate(used)}
    F = np.zeros((len(entries), len(used)), dtype=np.float32)
    for r, names in enumerate(entries["names"]):
        for n in names:
            F[r, col_of[pid_index[name_to_id[n]]]] = 1.0

    print(f"[4/5] scoring + payout ladder "
          f"({len(used)} rostered players, chunk={args.chunk})")
    gross = np.zeros(len(entries), dtype=np.float64)
    tally = {k: np.zeros(len(entries), dtype=np.int64)
             for k in ("win", "top10", "top100", "cash")}
    bands_spec = ladder_bands(int((payout > 0).sum()))
    bands = {k: np.zeros(len(entries), dtype=np.float64) for k in bands_spec}
    sim_sum = np.zeros(len(used), dtype=np.float64)
    tail_samples = []
    done = 0
    np.random.seed(args.seed)
    while done < args.n_sims:
        b = min(args.sim_batch, args.n_sims - done)
        sim = engine.simulate(b)
        sc = sim.results_matrix[:, used].astype(np.float32)
        accumulate_payouts(sc, F, payout, args.chunk, gross, tally, bands, bands_spec)
        sim_sum += sc.sum(axis=0)
        if len(tail_samples) < 4:
            tail_samples.append(sc[:min(5000, b)].copy())
        done += b
        print(f"      {done:,}/{args.n_sims:,} worlds")
        del sim, sc
    player_scores = np.concatenate(tail_samples, axis=0)
    sim_mean = sim_sum / args.n_sims
    mean_gross = gross / args.n_sims
    tally = {k: v / args.n_sims for k, v in tally.items()}
    bands = {k: v / args.n_sims for k, v in bands.items()}

    entries["ev_gross"] = mean_gross
    entries["ev_net"] = mean_gross - args.entry_fee
    entries["roi"] = entries["ev_net"] / args.entry_fee
    for k, v in tally.items():
        entries[f"p_{k}"] = v
    for k, v in bands.items():
        entries[f"ev_{k}"] = v
    # DK gives every tied entry the SAME (lowest) rank, so a naive
    # payout_full[rank-1] lookup pays each of them the top of their band and
    # never pays the slots underneath. Split the band the way DK does: a tie of
    # k entries at rank r occupies ranks r..r+k-1 and each takes the mean.
    # (The simulated payouts need none of this -- sim scores are continuous, so
    # exact ties have probability zero.)
    _rank = entries["rank"].astype(int).to_numpy()
    _tie_n = pd.Series(_rank).map(pd.Series(_rank).value_counts()).to_numpy()
    _cum = np.concatenate([[0.0], np.cumsum(payout_full)])
    _lo = _rank - 1
    _hi = np.minimum(_lo + _tie_n, len(payout_full))
    entries["realized_payout"] = (_cum[_hi] - _cum[_lo]) / np.maximum(_hi - _lo, 1)

    print("[5/5] aggregating")
    # DK's sidebar splits by ROSTER POSITION, not by player: a multi-eligible
    # player gets one row per slot he was used in (Ohtani 08/25: 12.56% at P +
    # 8.99% at OF). %Drafted must therefore be SUMMED across his rows -- keeping
    # only the first understates 35 of the 245 rostered players here, Ohtani by
    # 9 points. FPTS is identical on every row of a player, so first() is right
    # for that one.
    fp = field_players.groupby("player").agg(
        pct_drafted=("pct_drafted", "sum"), fpts=("fpts", "first"))
    dk_own = fp["pct_drafted"].to_dict()
    dk_fpts = fp["fpts"].to_dict()
    # Keyed by player_id, not name: a name-keyed map re-introduces exactly the
    # collision `name_to_id` just resolved, and would label the simulated LAD
    # Max Muncy with the ATH twin's team and salary.
    pid_team = dict(zip(players_df["player_id"].astype(int), players_df["team"]))
    pid_sal = dict(zip(players_df["player_id"].astype(int), players_df["salary"]))

    rows = []
    for name, pid in name_to_id.items():
        j = pid_index.get(int(pid))
        if j is None or j not in col_of:
            continue
        k = col_of[j]
        m = F[:, k] > 0
        n_in = int(m.sum())
        if n_in == 0:
            continue
        rows.append({
            "player": name,
            "team": pid_team.get(int(pid), ""),
            "salary": pid_sal.get(int(pid), np.nan),
            "n_entries": n_in,
            "exposure_pct": 100.0 * n_in / len(entries),
            "dk_drafted_pct": float(dk_own.get(name, np.nan)),
            "roi": float(entries.loc[m, "roi"].mean()),
            "ev_net": float(entries.loc[m, "ev_net"].mean()),
            "p_win": float(entries.loc[m, "p_win"].mean()),
            "p_top100": float(entries.loc[m, "p_top100"].mean()),
            "p_cash": float(entries.loc[m, "p_cash"].mean()),
            "sim_mean_fpts": float(sim_mean[k]),
            "sim_p99_fpts": float(np.percentile(player_scores[:, k], 99)),
            "realized_fpts": float(dk_fpts.get(name, np.nan)),
        })
    pl = pd.DataFrame(rows)

    # `roi` above is the mean ROI of every entry rostering the player, which
    # is confounded by who else those entries rostered -- a cheap punt next to
    # a popular ace inherits the ace's ROI. `roi_partial` is the ridge
    # coefficient from regressing entry ROI on the 245 roster indicators, i.e.
    # the player's marginal contribution holding the other nine slots fixed.
    # Ridge, not OLS: every lineup has exactly 10 players, so the indicator
    # columns sum to a constant and the design is exactly collinear with the
    # intercept.
    y = entries["roi"].to_numpy(dtype=np.float64)
    X = F.astype(np.float64)
    Xc = X - X.mean(axis=0)
    lam = 1.0
    beta = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]),
                           Xc.T @ (y - y.mean()))
    partial = {name: float(beta[col_of[pid_index[name_to_id[name]]]])
               for name in pl["player"]}
    pl["roi_partial"] = pl["player"].map(partial)
    pl = pl.sort_values("roi", ascending=False)

    lu = entries.copy()
    lu["lineup_key"] = lu["names"].map(lambda ns: "|".join(sorted(ns)))
    uniq = (lu.groupby("lineup_key")
              .agg(n_entries=("rank", "size"),
                   roi=("roi", "first"), ev_net=("ev_net", "first"),
                   ev_gross=("ev_gross", "first"), p_win=("p_win", "first"),
                   p_top10=("p_top10", "first"), p_top100=("p_top100", "first"), p_cash=("p_cash", "first"),
                   best_rank=("rank", "min"), points=("points", "first"),
                   lineup=("lineup_raw", "first"),
                   handles=("handle", lambda s: ", ".join(sorted(set(s))[:4])))
              .reset_index()
              .sort_values("roi", ascending=False))

    # Per-handle: the unit an entrant actually plays is the whole portfolio,
    # so the headline number is expected NET DOLLARS PER CONTEST (their entire
    # entry block, not one lineup). p_win sums exactly -- only one entry wins a
    # world -- so summing per-entry win rates over a handle IS P(handle wins).
    fee = args.entry_fee
    hd = (entries.groupby("handle")
          .agg(n_entries=("roi", "size"),
               ev_gross=("ev_gross", "sum"), ev_net=("ev_net", "sum"),
               p_win=("p_win", "sum"), e_top10=("p_top10", "sum"),
               e_top100=("p_top100", "sum"), e_cash=("p_cash", "sum"),
               realized_gross=("realized_payout", "sum"),
               best_rank=("rank", "min"), best_points=("points", "max"))
          .reset_index())
    hd["cost"] = fee * hd["n_entries"]
    hd["roi"] = hd["ev_net"] / hd["cost"]
    hd["ev_net_per_entry"] = hd["ev_net"] / hd["n_entries"]
    hd["realized_net"] = hd["realized_gross"] - hd["cost"]
    hd["realized_roi"] = hd["realized_net"] / hd["cost"]
    hd["cash_rate"] = hd["e_cash"] / hd["n_entries"]
    hd = hd.sort_values("ev_net", ascending=False)
    hd.to_csv(out_dir / "handle_sim_roi.csv", index=False)

    entries.drop(columns=["names"]).to_csv(out_dir / "entries_sim_roi.csv", index=False)
    uniq.to_csv(out_dir / "unique_lineups_sim_roi.csv", index=False)
    pl.to_csv(out_dir / "player_sim_roi.csv", index=False)
    print(f"      wrote {out_dir}/{{entries,unique_lineups,player,handle}}_sim_roi.csv")

    pd.set_option("display.width", 250)
    real_net = entries["realized_payout"].sum() - args.entry_fee * len(entries)
    print(f"\nField-wide: mean ROI {entries['roi'].mean():+.1%}  "
          f"median {entries['roi'].median():+.1%}  "
          f"share +EV {100*(entries['roi']>0).mean():.1f}%  "
          f"(realized field net ${real_net:,.0f} = {real_net/(args.entry_fee*len(entries)):+.1%})")
    print(f"Unique lineups: {len(uniq):,} of {len(entries):,} entries; "
          f"{100*(uniq['roi']>0).mean():.1f}% of unique lineups are +EV")

    # --- Why so many entries clear the rake -------------------------------
    # The +EV share is not bounded by the 20.1% cash rate: cash probability is
    # near-identical across the field (every lineup is ~20% to min-cash), and
    # min-cashing only returns $500 on a $333 ticket. What separates lineups is
    # ceiling equity, and the ladder is steep enough that the top bands carry
    # most of the EV -- so the +EV/-EV split is decided in the tail, not at the
    # cash line.
    q = entries["roi"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"\n=== ROI DISTRIBUTION ACROSS THE {len(entries):,} REAL ENTRIES ===")
    print("  " + "  ".join(f"p{int(k*100)}={v:+.1%}" for k, v in q.items())
          + f"   sd={entries['roi'].std():.3f}")
    print(f"  share +EV {100*(entries['roi']>0).mean():.1f}%  "
          f"(a normal at mean {entries['roi'].mean():+.3f}, "
          f"sd {entries['roi'].std():.3f} would give "
          f"{100*(1 - _norm_cdf((0-entries['roi'].mean())/entries['roi'].std())):.1f}%)")
    print("\n=== WHERE THE EXPECTED DOLLARS COME FROM (field mean per entry) ===")
    tot = float(entries["ev_gross"].mean())
    for k in bands_spec:
        v = float(entries[f"ev_{k}"].mean())
        print(f"  {k:<12} ${v:7.2f}/entry   {v/tot:6.1%} of gross EV")
    print(f"  {'TOTAL':<12} ${tot:7.2f}/entry   (fee ${args.entry_fee:.0f})")
    tail_key = next((k for k in bands_spec if k.startswith("rank101_")), None)
    top_only = sum(entries[f"ev_{k}"] for k in bands_spec if k != tail_key)
    line = (f"  ranks 1-100 alone: ${top_only.mean():.2f}/entry = "
            f"{top_only.mean()/tot:.1%} of gross EV")
    if tail_key:
        line += (f", and its spread across entries (sd ${top_only.std():.2f}) is "
                 f"{top_only.std()/entries['ev_'+tail_key].std():.1f}x the "
                 f"min-cash band's")
    print(line + " -- the tail decides who clears the rake.")
    if tail_key:
        print(f"  share of entries +EV counting ONLY the min-cash band "
              f"({tail_key.replace('rank','ranks ').replace('_','-')}): "
              f"{100*(entries['ev_'+tail_key] > args.entry_fee).mean():.1f}%")

    hd_cols = ["handle", "n_entries", "cost", "ev_net", "roi",
               "ev_net_per_entry", "p_win", "e_top10", "e_top100", "cash_rate",
               "realized_net", "realized_roi", "best_rank"]
    print("\n=== TOP 25 DK HANDLES BY EXPECTED NET $ / CONTEST ===")
    print(hd.head(25)[hd_cols].to_string(
        index=False, float_format=lambda x: f"{x:,.4f}"))
    print("\n=== BOTTOM 15 DK HANDLES BY EXPECTED NET $ / CONTEST ===")
    print(hd.tail(15)[hd_cols].to_string(
        index=False, float_format=lambda x: f"{x:,.4f}"))
    for floor in (25, 100):
        sub = hd[hd.n_entries >= floor].sort_values("roi", ascending=False)
        print(f"\n=== TOP 20 DK HANDLES BY ROI (>= {floor} entries, "
              f"{len(sub)} qualify) ===")
        print(sub.head(20)[hd_cols].to_string(
            index=False, float_format=lambda x: f"{x:,.4f}"))

    lu_cols = ["roi", "ev_net", "p_win", "p_top10", "p_top100", "p_cash",
               "n_entries", "best_rank", "points", "handles", "lineup"]
    print("\n=== TOP 30 UNIQUE LINEUPS BY SIM ROI ===")
    print(uniq.head(30)[lu_cols].to_string(
        index=False, float_format=lambda x: f"{x:,.4f}"))
    print("\n=== TOP 15 UNIQUE LINEUPS BY WIN EQUITY (P of taking the $200k) ===")
    print(uniq.sort_values("p_win", ascending=False).head(15)[lu_cols].to_string(
        index=False, float_format=lambda x: f"{x:,.4f}"))

    pl_cols = ["player", "team", "salary", "n_entries", "exposure_pct",
               "dk_drafted_pct", "roi", "roi_partial", "ev_net", "p_top100",
               "p_cash", "sim_mean_fpts", "sim_p99_fpts", "realized_fpts"]
    for floor in (15, 100):
        sub = pl[pl.n_entries >= floor]
        print(f"\n=== TOP 30 PLAYERS BY SIM ROI (>= {floor} entries, "
              f"{len(sub)} qualify) ===")
        print(sub.head(30)[pl_cols].to_string(
            index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\n=== TOP 25 PLAYERS BY PARTIAL (CO-ROSTER-ADJUSTED) ROI "
          "(>= 15 entries) ===")
    print(pl[pl.n_entries >= 15].sort_values("roi_partial", ascending=False)
          .head(25)[pl_cols].to_string(
              index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\n=== BOTTOM 20 PLAYERS BY SIM ROI (>= 15 entries) ===")
    print(pl[pl.n_entries >= 15].tail(20)[pl_cols].to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":
    main()
