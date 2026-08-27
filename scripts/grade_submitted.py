"""Grade an actually-submitted portfolio against each contest's REAL field.

Tier 2 of three: real human lineup COMPOSITIONS from the standings zips, scored
over simulated worlds. That matters because the alternative -- ContestSimulator's
ownership-sampled proxy field -- is exactly where the model flatters itself:
projected ownership missed Gavin Williams by 26 points on 08/25, so a simulated
field under-represents the crowding that actually happened.

Worlds are drawn with a seed unrelated to selection, so this is out-of-sample on
the sim as well as on the field.

Portfolio mode: each contest's entries are inserted TOGETHER and ranked against
the real field plus their own team-mates, so demotion between your own lineups
is priced. Marginal mode is reported alongside only to show the gap.

Baseline for "good": the field's own mean ROI is -rake by construction, so that
is the number to beat, not zero.
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

from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.api.dk_entries import parse_entry_file  # noqa: E402
from src.optimization.multi_contest import resolve_contest_slots  # noqa: E402
from analyze_contest_sim_roi import build_slate  # noqa: E402
import portfolio_grading as pg  # noqa: E402

_POS = ["P", "C", "1B", "2B", "3B", "SS", "OF"]
_SPLIT = re.compile(r"\s*\b(" + "|".join(_POS) + r")\b\s+")


def field_from_zip(zip_path, pid_index, name_to_id):
    """(Ffield, n_entries). Rows whose players don't resolve are dropped and
    reported -- a late roster add can leave a handful unmodellable."""
    with zipfile.ZipFile(zip_path) as zf:
        fn = next(n for n in zf.namelist() if n.endswith(".csv"))
        rows = list(csv.reader(io.StringIO(zf.read(fn).decode("utf-8-sig"))))
    ci = {c: i for i, c in enumerate(rows[0]) if c}
    lus, miss = [], 0
    for r in rows[1:]:
        if len(r) <= ci["Lineup"] or not r[ci["Lineup"]].strip():
            continue
        names = _SPLIT.split(r[ci["Lineup"]])[1:][1::2]
        ids = [pid_index.get(name_to_id.get(n)) for n in names]
        if len(ids) != 10 or any(i is None for i in ids):
            miss += 1
            continue
        lus.append(ids)
    F = np.zeros((len(lus), len(pid_index)), dtype=np.float32)
    for r, ids in enumerate(lus):
        for j in ids:
            F[r, j] = 1.0
    return F, len(lus), miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", required=True)
    ap.add_argument("--uploads", nargs="+", required=True)
    ap.add_argument("--entries", nargs="+", required=True)
    ap.add_argument("--zip-dir", required=True)
    ap.add_argument("--n-sims", type=int, default=60_000)
    ap.add_argument("--sim-batch", type=int, default=20_000)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--seed", type=int, default=90_210)
    ap.add_argument("--out", default="outputs/submitted_grade.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    players_df, grids, name_to_id = build_slate(Path(args.slate), cfg.get("gpp", {}))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    pid_index = {int(p): i for i, p in enumerate(engine.players_df["player_id"])}

    slots = {s.contest_name: s for s in resolve_contest_slots(
        [(Path(e), parse_entry_file(Path(e))) for e in args.entries])}

    # submitted lineups, grouped by contest, straight from the upload files
    # DK upload headers repeat column names (P,P,...,OF,OF,OF), so DictReader
    # collapses them and loses eight of the ten slots. Read positionally.
    sub: dict[str, list] = {}
    for up in args.uploads:
        rows_u = list(csv.reader(open(up)))
        hdr = rows_u[0]
        c_name = hdr.index("Contest Name")
        for r in rows_u[1:]:
            if len(r) < 14 or not r[c_name].strip():
                continue
            ids = [int(v) for v in r[4:14] if str(v).strip().isdigit()]
            if len(ids) == 10:
                sub.setdefault(r[c_name].strip(), []).append(ids)

    zips = {p.stem.replace("-", " ").lower(): p
            for p in Path(args.zip_dir).glob("*.zip")}

    rows = []
    for nm, lus in sub.items():
        slot = slots.get(nm)
        if slot is None:
            print(f"  {nm}: no resolved ladder, skipped"); continue
        # zip stems are the contest's distinctive words ("four-seamer",
        # "mini-max"); match the whole stem against the normalised name so
        # "base hit" cannot also match "bat flip".
        low = nm.lower().replace("-", " ")
        cand = [k for k in zips if k in low]
        if not cand:
            print(f"  {nm}: no standings zip, skipped"); continue
        zp = zips[max(cand, key=len)]
        Ff, n_field, miss = field_from_zip(zp, pid_index, name_to_id)
        payout = np.asarray(slot.payout_arr, dtype=np.float64)[:n_field]
        X = np.zeros((len(lus), len(pid_index)), dtype=np.float32)
        for r, ids in enumerate(lus):
            for p in ids:
                X[r, pid_index[int(p)]] = 1.0
        fee = slot.entry_fee
        rake = 1.0 - payout.sum() / (n_field * fee)
        print(f"  {nm[:38]:<38} k={len(lus):>3} field={n_field:,} "
              f"({miss} unmodelled) rake {rake:+.1%}")
        port, marg = pg.grade_portfolios_multi(
            engine, {"sub": X}, Ff, payout, args.n_sims,
            sim_batch=args.sim_batch, chunk=args.chunk, seed=args.seed, progress=False)
        sp = pg.summarize(port["sub"], fee); sm = pg.summarize(marg, fee)
        rows.append({
            "contest": nm, "k": len(lus), "field": n_field, "fee": fee,
            "rake": rake, "portfolio_roi": sp["roi"], "marginal_roi": sm["roi"],
            "edge_vs_field": sp["roi"] - (-rake),
            "net": sp["net"], "cost": sp["cost"],
            "pct_lineups_pos": sp["pct_lineups_positive"],
            "self_comp": sm["gross"] - sp["gross"],
        })
    d = pd.DataFrame(rows).sort_values("edge_vs_field", ascending=False)
    d.to_csv(PROJECT_ROOT / args.out, index=False)
    print("\n=== submitted portfolio vs REAL fields (fresh worlds) ===")
    print(d.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    tot_c, tot_n = d.cost.sum(), d.net.sum()
    print(f"\nACROSS ALL CONTESTS: cost ${tot_c:,.2f}  expected net ${tot_n:,.2f}  "
          f"ROI {tot_n/tot_c:+.1%}")
    print(f"field baseline (weighted -rake): "
          f"{-(d.rake*d.cost).sum()/tot_c:+.1%}")


if __name__ == "__main__":
    main()
