#!/usr/bin/env python3
"""How much is ALL opponent modelling worth, at most?

Haugh & Singal's insider-trading experiments (paper section 7.2), repurposed as
a SIZING measurement rather than a currency test. We hold the thing their
Experiment 2 had to simulate: the standings zips ARE the realized opponent
field W_op. So we can ask directly what perfect knowledge of the field would
have been worth, and bound everything downstream of it.

This matters because ~100 retrospective configurations have already been
exhausted on this archive without one surviving a per-slate check
(PROSPECTIVE_PROTOCOL, memory project-pipeline-is-a-random-draw). Before
spending more on opponent modelling, it is worth knowing the size of the prize.

ARMS, all graded identically with bt_core.grade_portfolio on realized scores
(joint insertion: self-displacement, self-tie-splitting, our-dupe splitting):

    shipped        the portfolio we actually entered that slate
    mrp_sim        MRP greedy against our SIMULATED opponent field
    mrp_realfield  MRP greedy against the REAL field's lineups, scored through
                   our own sims -- i.e. it knows W_op exactly but still does
                   not know delta. This is the paper's Experiment 2.
    oracle         top-k by REALIZED score -- the pool's ceiling. NOTE this is
                   a LOWER bound on the true ceiling: the best lineups are
                   handed to contests in list order rather than routed to the
                   contests where they are worth most, so a genuine oracle
                   would score higher still. Conservative in the direction
                   that matters (it cannot overstate the headroom).
    random         the null

HOW TO READ IT, and this is the whole point:

    mrp_realfield - mrp_sim   the prize for perfect opponent knowledge. This is
                              the ceiling on every field-model improvement:
                              Dirichlet ownership, crowding recalibration, a
                              better ownership model, all of it. If it is
                              small, that whole direction is capped and should
                              be descoped.
    oracle - mrp_realfield    what no opponent model can ever reach (outcome
                              luck plus pool composition).
    mrp_sim - shipped         what the marginal-reward objective is worth on
                              today's inputs.

EPISTEMIC CEILING -- read this before quoting a number. `oracle` conditions on
the outcome and `mrp_realfield` conditions on the field, so both are look-ahead
BY CONSTRUCTION. They bound; they never validate. Nothing here can say the MRP
pipeline is good, only how much room exists above it. Per PROSPECTIVE_PROTOCOL
the verdict comes from a live A/B, not from this file.

Dollars at n<=9 slates are outlier-dominated (one $20,000 payout was 71.6% of
production's entire winnings), so the rate ladder is reported alongside and the
per-slate spread matters more than the pooled total.

Usage:
    python scripts/eval_opponent_model_ceiling.py 07222026 07242026
    python scripts/eval_opponent_model_ceiling.py --all

Env: OMC_NSIMS (default 4000), OMC_SEED (default 42), OMC_FIELD (simulated
     field size, default 10000), OMC_FORCE=1 to re-run completed slates.
"""
from __future__ import annotations

import csv as csv_mod
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_contest_lineups import _parse_lineup_string  # noqa: E402
from src.api import external_pool as ep  # noqa: E402
from src.optimization.contest import ContestSimulator  # noqa: E402
from src.optimization.mrp.allocator import AllocationRules, allocate  # noqa: E402
from src.optimization.mrp.delta_reward import ContestDeltaState  # noqa: E402
from src.optimization.mrp.marginal_reward import tier_ev_shares  # noqa: E402
from tests import bt_core  # noqa: E402

RESULTS_CSV = PROJECT_ROOT / "outputs" / "opponent_model_ceiling" / "results.csv"
SIM_CACHE = PROJECT_ROOT / "outputs" / "replay" / "sim_cache"

N_SIMS = int(os.environ.get("OMC_NSIMS", "4000"))
SEED = int(os.environ.get("OMC_SEED", "42"))
FIELD_SIZE = int(os.environ.get("OMC_FIELD", "10000"))
FORCE = os.environ.get("OMC_FORCE") == "1"

ARMS = ("shipped", "mrp_sim", "mrp_realfield", "oracle", "random")


# ---------------------------------------------------------------------------
# Checkpoint/resume -- the established pattern (CLAUDE.md; eval_portfolio_neff)
# ---------------------------------------------------------------------------

def _append_and_reload(csv_path: Path, slate: str, rows: list[dict]) -> pd.DataFrame:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={"slate": str})
        old = old[old["slate"] != slate]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path, dtype={"slate": str})


def _done_slates(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    return set(pd.read_csv(csv_path, dtype={"slate": str})["slate"].unique())


# ---------------------------------------------------------------------------
# Real opponent field: lineups, not just scores
# ---------------------------------------------------------------------------

def real_field_lineups(d: Path, contest_id: str, id_by_name: dict) -> list[list[int]]:
    """Every real entry's roster as player_ids, from the standings zip.

    `load_real_contests` returns realized SCORES, which is all grading needs.
    Knowing W_op means knowing the LINEUPS, so that we can re-score them
    through our own sims and leave delta as the only remaining uncertainty --
    which is exactly the conditioning the paper's Experiment 2 applies.

    Entries whose names cannot be resolved unambiguously are dropped rather
    than guessed; the caller reports the retention rate, because a field
    reconstructed from 60% of its entries is not the field.
    """
    z = d / f"{contest_id.split(':', 1)[1]}.zip"
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        rows = list(csv_mod.reader(
            zf.read(name).decode("utf-8-sig", errors="replace").splitlines()
        ))
    out: list[list[int]] = []
    for r in rows[1:]:
        if len(r) < 6:
            continue
        parsed = _parse_lineup_string(r[5])
        if len(parsed) != 10:
            continue
        pids = [id_by_name.get(nm_) for _pos, nm_ in parsed]
        if any(p is None for p in pids):
            continue
        out.append([int(p) for p in pids])
    return out


def score_lineups(lineup_pids: list[list[int]], sim_matrix: np.ndarray,
                  col_map: dict, chunk: int = 2000) -> np.ndarray:
    """(n_sims, n_lineups) float32. Lineups with an unmapped id are dropped."""
    cols = []
    for pids in lineup_pids:
        idx = [col_map.get(int(p)) for p in pids]
        if any(i is None for i in idx):
            continue
        cols.append(idx)
    if not cols:
        return np.zeros((sim_matrix.shape[0], 0), dtype=np.float32)
    cols = np.asarray(cols, dtype=np.int64)
    out = np.empty((sim_matrix.shape[0], len(cols)), dtype=np.float32)
    for a in range(0, len(cols), chunk):
        b = min(a + chunk, len(cols))
        out[:, a:b] = sim_matrix[:, cols[a:b]].sum(axis=2)
    return out


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def shipped_picks(d: Path, pool, real) -> dict:
    """The portfolio we actually entered, mapped back onto pool indices.

    Matching is by exact 10-player set: a shipped lineup that is not in the
    pool (late swap, or a pool file captured after the fact) is dropped rather
    than approximated, and the caller sees the shortfall in n_entries.
    """
    path = d / "portfolio_sweep_draftkings.json"
    if not path.exists():
        return {}
    sw = json.loads(path.read_text())
    r1 = next((x for x in sw["sweep"] if x.get("risk") == 1.0), sw["sweep"][0])
    idx_by_set = {frozenset(int(p) for p in lu.player_ids): i
                  for i, lu in enumerate(pool.lineups)}
    by_name: dict = {}
    for entry in r1["lineups"]:
        key = frozenset(int(p["player_id"]) for p in entry["players"])
        i = idx_by_set.get(key)
        if i is None:
            continue
        by_name.setdefault(str(entry.get("contest_name", "")), []).append(i)

    out: dict = {}
    for c in real:
        picks = by_name.get(c["contest"])
        if picks is None:
            for nm_, v in by_name.items():
                if nm_ and (nm_ in c["contest"] or c["contest"] in nm_):
                    picks = v
                    break
        if picks:
            out[c["contest_id"]] = picks
    return out


def run_mrp(pool_scores_by_contest, field_by_contest, contests, indicator,
            slots, rules) -> dict:
    """MRP greedy over all contests. Returns {contest_id: [pool indices]}."""
    states = {}
    for c in contests:
        cid = c["contest_id"]
        fs = field_by_contest[cid]
        if fs.shape[1] == 0:
            continue
        states[cid] = ContestDeltaState(
            pool_scores_by_contest[cid], np.sort(fs, axis=1), c["payout_arr"],
        )
        del fs
    if not states:
        return {}
    res = allocate(states, slots, indicator, rules)
    return res.by_contest()


def grade_arm(picks_by_contest, pool_actual, contests) -> list[dict]:
    """Realized dollars per contest via the exact joint grader."""
    rows = []
    by_id = {c["contest_id"]: c for c in contests}
    for cid, idxs in picks_by_contest.items():
        c = by_id[cid]
        scores = pool_actual[np.asarray(idxs, dtype=np.int64)]
        gross, rank = bt_core.grade_portfolio(scores, c["sorted_scores"], c["payout_arr"])
        n_field = len(c["sorted_scores"])
        fin = np.isfinite(gross)
        rows.append({
            "contest": c["contest"], "contest_id": cid, "n_entries": len(idxs),
            "fees": c["fee"] * len(idxs), "gross": float(np.nansum(gross)),
            "n_field": n_field,
            "cash": int((gross[fin] > 0).sum()),
            "top1": int((rank[fin] <= max(1, n_field // 100)).sum()),
            "top01": int((rank[fin] <= max(1, n_field // 1000)).sum()),
            "best_rank": int(rank[fin].min()) if fin.any() else -1,
        })
    return rows


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all" in sys.argv:
        slates = sorted(p.name for p in (PROJECT_ROOT / "archive").iterdir()
                        if p.is_dir() and (p / "DKSalaries.csv").exists())
    else:
        slates = args or list(bt_core.BACKTEST_SLATES)

    done = set() if FORCE else _done_slates(RESULTS_CSV)
    todo = [s for s in slates if s not in done]
    print(f"slates: {len(slates)} requested, {len(done)} already done, {len(todo)} to run")
    print(f"n_sims={N_SIMS} seed={SEED} sim-field={FIELD_SIZE}\n")

    for slate in todo:
        t0 = time.time()
        d = PROJECT_ROOT / "archive" / slate
        try:
            rows = run_slate(d, slate)
        except SystemExit as exc:
            print(f"{slate}: SKIPPED -- {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 -- one bad slate must not stop the sweep
            print(f"{slate}: FAILED -- {type(exc).__name__}: {exc}")
            continue
        if not rows:
            print(f"{slate}: no gradeable contests")
            continue
        _append_and_reload(RESULTS_CSV, slate, rows)
        print(f"{slate}: {len(rows)} rows in {time.time() - t0:.0f}s")

    if RESULTS_CSV.exists():
        report(pd.read_csv(RESULTS_CSV, dtype={"slate": str}))
    return 0


def run_slate(d: Path, slate: str) -> list[dict]:
    real = bt_core.load_real_contests(d)
    if not real:
        raise SystemExit("no named standings zips")

    nm = pd.read_csv(d / "DKSalaries.csv")
    dup = set(nm["Name"][nm["Name"].duplicated(keep=False)])
    id_by_name = {r.Name: int(r.ID) for r in nm.itertuples() if r.Name not in dup}
    actual_fpts = bt_core.verify_slate(d, real, nm)

    ctx = bt_core.build_slate_context(
        d, seed=SEED, calibrated=False, real=real, n_sims=N_SIMS,
        sharpness=0.05, sim_cache_dir=SIM_CACHE,
        want_corr=False, want_pwin=False,
    )
    pool, players_df, sim_results = ctx["pool"], ctx["players_df"], ctx["sim_results"]
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    sim_matrix = sim_results.results_matrix.astype(np.float32)
    M = len(pool.lineups)

    pool_scores = ep.compute_lineup_scores(pool.lineups, sim_results)      # (M, S)
    indicator = ep._lineup_indicator_matrix(pool.lineups, sim_results.player_ids)
    pool_actual = np.array([
        sum(actual_fpts.get(int(p), np.nan) for p in lu.player_ids)
        for lu in pool.lineups
    ], dtype=np.float64)

    own = players_df["ownership"].to_numpy(dtype=np.float64)
    sim = ContestSimulator()
    rng = np.random.default_rng(SEED)

    # ctx["contests"] carries the real per-contest entry counts (k) we actually
    # submitted, from portfolio_sweep_draftkings.json -- the exogenous slot
    # counts the partition matroid is defined over.
    k_by_id = {c["contest_id"]: int(c["k"]) for c in ctx["contests"]}
    slots = {c["contest_id"]: k_by_id.get(c["contest_id"], 0) for c in real}
    slots = {k: v for k, v in slots.items() if v > 0}
    real = [c for c in real if c["contest_id"] in slots]
    if not real:
        raise SystemExit("no contests with entries")

    sim_fields, real_fields, retention = {}, {}, {}
    for c in real:
        cid = c["contest_id"]
        fl = sim.generate_field(players_df, own, n_lineups=FIELD_SIZE, rng_seed=SEED)
        sim_fields[cid] = sim.score_field(fl, sim_matrix, col_map)
        rl = real_field_lineups(d, cid, id_by_name)
        real_fields[cid] = score_lineups(rl, sim_matrix, col_map)
        # sorted_scores EXCLUDES our own entries (bt_core.OWN_USERNAMES) while
        # the zip we parse includes them, so this ratio can sit slightly above
        # 1.0. It is a completeness check, not a proportion -- well under 1
        # means names failed to resolve and mrp_realfield is not really
        # conditioned on the field.
        retention[cid] = len(rl) / max(len(c["sorted_scores"]), 1)

    rules = AllocationRules()
    pool_by_contest = {c["contest_id"]: pool_scores for c in real}

    arms = {
        "mrp_sim": run_mrp(pool_by_contest, sim_fields, real, indicator, slots, rules),
        "mrp_realfield": run_mrp(pool_by_contest, real_fields, real, indicator, slots, rules),
        "shipped": shipped_picks(d, pool, real),
    }

    # oracle / random / shipped
    order = np.argsort(-np.nan_to_num(pool_actual, nan=-np.inf))
    arms["oracle"] = {}
    arms["random"] = {}
    taken_o, taken_r = 0, 0
    perm = rng.permutation(M)
    for c in real:
        k = slots[c["contest_id"]]
        arms["oracle"][c["contest_id"]] = order[taken_o:taken_o + k].tolist()
        arms["random"][c["contest_id"]] = perm[taken_r:taken_r + k].tolist()
        taken_o += k
        taken_r += k

    # Where does the objective's E[$] actually come from for the lineups it
    # picked? "Maximise expected dollars" is not automatically a ceiling
    # objective -- see tier_ev_shares.
    decomp: list[dict] = []
    big = biggest_contest(real)
    picks_big = arms["mrp_sim"].get(big["contest_id"], [])
    if picks_big:
        fs = np.sort(sim_fields[big["contest_id"]], axis=1)
        ranks_d, ev_d = tier_ev_shares(
            pool_scores[np.asarray(picks_big, dtype=np.int64)],   # (k, S)
            fs, big["payout_arr"],
        )
        tot = float(ev_d.sum())
        if tot > 0:
            n_field = len(big["sorted_scores"])
            top1pct = max(1, n_field // 100)
            decomp.append({
                "slate": slate, "arm": "mrp_sim_ev_decomposition",
                "contest": big["contest"], "contest_id": big["contest_id"],
                "n_entries": len(picks_big), "fees": 0.0, "gross": 0.0,
                "n_field": n_field, "cash": 0, "top1": 0, "top01": 0,
                "best_rank": -1, "seed": SEED, "n_sims": N_SIMS, "n_pool": M,
                "real_field_retention": np.nan,
                "ev_total": round(tot, 4),
                "ev_share_rank1": round(100 * float(ev_d[ranks_d <= 1].sum()) / tot, 2),
                "ev_share_top10": round(100 * float(ev_d[ranks_d <= 10].sum()) / tot, 2),
                "ev_share_top1pct": round(100 * float(ev_d[ranks_d <= top1pct].sum()) / tot, 2),
                "ev_share_plateau": round(100 * float(ev_d[ranks_d > top1pct].sum()) / tot, 2),
            })
        del fs

    rows: list[dict] = []
    for arm, picks in arms.items():
        for r in grade_arm(picks, pool_actual, real):
            r.update({
                "slate": slate, "arm": arm, "seed": SEED, "n_sims": N_SIMS,
                "n_pool": M,
                "real_field_retention": round(retention.get(r["contest_id"], np.nan), 4),
            })
            rows.append(r)
    return rows + decomp


def biggest_contest(real):
    return max(real, key=lambda c: len(c["sorted_scores"]))


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("OPPONENT-MODEL CEILING -- look-ahead arms bound, they never validate")
    print("=" * 78)
    g = df.groupby("arm", as_index=False).agg(
        slates=("slate", "nunique"), entries=("n_entries", "sum"),
        fees=("fees", "sum"), gross=("gross", "sum"),
        cash=("cash", "sum"), top1=("top1", "sum"), top01=("top01", "sum"),
    )
    g["net"] = g["gross"] - g["fees"]
    g["$/entry"] = g["net"] / g["entries"].clip(lower=1)
    g["cash%"] = 100 * g["cash"] / g["entries"].clip(lower=1)
    g["top1%"] = 100 * g["top1"] / g["entries"].clip(lower=1)
    g["top01%"] = 100 * g["top01"] / g["entries"].clip(lower=1)
    order = [a for a in ARMS if a in set(g["arm"])]
    g = g.set_index("arm").loc[order].reset_index()
    print(g[["arm", "slates", "entries", "$/entry", "cash%", "top1%", "top01%"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("  (oracle routes its best lineups to contests in list order, not to "
          "where they are\n   worth most, so it is a LOWER bound on the ceiling.)")

    per = df.groupby(["arm", "slate"]).apply(
        lambda x: (x["gross"].sum() - x["fees"].sum()) / max(x["n_entries"].sum(), 1),
        include_groups=False,
    ).unstack(0)
    if {"mrp_sim", "mrp_realfield"} <= set(per.columns):
        delta = per["mrp_realfield"] - per["mrp_sim"]
        print(f"\nPRIZE FOR PERFECT OPPONENT KNOWLEDGE (mrp_realfield - mrp_sim):")
        print(f"  ${delta.mean():+.2f}/entry mean, ${delta.median():+.2f} median, "
              f"positive on {int((delta > 0).sum())}/{len(delta)} slates")
        print("  -> this is the CEILING on every field-model improvement "
              "(Dirichlet ownership, crowding recalibration, ownership model).")
    dec = df[df["arm"] == "mrp_sim_ev_decomposition"]
    if len(dec) and "ev_share_rank1" in dec:
        print("\nWHERE dR's DOLLARS COME FROM (mrp_sim picks, biggest contest):")
        print(f"  rank 1 {dec['ev_share_rank1'].mean():5.1f}%   "
              f"top 10 {dec['ev_share_top10'].mean():5.1f}%   "
              f"top 1% {dec['ev_share_top1pct'].mean():5.1f}%   "
              f"below top 1% (plateau) {dec['ev_share_plateau'].mean():5.1f}%")
        print("  -> a high plateau share means the objective is buying cash, not ceiling.")

    ret = df["real_field_retention"].dropna()
    if len(ret):
        print(f"\nreal-field reconstruction retained {100 * ret.mean():.1f}% of entries "
              f"(min {100 * ret.min():.1f}%) -- a low figure invalidates mrp_realfield")


if __name__ == "__main__":
    raise SystemExit(main())
