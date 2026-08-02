"""Build the exact grading substrate: grade the WHOLE pool, once, per slate.

tests/backtest.py re-runs the pipeline per arm and grades only the ~150
lineups that arm happened to pick. That makes every experiment cost a
pipeline run and gives each one a sample of 150 entries. But the realized
outcome of every lineup in the pool is knowable -- the standings zip contains
every player's FPTS, so every one of the ~10,000 pool lineups has an exact
realized score and therefore an exact payout in every contest that slate.

Precomputing that table changes the economics of the whole investigation:

  * A strategy's realized dollars become an index lookup. Grading goes from
    minutes to microseconds, so hundreds of arms and bootstrap resampling
    become affordable.
  * Currency quality can be measured on ~80,000 lineup-contest cells
    (10k lineups x ~10 contests x 8 slates) instead of 24 portfolio draws --
    roughly three orders of magnitude more statistical power. With only 8
    slates available (DK stops serving standings ~10 days out) that power
    difference is the difference between measuring something and guessing.

Two artifact families land in tests/backtest_output/oracle/:

  {slate}_real.npz     realized score, payout and rank for every (lineup,
                       contest), plus contest metadata and the static
                       per-lineup features that don't depend on a sim seed.
  {slate}_s{seed}_c{0|1}.npz
                       every field-relative currency (p_win, expected dollars
                       under the real payout table, P(cash), tail EV,
                       P(beat p99)) for the A and B sim halves, from
                       bt_core.accumulate_currencies.

Both are pure functions of archived inputs, so they're cached and skipped on
re-run. Usage:

    source venv/bin/activate
    python tests/backtest_oracle.py                 # all 8 slates, all seeds
    python tests/backtest_oracle.py 07262026        # one slate
    BT_SEEDS=42 python tests/backtest_oracle.py     # one seed
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.bt_core import (  # noqa: E402
    BACKTEST_SLATES, LIVE_CFG, accumulate_currencies, build_slate_context,
    grade_pool, load_real_contests, verify_slate,
)

OUT_DIR = PROJECT_ROOT / "tests" / "backtest_output"
ORACLE_DIR = OUT_DIR / "oracle"
SIM_CACHE_DIR = OUT_DIR / "sim_cache"

N_SIMS = int(os.environ.get("BT_NSIMS", LIVE_CFG["simulation"]["n_sims"]))
SHARPNESS = float(LIVE_CFG["gpp"].get("external_pool_pwin_sharpness", 0.05))
SEEDS = [int(s) for s in os.environ.get("BT_SEEDS", "42,137,4242").split(",")]


def lineup_features(pool, players_df) -> dict:
    """Static per-lineup features -- everything knowable pre-lock that does
    not depend on which sim seed was drawn. Kept alongside the realized
    outcome so a value model (or a decile-lift check) never has to rebuild a
    slate context just to read a salary sum.
    """
    by_id = players_df.set_index("player_id")
    mean = by_id["mean"].astype(float)
    own = by_id["ownership"].astype(float)
    sal = by_id["salary"].astype(float)
    team = by_id["team"].astype(str)
    pos = by_id["position"].astype(str)
    game = by_id["game"].astype(str) if "game" in by_id.columns else team

    M = len(pool.lineups)
    out = {k: np.zeros(M) for k in
           ("proj_score", "own_sum", "own_prod_log", "salary_sum",
            "max_stack", "n_hitter_teams", "n_games", "pitcher_proj")}
    for i, lu in enumerate(pool.lineups):
        pids = [int(p) for p in lu.player_ids]
        out["proj_score"][i] = mean.reindex(pids).sum()
        o = own.reindex(pids).fillna(0.1).clip(lower=0.01)
        out["own_sum"][i] = o.sum()
        out["own_prod_log"][i] = np.log(o / 100.0).sum()
        out["salary_sum"][i] = sal.reindex(pids).sum()
        p_mask = pos.reindex(pids).fillna("") == "P"
        out["pitcher_proj"][i] = mean.reindex(pids)[p_mask.values].sum()
        h_teams = team.reindex(pids)[~p_mask.values]
        vc = h_teams.value_counts()
        out["max_stack"][i] = vc.iloc[0] if len(vc) else 0
        out["n_hitter_teams"][i] = len(vc)
        out["n_games"][i] = game.reindex(pids).nunique()
    return out


def build_real(slate: str) -> Path:
    """{slate}_real.npz -- realized payout/rank for every (lineup, contest)."""
    path = ORACLE_DIR / f"{slate}_real.npz"
    if path.exists():
        return path
    d = PROJECT_ROOT / "archive" / slate
    real = load_real_contests(d)
    raw = pd.read_csv(d / "DKSalaries.csv")
    nm = raw[["ID", "Name"]].astype({"ID": str})
    fpts = verify_slate(d, real, nm)

    # calib flag is irrelevant to anything saved here (realized outcomes and
    # static features don't depend on the sim), but a context is the cheapest
    # way to get the identical pool/players_df the arms will see.
    ctx = build_slate_context(d, SEEDS[0], False, real, n_sims=N_SIMS,
                              sharpness=SHARPNESS, sim_cache_dir=SIM_CACHE_DIR,
                              want_corr=False)
    pool, contests = ctx["pool"], ctx["contests"]
    M = len(pool.lineups)

    actual = np.array([
        sum(fpts.get(int(p), float("nan")) for p in lu.player_ids)
        for lu in pool.lineups
    ], dtype=np.float64)

    gross = np.zeros((len(contests), M))
    rank = np.zeros((len(contests), M), dtype=np.int64)
    for j, c in enumerate(contests):
        gross[j], rank[j] = grade_pool(actual, c["sorted_scores"], c["payout_arr"])

    feats = lineup_features(pool, ctx["players_df"])
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        actual_score=actual,
        gross=gross, rank=rank,
        contest_id=np.array([c["contest_id"] for c in contests]),
        contest=np.array([c["contest"] for c in contests]),
        k=np.array([c["k"] for c in contests]),
        fee=np.array([float(c["fee"]) for c in contests]),
        n_field=np.array([c["n_field"] for c in contests]),
        player_ids=np.array([[int(p) for p in lu.player_ids] for lu in pool.lineups]),
        **feats,
    )
    n_amb = int(np.isnan(actual).sum())
    print(f"  {slate}: M={M} contests={len(contests)} entries={sum(c['k'] for c in contests)} "
          f"ambiguous-name drops={n_amb} best_realized={np.nanmax(actual):.1f}", flush=True)
    return path


def build_currencies(slate: str, seed: int, calibrated: bool) -> Path:
    path = ORACLE_DIR / f"{slate}_s{seed}_c{int(calibrated)}.npz"
    if path.exists():
        return path
    d = PROJECT_ROOT / "archive" / slate
    real = load_real_contests(d)
    # Only the first (seed, calib) combination per slate pays for production's
    # own compute_p_win, purely so the equivalence guard below has something
    # independent to check against; the rest take accumulate_currencies' copy.
    verify = (seed == SEEDS[0] and not calibrated)
    ctx = build_slate_context(d, seed, calibrated, real, n_sims=N_SIMS,
                              sharpness=SHARPNESS, sim_cache_dir=SIM_CACHE_DIR,
                              want_corr=False, want_pwin=verify)
    contests, exps = ctx["contests"], ctx["exponents"]
    cids = [c["contest_id"] for c in contests]

    out = {"contest_id": np.array(cids)}
    for half, (ps, fs) in (("A", (ctx["scores_A"], ctx["field_scores_A"])),
                           ("B", (ctx["scores_B"], ctx["field_scores_B"]))):
        cur = accumulate_currencies(ps, fs, contests, exps)
        for name in ("p_win", "ev_dollars", "p_cash", "ev_tail"):
            out[f"{half}_{name}"] = np.array([cur[name][c] for c in cids])
        out[f"{half}_p_beat99"] = cur["p_beat99"]
        out[f"{half}_sim_mean"] = cur["sim_mean"]
        out[f"{half}_sim_std"] = cur["sim_std"]
        out[f"{half}_sim_p99"] = cur["sim_p99"]
        # Guard: accumulate_currencies must reproduce production's
        # compute_p_win bit-for-bit, or every downstream comparison against
        # the existing arms is against a different quantity.
        ref = ctx["p_win_cull"] if half == "A" else ctx["p_win_select"]
        if ref is None:
            continue
        err = max(float(np.abs(cur["p_win"][c] - ref[c]).max()) for c in cids)
        if err > 0:
            raise SystemExit(
                f"{slate} s{seed} c{int(calibrated)} half {half}: accumulate_currencies "
                f"p_win differs from ep.compute_p_win by {err:.3g} -- the fast substrate "
                "is not the production quantity, stop and fix before grading anything."
            )
    out["proj_score"] = ctx["proj_scores"]
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def repair_real(slate: str) -> bool:
    """Regrade {slate}_real.npz in place from the stored player_ids.

    Only the realized-FPTS map can change (see
    bt_core.resolve_duplicate_names, which recovers shared-name players the
    conservative pass discarded); the pool, its features and the sim-derived
    currencies are untouched. So this rebuilds the outcome columns directly
    rather than paying for a whole slate context again. Returns True if
    anything changed.
    """
    path = ORACLE_DIR / f"{slate}_real.npz"
    if not path.exists():
        return False
    z = dict(np.load(path, allow_pickle=False))
    d = PROJECT_ROOT / "archive" / slate
    real = load_real_contests(d)
    raw = pd.read_csv(d / "DKSalaries.csv")
    nm = raw[["ID", "Name"]].astype({"ID": str})
    fpts = verify_slate(d, real, nm)

    actual = np.array([sum(fpts.get(int(p), float("nan")) for p in row)
                       for row in z["player_ids"]], dtype=np.float64)
    before = int(np.isnan(z["actual_score"]).sum())
    after = int(np.isnan(actual).sum())
    if before == after and np.allclose(np.nan_to_num(actual, nan=-1),
                                       np.nan_to_num(z["actual_score"], nan=-1)):
        return False
    by_id = {c["contest_id"]: c for c in real}
    gross, rank = z["gross"].copy(), z["rank"].copy()
    for j, cid in enumerate([str(x) for x in z["contest_id"]]):
        c = by_id[cid]
        gross[j], rank[j] = grade_pool(actual, c["sorted_scores"], c["payout_arr"])
    z["actual_score"], z["gross"], z["rank"] = actual, gross, rank
    np.savez_compressed(path, **z)
    print(f"  {slate}: ungradeable {before} -> {after} "
          f"(+{before - after} lineups recovered), best realized "
          f"{np.nanmax(actual):.1f}", flush=True)
    return True


def build_field(slate: str) -> Path:
    """{slate}_field.npz -- every real contest's own full sorted-score ladder
    and payout table, sidecar to {slate}_real.npz.

    grade_pool/grade_portfolio need the WHOLE (sorted_scores, payout_arr) per
    contest, not just the realized payout for our lineups -- grade_portfolio
    in particular has to re-derive rank/gross for an arbitrary set of our own
    entries against the real ladder, which {slate}_real.npz alone (already
    reduced to our-lineup outcomes) can't supply. Aligned to the SAME
    contest_id order as {slate}_real.npz so callers can zip the two by index
    without a second lookup; drift between the two is a hard failure, not a
    warning, since a silent misalignment would grade lineups against the
    wrong contest's payout table.
    """
    path = ORACLE_DIR / f"{slate}_field.npz"
    if path.exists():
        return path
    real_path = ORACLE_DIR / f"{slate}_real.npz"
    if not real_path.exists():
        build_real(slate)
    with np.load(real_path, allow_pickle=False) as z:
        real_cids = [str(x) for x in z["contest_id"]]

    d = PROJECT_ROOT / "archive" / slate
    real = load_real_contests(d)
    by_id = {c["contest_id"]: c for c in real}
    missing = [cid for cid in real_cids if cid not in by_id]
    if missing:
        raise SystemExit(
            f"{slate}: {real_path.name} references contest_id(s) {missing} not "
            f"found in {d}'s standings zips -- archive drift, investigate before "
            "building the field sidecar."
        )

    out: dict = {"contest_id": np.array(real_cids)}
    for j, cid in enumerate(real_cids):
        c = by_id[cid]
        out[f"scores_{j}"] = c["sorted_scores"].astype(np.float64)
        out[f"payout_{j}"] = c["payout_arr"].astype(np.float64)

    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def main() -> None:
    if sys.argv[1:2] == ["repair"]:
        for slate in ([s for s in sys.argv[2:] if s.isdigit()] or BACKTEST_SLATES):
            if (ORACLE_DIR / f"{slate}_real.npz").exists():
                if not repair_real(slate):
                    print(f"  {slate}: unchanged")
        return
    if sys.argv[1:2] == ["field"]:
        for slate in ([s for s in sys.argv[2:] if s.isdigit()] or BACKTEST_SLATES):
            build_field(slate)
            print(f"  {slate}: field sidecar ready")
        return
    slates = [s for s in sys.argv[1:] if s.isdigit()] or BACKTEST_SLATES
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    for slate in slates:
        t0 = time.time()
        build_real(slate)
        for seed in SEEDS:
            for calib in (False, True):
                build_currencies(slate, seed, calib)
        build_field(slate)
        print(f"  {slate}: currencies for {len(SEEDS)} seed(s) x2 calib "
              f"in {time.time() - t0:.0f}s", flush=True)
    print(f"\noracle -> {ORACLE_DIR}")


if __name__ == "__main__":
    main()
