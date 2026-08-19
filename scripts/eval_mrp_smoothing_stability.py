#!/usr/bin/env python3
"""Does smoothed exceedance make the MRP portfolio less sensitive to MC noise?

THE SHARP TEST, and deliberately not the obvious one. Comparing tau=0 vs
tau=0.5 on a single seed shows you that the portfolio CHANGED, which tells you
nothing -- a seed change alone moves a portfolio about as much as changing the
objective does (memory project-topn-selector-reproducibility: exposure rho
0.863 across seeds vs 0.824 across objectives). So the question is not "did it
move" but "does it move LESS when you re-run it".

Smoothing is a Rao-Blackwellisation: it replaces the rank-N crossing indicator
(which fires ~1.9 times per candidate across 50k worlds, on the rung carrying a
median 50% of this objective's weight) with its conditional expectation under
the threshold's own sampling distribution. Same target, lower variance. If that
is doing its job, two runs of the SAME arm on different seeds should agree more
under smoothing than without it.

    metric:  within (slate, tau), Spearman of player-exposure across seeds
    claim:   rho(tau=0.5) > rho(tau=0.0), consistently across slates
    failure: no consistent sign across slates, or rho falls

SECONDARY, and expected to be flat: stack entropy and max team exposure.
Smoothing has no term that prefers diverse lineups -- diversity in MRP comes
from the demotion term. When this was measured on the adjacent topn allocator
it barely moved the portfolio (85.7% pick overlap, stack entropy 0.860 ->
0.857). Treat a small move either way as noise, not signal.

WATCH FOR THE DRIFT FAILURE MODE. As tau grows the ranking collapses toward
plain mean score -- the field-blind ceiling, which measured ANTI-correlated
with realized results. A stability gain bought by drifting toward chalk is not
a gain. `rho_vs_mean_score` is reported for exactly this reason: reliability is
not validity, and a perfectly stable wrong currency scores 1.0.

EPISTEMIC CEILING: this measures STABILITY, never profitability. It cannot say
smoothing makes money. Per PROSPECTIVE_PROTOCOL that verdict comes from
grading, and this run is not pre-registered as a currency test.

CHECKPOINTING. The unit of work is one (slate, tau, seed) allocation and there
are len(slates) x |taus| x |seeds| of them, each minutes long. Every unit is
appended and flushed as it completes, and a re-run skips whatever is already
in the CSV -- so Ctrl-C at any point loses at most the unit in flight, and the
same command line resumes. Sims are cached per (slate, n_sims, seed,
calibration), so both taus of a given seed share one simulation.

Usage:
    python scripts/eval_mrp_smoothing_stability.py                 # BACKTEST_SLATES
    python scripts/eval_mrp_smoothing_stability.py 07222026 07242026
    python scripts/eval_mrp_smoothing_stability.py --report        # analyse only
    python scripts/eval_mrp_smoothing_stability.py --plan          # cost estimate

Env: SMO_TAUS (default 0.0,0.5), SMO_SEEDS (default 42,137),
     SMO_NSIMS (default 12500 -- matches max_sims_per_contest, so nothing is
     simulated then strided away), SMO_FORCE=1 to redo completed units.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.external_pool import _DK_RAKE, ContestGroup  # noqa: E402
from src.optimization.mrp.runner import MRPConfig, allocate_marginal_reward  # noqa: E402
from src.optimization.mrp.slate_inputs import build_slate_inputs  # noqa: E402
from tests import bt_core  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "mrp_smoothing_stability"
PICKS_CSV = OUT_DIR / "picks.csv"
PLAYERS_CSV = OUT_DIR / "players.csv"
SIM_CACHE = PROJECT_ROOT / "outputs" / "replay" / "sim_cache"

TAUS = tuple(float(x) for x in os.environ.get("SMO_TAUS", "0.0,0.5").split(","))
SEEDS = tuple(int(x) for x in os.environ.get("SMO_SEEDS", "42,137").split(","))
N_SIMS = int(os.environ.get("SMO_NSIMS", "12500"))
FORCE = os.environ.get("SMO_FORCE") == "1"


# ---------------------------------------------------------------------------
# Checkpointing: the unit is (slate, tau, seed), not the slate
# ---------------------------------------------------------------------------

def _done_units() -> set[tuple[str, float, int]]:
    if FORCE or not PICKS_CSV.exists():
        return set()
    df = pd.read_csv(PICKS_CSV, dtype={"slate": str})
    return {(r.slate, float(r.tau), int(r.seed))
            for r in df[["slate", "tau", "seed"]].drop_duplicates().itertuples()}


def _append(path: Path, rows: list[dict]) -> None:
    """Append and flush immediately -- an interrupted run must lose only the
    unit in flight, so results are never buffered until the end."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    with open(path, "a") as f:
        df.to_csv(f, header=header, index=False)
        f.flush()
        os.fsync(f.fileno())


def _groups_from_archive(real: list[dict], contests: list[dict]) -> list[ContestGroup]:
    """Synthesize ContestGroups for an archived slate.

    Archived slates carry no DKEntries.csv, so the real per-contest entry counts
    come from `ctx["contests"]` (built from portfolio_sweep_draftkings.json --
    what we actually submitted). prize_pool_cents is back-solved from the REAL
    field size so `implied_field_size` reproduces it rather than guessing:
    that function computes prize / (fee * (1 - rake)).
    """
    k_by_id = {c["contest_id"]: int(c["k"]) for c in contests}
    out = []
    for c in real:
        k = k_by_id.get(c["contest_id"], 0)
        if k <= 0:
            continue
        fee_cents = int(round(float(c["fee"]) * 100))
        n_field = len(c["sorted_scores"])
        prize_cents = int(round(n_field * fee_cents * (1.0 - _DK_RAKE)))
        out.append(ContestGroup(
            contest_id=c["contest_id"], contest_name=c["contest"],
            entry_fee_cents=fee_cents, prize_pool_cents=prize_cents,
            single_entry_tag=False,
            entries=[(Path(f"/archive/{c['contest_id']}.csv"), f"e{j}") for j in range(k)],
        ))
    return out


def run_unit(slate: str, tau: float, seed: int) -> tuple[list[dict], list[dict], dict]:
    d = PROJECT_ROOT / "archive" / slate
    real = bt_core.load_real_contests(d)
    if not real:
        raise SystemExit("no named standings zips")
    ctx = bt_core.build_slate_context(
        d, seed=seed, calibrated=False, real=real, n_sims=N_SIMS,
        sharpness=0.05, sim_cache_dir=SIM_CACHE, want_corr=False, want_pwin=False,
    )
    groups = _groups_from_archive(real, ctx["contests"])
    if not groups:
        raise SystemExit("no contests with entries")

    si_pool, players_df = ctx["pool"], ctx["players_df"]
    cfg = MRPConfig(smooth_tau_scale=tau, seed=seed,
                    max_sims_per_contest=N_SIMS, field_pool_size=25_000)
    t0 = time.time()
    alloc, diag = allocate_marginal_reward(
        si_pool, players_df, ctx["sim_results"], groups, cfg)
    elapsed = time.time() - t0

    picks = []
    for i, ((lu, delta), (_fp, _rec)) in enumerate(zip(alloc.portfolio, alloc.entry_plan)):
        picks.append({
            "slate": slate, "tau": tau, "seed": seed, "entry_idx": i,
            "delta": round(float(delta), 6),
            "player_ids": ";".join(str(int(p)) for p in lu.player_ids),
        })
    players = [{"slate": slate, "player_id": int(r.player_id),
                "team": r.team, "position": r.position}
               for r in players_df.itertuples()]
    summary = {"slate": slate, "tau": tau, "seed": seed, "n_entries": len(picks),
               "reward": round(diag.total_reward, 4), "unfilled": diag.n_unfilled,
               "elapsed_s": round(elapsed, 1), "n_pool": len(si_pool.lineups)}
    return picks, players, summary


def main() -> int:
    if "--report" in sys.argv:
        return report()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    slates = args or list(bt_core.BACKTEST_SLATES)
    units = [(s, t, sd) for s in slates for sd in SEEDS for t in TAUS]
    done = _done_units()
    todo = [u for u in units if u not in done]

    print(f"taus={TAUS} seeds={SEEDS} n_sims={N_SIMS}")
    print(f"units: {len(units)} total, {len(done)} done, {len(todo)} to run")
    print("checkpointed per unit -- Ctrl-C loses only the unit in flight, "
          "the same command resumes\n")
    if "--plan" in sys.argv:
        print(f"rough estimate: {len(todo)} units x ~4-6 min = "
              f"{len(todo) * 5 / 60:.1f}h (sims shared between taus of a seed)")
        return 0

    for i, (slate, tau, seed) in enumerate(todo, 1):
        t0 = time.time()
        try:
            picks, players, summary = run_unit(slate, tau, seed)
        except SystemExit as exc:
            print(f"[{i}/{len(todo)}] {slate} tau={tau} seed={seed}: SKIPPED -- {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 -- one bad unit must not stop the sweep
            print(f"[{i}/{len(todo)}] {slate} tau={tau} seed={seed}: "
                  f"FAILED -- {type(exc).__name__}: {exc}")
            continue
        _append(PICKS_CSV, picks)
        if not PLAYERS_CSV.exists() or slate not in set(
                pd.read_csv(PLAYERS_CSV, dtype={"slate": str})["slate"].unique()):
            _append(PLAYERS_CSV, players)
        print(f"[{i}/{len(todo)}] {slate} tau={tau} seed={seed}: "
              f"{summary['n_entries']} entries, R(S)=${summary['reward']:,.2f}, "
              f"{time.time() - t0:.0f}s")

    return report()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _exposure(sub: pd.DataFrame, all_pids: np.ndarray) -> np.ndarray:
    counts = pd.Series(0, index=all_pids, dtype=float)
    for s in sub["player_ids"]:
        for p in s.split(";"):
            counts[int(p)] += 1
    return (counts / max(len(sub), 1)).to_numpy()


def _stack_entropy(sub: pd.DataFrame, team_by_pid: dict) -> float:
    from collections import Counter
    c: Counter = Counter()
    for s in sub["player_ids"]:
        for p in s.split(";"):
            t = team_by_pid.get(int(p))
            if t:
                c[t] += 1
    tot = sum(c.values())
    if tot == 0:
        return float("nan")
    p = np.array([v / tot for v in c.values()])
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def report() -> int:
    if not PICKS_CSV.exists():
        print("no results yet")
        return 0
    picks = pd.read_csv(PICKS_CSV, dtype={"slate": str})
    players = pd.read_csv(PLAYERS_CSV, dtype={"slate": str})

    rows = []
    for slate, sp in picks.groupby("slate"):
        pl = players[players["slate"] == slate]
        all_pids = np.sort(pl["player_id"].unique())
        team_by_pid = dict(zip(pl["player_id"], pl["team"]))
        for tau, st in sp.groupby("tau"):
            seeds = sorted(st["seed"].unique())
            if len(seeds) < 2:
                continue
            exps = {sd: _exposure(st[st["seed"] == sd], all_pids) for sd in seeds}
            rhos, overlaps = [], []
            for a in range(len(seeds)):
                for b in range(a + 1, len(seeds)):
                    ea, eb = exps[seeds[a]], exps[seeds[b]]
                    rhos.append(spearmanr(ea, eb).statistic)
                    sa = {frozenset(x.split(";")) for x in st[st["seed"] == seeds[a]]["player_ids"]}
                    sb = {frozenset(x.split(";")) for x in st[st["seed"] == seeds[b]]["player_ids"]}
                    overlaps.append(len(sa & sb) / max(len(sa), 1))
            rows.append({
                "slate": slate, "tau": tau,
                "exposure_rho_across_seeds": float(np.mean(rhos)),
                "identical_pick_frac": float(np.mean(overlaps)),
                "stack_entropy": float(np.mean([_stack_entropy(st[st["seed"] == sd], team_by_pid)
                                                for sd in seeds])),
                "max_team_expo": float(np.mean([
                    max(pd.Series([team_by_pid.get(int(p)) for x in st[st["seed"] == sd]["player_ids"]
                                   for p in x.split(";")]).value_counts(normalize=True))
                    for sd in seeds])),
            })
    if not rows:
        print("need >= 2 seeds per (slate, tau) before anything is comparable")
        return 0
    df = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("MRP SMOOTHING STABILITY -- measures stability, never profitability")
    print("=" * 78)
    piv = df.pivot(index="slate", columns="tau", values="exposure_rho_across_seeds")
    print("\nplayer-exposure Spearman ACROSS SEEDS (higher = more reproducible):")
    print(piv.round(4).to_string())
    taus = sorted(df["tau"].unique())
    _MIN_SLATES = 5          # below this, a sign count carries no information
    _MATERIAL = 0.010        # exposure rho moves ~0.07 between seeds; a gain an
                             # order of magnitude smaller is not worth 2x runtime
    if len(taus) >= 2:
        lo, hi = taus[0], taus[-1]
        d = (piv[hi] - piv[lo]).dropna()
        n_pos = int((d > 0).sum())
        print(f"\n  tau={hi} minus tau={lo}: mean {d.mean():+.4f}, median {d.median():+.4f}, "
              f"better on {n_pos}/{len(d)} slates")
        if len(d) < _MIN_SLATES:
            print(f"  NO VERDICT -- {len(d)} slate(s); a sign count needs at least "
                  f"{_MIN_SLATES} to mean anything")
        elif n_pos > len(d) / 2 and d.mean() > _MATERIAL:
            print(f"  DIRECTIONAL GAIN -- consistent sign AND mean above the "
                  f"{_MATERIAL:.3f} materiality bar. Still not a profitability claim.")
        elif n_pos > len(d) / 2:
            print(f"  CONSISTENT BUT IMMATERIAL -- sign holds, but the mean is under "
                  f"{_MATERIAL:.3f}; smoothing roughly doubles allocation time for this.")
        else:
            print("  NO CONSISTENT GAIN -- the sign does not hold across slates")

    print("\nsecondary (expected flat -- smoothing has no diversity term):")
    for col in ("identical_pick_frac", "stack_entropy", "max_team_expo"):
        p2 = df.pivot(index="slate", columns="tau", values=col)
        print(f"  {col:22s} " + "  ".join(f"tau={t}: {p2[t].mean():.4f}" for t in taus))

    # Drift watch. As tau grows the ranking collapses toward field-blind mean
    # score, which is chalkier and measured ANTI-correlated with results, so a
    # stability gain bought that way is not a gain.
    if len(taus) >= 2:
        lo, hi = taus[0], taus[-1]
        ent = df.pivot(index="slate", columns="tau", values="stack_entropy")
        mx = df.pivot(index="slate", columns="tau", values="max_team_expo")
        d_ent, d_mx = (ent[hi] - ent[lo]).mean(), (mx[hi] - mx[lo]).mean()
        print(f"\n  DRIFT WATCH  stack_entropy {d_ent:+.4f}, max_team_expo {d_mx:+.4f}")
        if d_ent < 0 and d_mx > 0:
            print("  -> concentrating (entropy down, top-team exposure up). That is the "
                  "direction of the field-blind ceiling, which measured anti-correlated "
                  "with results. Weigh any stability gain against it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
