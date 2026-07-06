"""
PIT calibration of the simulation's tails against realized results, across
the contest archive — the diagnostic that localises the "model p99 worlds
!= real p99 worlds" problem (sim-p99 candidate ranking anti-selects real
ceiling; see memory/pool-ceiling findings).

For every archived slate with market_odds_projections.csv + DKSalaries.csv +
a contest-standings zip, replay the simulation (empirical copula + PCA
mixture/Gaussian marginals, archived means/stds/batting slots) and compute
probability integral transforms of what actually happened:

  player PIT     — where each player's realized FPTS falls in his simulated
                   marginal. Uniform(0,1) if marginals are calibrated.
  team-sum PIT   — where each team's realized batter-sum falls in the
                   simulated distribution of that same sum. Uniform if the
                   *joint* (dependence) structure is calibrated. Restricted
                   to teams whose 9 batting slots were all slot_confirmed at
                   fetch time, so scratches don't masquerade as thin tails.
  slate-sum PIT  — realized sum over all confirmed teams vs its simulated
                   distribution: tests cross-game dependence (the sim samples
                   games independently).

The player-vs-team contrast is the point: calibrated player PITs with
heavy-tailed team PITs (mass piled at 0/1, actual exceeding sim p99 far more
than 1% of the time) means the within-game dependence (copula/env coupling)
is too weak, independent of marginal quality.

Note the replay uses the parametric marginal path (PCA mixture + Gaussian),
not the production market-implied quantile grids (not archived) — player PIT
judges *this replay's* marginals; the contrast argument is unaffected.

Usage
-----
    python scripts/diagnose_sim_tails.py                  # all eligible slates
    python scripts/diagnose_sim_tails.py --recent 10
    python scripts/diagnose_sim_tails.py --n-sims 4000

Output
------
  - Aggregate PIT tables printed to stdout.
  - archive/sim_tail_pits.csv (one row per player-slate and team-slate PIT).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_candidate_pool as acp  # noqa: E402 — _slate_sort_key
from evaluate_ownership import _parse_contest_zip, _normalise  # noqa: E402

from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.models.batter_model import BatterPCAModel  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402


def load_actuals(archive_dir: Path, slate_df: pd.DataFrame) -> dict[int, float]:
    """player_id -> realized FPTS via the standings zip sidebar (covers every
    archived slate), name-matched; ambiguous duplicate names dropped."""
    zips = sorted(archive_dir.glob("contest-standings-*.zip"))
    _, ownership_df = _parse_contest_zip(zips[0])
    sidebar = {}
    for r in ownership_df.itertuples(index=False):
        if r.actual_fpts is None or pd.isna(r.actual_fpts):
            continue
        sidebar[_normalise(str(r.player_name))] = float(r.actual_fpts)
    names = slate_df["name"].astype(str).map(_normalise)
    dup = names.value_counts().loc[lambda s: s > 1].index
    out = {}
    for pid, nm in zip(slate_df["player_id"].astype(int), names):
        if nm in dup:
            continue
        v = sidebar.get(nm)
        if v is not None:
            out[pid] = v
    return out


def randomized_pit(sim_col: np.ndarray, actual: float, rng: np.random.Generator) -> float:
    """PIT with randomized tie-breaking (keeps atoms — e.g. the batter mass
    at 0 — from distorting uniformity)."""
    lo = float((sim_col < actual).mean())
    eq = float((sim_col == actual).mean())
    return lo + rng.random() * eq


def analyze_slate(
    archive_dir: Path, copula, pca, grid, n_sims: int, rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict | None]:
    proj_path = archive_dir / "market_odds_projections.csv"
    sal_path = archive_dir / "DKSalaries.csv"
    proj = pd.read_csv(proj_path)
    slate = DraftKingsSlateIngestor(str(sal_path)).get_slate_dataframe()
    slate["player_id"] = slate["player_id"].astype(int)

    df = slate.merge(
        proj[["player_id", "mean", "std_dev", "lineup_slot", "slot_confirmed"]],
        on="player_id", how="inner",
    )
    df = df[df["lineup_slot"].notna()].copy()
    df["slot"] = df["lineup_slot"].astype(int)
    df = df[(df["slot"] >= 1) & (df["slot"] <= 10)].reset_index(drop=True)

    actuals = load_actuals(archive_dir, slate)
    engine = SimulationEngine(copula, df, batter_pca_model=pca, score_grid=grid)
    res = engine.simulate(n_sims)
    mat = res.results_matrix  # (n_sims, n_players)
    col = {int(p): i for i, p in enumerate(res.player_ids)}

    player_rows = []
    for r in df.itertuples(index=False):
        pid = int(r.player_id)
        a = actuals.get(pid)
        if a is None:
            continue
        player_rows.append({
            "slate": archive_dir.name,
            "player_id": pid,
            "team": r.team,
            "slot": int(r.slot),
            "is_batter": 1 <= int(r.slot) <= 9,
            "confirmed": bool(r.slot_confirmed),
            "actual": a,
            "pit": randomized_pit(mat[:, col[pid]], a, rng),
        })

    team_rows = []
    dep_rows = []  # per confirmed team-game: batter PIT vector + sim dependence
    slate_sim = np.zeros(n_sims)
    slate_actual = 0.0
    n_slate_teams = 0
    for team, g in df[(df["slot"] >= 1) & (df["slot"] <= 9)].groupby("team"):
        if len(g) != 9 or not g["slot_confirmed"].all():
            continue
        g = g.sort_values("slot")
        pids = [int(p) for p in g["player_id"]]
        if any(p not in actuals for p in pids):
            continue
        a_sum = sum(actuals[p] for p in pids)
        cols = [col[p] for p in pids]
        s_sum = mat[:, cols].sum(axis=1)
        team_rows.append({
            "slate": archive_dir.name,
            "team": team,
            "actual_sum": a_sum,
            "sim_p50": float(np.median(s_sum)),
            "sim_p99": float(np.percentile(s_sum, 99)),
            "pit": randomized_pit(s_sum, a_sum, rng),
        })
        # Realized batter PIT vector (slot order) and the sim's own mean
        # pairwise teammate Spearman for the same 9 columns. Comparing the
        # two pools is marginal-robust: PIT is monotone in the actual score,
        # so rank correlations survive marginal miscalibration.
        pit_vec = [randomized_pit(mat[:, c], actuals[p], rng) for p, c in zip(pids, cols)]
        sub = mat[:, cols]
        ranks = np.argsort(np.argsort(sub, axis=0), axis=0).astype(np.float64)
        cm = np.corrcoef(ranks.T)
        sim_rho = float(cm[np.triu_indices(9, k=1)].mean())
        # Opposing starter (this unit's slot-10 pitcher) for the
        # batter-vs-opposing-pitcher dependence measurement.
        opp = g["opponent"].iloc[0]
        opp_p = df[(df["team"] == opp) & (df["slot"] == 10)]
        opp_pit = None
        if len(opp_p) == 1:
            ppid = int(opp_p["player_id"].iloc[0])
            if ppid in actuals:
                opp_pit = randomized_pit(mat[:, col[ppid]], actuals[ppid], rng)
        dep_rows.append({"slate": archive_dir.name, "team": team,
                         "pits": pit_vec, "sim_rho": sim_rho,
                         "opp_pitcher_pit": opp_pit})
        slate_sim += s_sum
        slate_actual += a_sum
        n_slate_teams += 1

    slate_row = None
    if n_slate_teams >= 4:
        slate_row = {
            "slate": archive_dir.name,
            "n_teams": n_slate_teams,
            "pit": randomized_pit(slate_sim, slate_actual, rng),
        }
    return player_rows, team_rows, slate_row, dep_rows


def tail_table(pits: np.ndarray, label: str) -> None:
    n = len(pits)
    print(
        f"{label:>28}: n={n:>5}  <0.01: {np.mean(pits < 0.01):.1%}  <0.05: {np.mean(pits < 0.05):.1%}  "
        f">0.95: {np.mean(pits > 0.95):.1%}  >0.99: {np.mean(pits > 0.99):.1%}  "
        f"(uniform expects 1% / 5% / 5% / 1%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIT-calibrate simulated tails against realized results.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("--recent", type=int, default=0, metavar="N")
    parser.add_argument("--n-sims", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    archive_root = PROJECT_ROOT / "archive"
    dirs = sorted(
        (d for d in archive_root.iterdir()
         if d.is_dir()
         and (d / "market_odds_projections.csv").exists()
         and (d / "DKSalaries.csv").exists()
         and list(d.glob("contest-standings-*.zip"))),
        key=lambda d: acp._slate_sort_key(d.name),
    )
    if args.recent:
        dirs = dirs[-args.recent:]
    print(f"Replaying {len(dirs)} slates at n_sims={args.n_sims}")

    copula = EmpiricalCopula(str(PROJECT_ROOT / "data/processed/empirical_copula.parquet"))
    pca = BatterPCAModel.load(str(PROJECT_ROOT / "data/processed/batter_pca_model.npz"))
    grid = np.load(PROJECT_ROOT / "data/processed/batter_score_grid.npy")
    rng = np.random.default_rng(args.seed)

    players, teams, slates, deps = [], [], [], []
    for d in dirs:
        try:
            p, t, s, dep = analyze_slate(d, copula, pca, grid, args.n_sims, rng)
        except Exception as exc:
            print(f"  {d.name}: skipped ({exc})")
            continue
        players.extend(p)
        teams.extend(t)
        if s:
            slates.append(s)
        deps.extend(dep)
        print(f"  {d.name}: {len(p)} player PITs, {len(t)} confirmed-team PITs")

    pdf = pd.DataFrame(players)
    tdf = pd.DataFrame(teams)
    print("\n=== PIT tail masses ===")
    # Confirmed players only for the marginal test: unconfirmed rows are
    # mostly bench players the sim projected but who never played (actual 0),
    # which floods the low tail with non-calibration mass.
    conf = pdf[pdf["confirmed"]]
    tail_table(conf[conf["is_batter"]]["pit"].values, "confirmed batters (marginals)")
    tail_table(conf[~conf["is_batter"]]["pit"].values, "confirmed pitchers (marginals)")
    tail_table(tdf["pit"].values, "team 9-batter sums (JOINT)")
    if slates:
        tail_table(pd.DataFrame(slates)["pit"].values, "slate sums (cross-game)")

    if len(tdf):
        exceed = (tdf["actual_sum"] > tdf["sim_p99"]).mean()
        print(f"\nTeam sums exceeding sim p99: {exceed:.1%} (calibrated = 1.0%)")

    # --- Dependence comparison (marginal-robust) ---
    # Realized: pooled Spearman between teammate PITs across team-games,
    # averaged over the 36 batting-slot pairs. Sim: mean pairwise teammate
    # Spearman of the simulated score columns (averaged over team-games).
    if deps:
        pit_mat = np.array([r["pits"] for r in deps])  # (n_team_games, 9)
        n_tg = len(pit_mat)
        rhos = []
        for i in range(9):
            for j in range(i + 1, 9):
                a = pd.Series(pit_mat[:, i]).rank().values
                b = pd.Series(pit_mat[:, j]).rank().values
                rhos.append(np.corrcoef(a, b)[0, 1])
        realized_rho = float(np.mean(rhos))
        sim_rho = float(np.mean([r["sim_rho"] for r in deps]))
        se = float(np.std(rhos) / np.sqrt(len(rhos)))
        print(
            f"\nTeammate rank dependence (batter-batter, {n_tg} confirmed team-games):\n"
            f"  realized mean pairwise Spearman: {realized_rho:+.4f} (±{se:.4f} over slot pairs)\n"
            f"  simulated mean pairwise Spearman: {sim_rho:+.4f}\n"
            f"  (sim < realized ⇒ within-team dependence too weak — the joint\n"
            f"   tail understates team explosions/duds even with perfect marginals)"
        )
        # Batter vs opposing starting pitcher (same unit, slot 10)
        with_p = [(r["pits"], r["opp_pitcher_pit"]) for r in deps if r["opp_pitcher_pit"] is not None]
        if with_p:
            bp = []
            pvec = np.array([p for _, p in with_p])
            bmat = np.array([b for b, _ in with_p])
            for i in range(9):
                a = pd.Series(bmat[:, i]).rank().values
                b = pd.Series(pvec).rank().values
                bp.append(np.corrcoef(a, b)[0, 1])
            print(
                f"Batter vs opposing pitcher ({len(with_p)} unit-games): "
                f"realized mean Spearman {np.mean(bp):+.4f}"
            )

    out = archive_root / "sim_tail_pits.csv"
    pd.concat([pdf.assign(kind="player"), tdf.assign(kind="team")]).to_csv(out, index=False)
    print(f"\nPITs written -> {out}")


if __name__ == "__main__":
    main()
