"""
Post-contest simulation: grade whole portfolios against the REAL field over
many simulated worlds, rather than against the one world that happened.

Why this exists
---------------
scripts/analyze_hit99_signals.py grades a portfolio on the single realized
outcome of each slate. With a handful of settled slates that is a handful of
draws, which is why nothing there clears the noise bar. A post-contest sim
replaces "did it hit" with "how often would it hit", turning each slate into
n_sims draws and giving the comparison real power. It is also the right
question: a portfolio's quality is its performance across the distribution of
outcomes, not its luck in one of them.

What it does
------------
For an archived slate it rebuilds the same players_df/simulation the external
pipeline uses, parses EVERY entrant's lineup out of the contest-standings zip
(so the opponent field is the real one, not a simulated proxy), scores all of
them plus any portfolios under test across n_sims worlds, and reports per
portfolio:

  win_rate    P(the portfolio's best entry is the contest's top score)
  top1_rate   share of the portfolio's entries landing in a world's top 1%
  mean_pctl   average percentile of an entry against the field that world
  best_pctl   average percentile of the portfolio's BEST entry per world

Portfolios compared: any --entrant handles found in the standings, our own
persisted portfolio (archive/<slate>/portfolio_sweep_draftkings.json), and a
uniformly random draw from the field as a control.

Important: this measures what OUR model thinks of a portfolio. Agreement with
SaberSim's own post-contest numbers is evidence about both models; a
disagreement is a finding about where they differ, not proof either is wrong.

Usage
-----
    python scripts/sim_evaluate_portfolios.py --slate 07262026 --entrant ShaidyAdvice
    python scripts/sim_evaluate_portfolios.py --slate 07262026 --entrant ShaidyAdvice \\
        --n-sims 20000 --chunk 2000

    # --build: construct a portfolio through the production p_win path and
    # grade it out-of-sample (see run_build_mode's docstring for the
    # build/eval split). --n-sims here is the TOTAL split three ways, so
    # 25000 -> ~8333 sims each for stage-A / stage-B / eval:
    python scripts/sim_evaluate_portfolios.py --slate 07262026 --build --n-sims 25000
"""
import argparse
import sys
from pathlib import Path

import logging

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.api.external_pool import ContestGroup  # noqa: E402
from src.api import external_pool as ep  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.optimization.contest import ContestSimulator  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from analyze_rival_portfolio import parse_standings  # noqa: E402


def _derive_opponent(team: str, game: str) -> str:
    m = str(game).split(" ")[0]
    away, _, home = m.partition("@")
    return home if team == away else away


def build_slate(archive_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    """(players_df, quantile_grids, name->player_id). Every player appearing
    in any real field lineup is forced into players_df, so no entrant has to
    be dropped for lack of a simulated distribution."""
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
    if not found["projections_path"]:
        raise FileNotFoundError(f"no SaberSim projections CSV in {archive_dir}")
    proj_ext = ep.parse_player_projections(found["projections_path"])
    name_to_id = dict(zip(slate_df["name"], slate_df["player_id"]))
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, set(slate_df["player_id"]), _derive_opponent,
    )
    return players_df, ep.build_quantile_grids(proj_ext), name_to_id


def indicator_for(lineups: list, pid_index: dict) -> np.ndarray:
    """(n_lineups, n_players) float32. Lineups referencing an unmodelled
    player are returned as all-zero rows and filtered by the caller."""
    m = np.zeros((len(lineups), len(pid_index)), dtype=np.float32)
    for r, lu in enumerate(lineups):
        for pid in lu:
            j = pid_index.get(int(pid))
            if j is None:
                m[r, :] = 0.0
                break
            m[r, j] = 1.0
    return m


def evaluate(archive_dir: Path, entrants: list[str], n_sims: int, chunk: int,
             seed: int, rank_entrants: bool = False) -> pd.DataFrame:
    players_df, grids, name_to_id = build_slate(archive_dir)
    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    np.random.seed(seed)
    sim = engine.simulate(n_sims)
    scores = sim.results_matrix                      # (n_sims, n_players)
    pid_index = {int(p): i for i, p in enumerate(sim.player_ids)}

    entries, _ = parse_standings(archive_dir)
    field_lineups, field_owner = [], []
    for names, handle in zip(entries["names"], entries["handle"]):
        ids = [name_to_id.get(n) for n in names]
        if any(i is None for i in ids):
            continue
        field_lineups.append(ids)
        field_owner.append(handle)
    field_owner = np.array(field_owner)
    F = indicator_for(field_lineups, pid_index)
    keep = F.sum(axis=1) == 10
    F, field_owner = F[keep], field_owner[keep]
    print(f"  field: {len(F):,} of {len(entries):,} entries modelled")

    # Portfolios under test, as row-index sets into F where they are real
    # field entries, or as extra indicator rows where they are ours.
    portfolios: dict[str, np.ndarray] = {}
    for h in entrants:
        idx = np.where(np.char.lower(field_owner.astype(str)) == h.lower())[0]
        if len(idx):
            portfolios[h] = idx
            print(f"  {h}: {len(idx)} entries")
        else:
            print(f"  {h}: not found in this contest")
    rng = np.random.default_rng(seed)
    size = max((len(v) for v in portfolios.values()), default=150)
    portfolios["random field"] = rng.choice(len(F), size, replace=False)

    ours = None
    sweep = archive_dir / "portfolio_sweep_draftkings.json"
    if sweep.exists():
        import json
        sw = json.load(open(sweep))["sweep"][0]
        lus = [[name_to_id.get(p["name"]) for p in lu["players"]] for lu in sw["lineups"]]
        O = indicator_for([l for l in lus if all(x is not None for x in l)], pid_index)
        O = O[O.sum(axis=1) == 10]
        if len(O):
            ours = O
            print(f"  ours (prod risk 1): {len(O)} entries")

    n_field = len(F)
    # Per-sim accumulators. Ranking is done against the full real field, with
    # the portfolio's own entries excluded from its own competition so a
    # 150-entry portfolio isn't penalised for beating itself.
    acc = {k: {"win": 0, "top1": 0.0, "pctl": 0.0, "best_pctl": 0.0}
           for k in list(portfolios) + (["ours (prod)"] if ours is not None else [])}
    done = 0
    for start in range(0, n_sims, chunk):
        s = scores[start:start + chunk]                      # (c, n_players)
        c = len(s)
        FS = s @ F.T                                         # (c, n_field)
        order = np.sort(FS, axis=1)
        for label, idx in portfolios.items():
            P = FS[:, idx]                                   # (c, k)
            mask = np.zeros(n_field, dtype=bool)
            mask[idx] = True
            others = FS[:, ~mask]
            other_max = others.max(axis=1)
            other_sorted = np.sort(others, axis=1)
            cut1 = other_sorted[:, int(others.shape[1] * 0.99)]
            acc[label]["win"] += int((P.max(axis=1) > other_max).sum())
            acc[label]["top1"] += float((P > cut1[:, None]).mean(axis=1).sum())
            r = np.array([np.searchsorted(order[i], P[i], side="right") for i in range(c)])
            acc[label]["pctl"] += float((r / n_field).mean(axis=1).sum())
            acc[label]["best_pctl"] += float((r.max(axis=1) / n_field).sum())
        if ours is not None:
            P = s @ ours.T
            fmax = FS.max(axis=1)
            cut1 = order[:, int(n_field * 0.99)]
            acc["ours (prod)"]["win"] += int((P.max(axis=1) > fmax).sum())
            acc["ours (prod)"]["top1"] += float((P > cut1[:, None]).mean(axis=1).sum())
            r = np.array([np.searchsorted(order[i], P[i], side="right") for i in range(c)])
            acc["ours (prod)"]["pctl"] += float((r / n_field).mean(axis=1).sum())
            acc["ours (prod)"]["best_pctl"] += float((r.max(axis=1) / n_field).sum())
        done += c
        print(f"    simulated {done:,}/{n_sims:,}", end="\r")
    print()

    if rank_entrants:
        # P(win) for every handle in the field: per world the contest winner
        # is one argmax, so this is a bincount over owners. Independent of any
        # portfolio under test, and the direct check of a "tops in the field"
        # claim from another simulator.
        handles = pd.unique(field_owner)
        h_index = {h: i for i, h in enumerate(handles)}
        owner_idx = np.array([h_index[h] for h in field_owner])
        wins = np.zeros(len(handles))
        for start in range(0, n_sims, chunk):
            FS = scores[start:start + chunk] @ F.T
            np.add.at(wins, owner_idx[FS.argmax(axis=1)], 1.0)
        n_ent = pd.Series(field_owner).value_counts()
        board = pd.DataFrame({
            "handle": handles, "entries": [n_ent[h] for h in handles],
            "win_rate": wins / n_sims,
        })
        # `lift` is the number that matters: if every entry in the field were
        # exchangeable, P(a handle wins) would be exactly its share of the
        # entries, so lift = win_rate / entry_share has null 1.0 by
        # construction. Anything above 1 is construction skill, not volume.
        board["entry_share"] = board["entries"] / n_field
        board["lift"] = board["win_rate"] / board["entry_share"]
        # Monte-Carlo SE on win_rate, propagated to lift.
        board["lift_se"] = (np.sqrt(board["win_rate"] * (1 - board["win_rate"]) / n_sims)
                            / board["entry_share"])
        board = board.sort_values("win_rate", ascending=False).reset_index(drop=True)
        board.index += 1
        print(f"\n  Field leaderboard by simulated P(win), {len(handles):,} distinct handles, "
              f"{n_field:,} entries, {n_sims:,} sims")
        print("  (lift = win_rate / entry_share; 1.00 = wins exactly in proportion to volume)")
        print(board.head(15)[["handle", "entries", "win_rate", "lift", "lift_se"]]
              .to_string(float_format=lambda x: f"{x:.4f}"))

        # Entry count dominates raw P(win), so the honest comparison holds it
        # fixed: rank only the max-entry cohort against each other.
        top_n = int(board["entries"].max())
        cohort = board[board["entries"] == top_n].copy().reset_index(drop=True)
        cohort.index += 1
        print(f"\n  Max-entry cohort ({top_n} entries each, {len(cohort)} handles) — "
              f"entry count held constant, so this is construction alone:")
        print(cohort.head(12)[["handle", "win_rate", "lift", "lift_se"]]
              .to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"  cohort lift: mean {cohort['lift'].mean():.3f}  "
              f"median {cohort['lift'].median():.3f}  max {cohort['lift'].max():.3f}")
        singles = board[board["entries"] == 1]
        if len(singles):
            print(f"  single-entry handles ({len(singles)}): mean lift {singles['lift'].mean():.3f}")

        for h in entrants:
            m = board[board["handle"].str.lower() == h.lower()]
            if len(m):
                r_all = m.index[0]
                c = cohort[cohort["handle"].str.lower() == h.lower()]
                r_co = c.index[0] if len(c) else None
                print(f"\n  --> {h}: P(win) {m.iloc[0]['win_rate']:.4f} over "
                      f"{int(m.iloc[0]['entries'])} entries")
                print(f"      lift {m.iloc[0]['lift']:.2f}x +/- {m.iloc[0]['lift_se']:.2f} "
                      f"(wins {m.iloc[0]['lift']:.2f}x as often as its entry share implies)")
                print(f"      rank {r_all} of {len(board)} overall"
                      + (f", {r_co} of {len(cohort)} within the max-entry cohort"
                         if r_co else ""))

    rows = []
    for label, a in acc.items():
        n = len(portfolios[label]) if label in portfolios else len(ours)
        rows.append({"portfolio": label, "entries": n,
                     "win_rate": a["win"] / n_sims,
                     # Wins relative to what this many entries would take if
                     # every entry in the field were exchangeable (null 1.0).
                     "lift": (a["win"] / n_sims) / (n / n_field),
                     "top1_rate": a["top1"] / n_sims,
                     "mean_pctl": a["pctl"] / n_sims,
                     "best_pctl": a["best_pctl"] / n_sims})
    return pd.DataFrame(rows).sort_values("win_rate", ascending=False)


def _score_portfolios(
    eval_sims: np.ndarray, F: np.ndarray, matrices: dict[str, np.ndarray], chunk: int,
) -> tuple[dict[str, dict], int]:
    """Score each (k, n_players) indicator matrix in `matrices` — already in
    the same player-index space as `F` — against the real field `F` over
    `eval_sims` ((S, n_players), a slice of sim.results_matrix disjoint from
    whatever produced any of the matrices). Unlike `evaluate()`'s loop, this
    does not self-exclude a portfolio's own entries from the field it is
    compared against: none of the matrices here are rows of F (the built
    portfolio and the pool references never appear in the real field), so
    there is nothing to exclude — the one caller that DOES need
    self-exclusion (a real entrant's own logged entries) still uses
    `evaluate()`."""
    n_sims_slice = eval_sims.shape[0]
    n_field = len(F)
    acc = {k: {"win": 0, "top1": 0.0, "pctl": 0.0} for k in matrices}
    for start in range(0, n_sims_slice, chunk):
        s = eval_sims[start:start + chunk]
        c = len(s)
        FS = s @ F.T
        order = np.sort(FS, axis=1)
        cut1 = order[:, int(n_field * 0.99)]
        fmax = FS.max(axis=1)
        for label, M in matrices.items():
            P = s @ M.T
            acc[label]["win"] += int((P.max(axis=1) > fmax).sum())
            acc[label]["top1"] += float((P > cut1[:, None]).mean(axis=1).sum())
            r = np.array([np.searchsorted(order[i], P[i], side="right") for i in range(c)])
            acc[label]["pctl"] += float((r / n_field).mean(axis=1).sum())
        print(f"    eval {start + c:,}/{n_sims_slice:,}", end="\r")
    print()
    return acc, n_sims_slice


def _print_structure(portfolio: list, players_df: pd.DataFrame) -> None:
    """Primary-stack team count and within/between-team overlap for a built
    portfolio — the structural sanity check from analyze_rival_portfolio.py.
    ShaidyAdvice runs 19-21 primary teams with within/between overlap
    ~3.9/0.9; collapsing onto a handful of teams (the known failure mode of
    an unconstrained value-only build) would show up here as a low team
    count and/or within overlap far above that band."""
    team_of = dict(zip(players_df["player_id"], players_df["team"]))
    pos_of = dict(zip(players_df["player_id"], players_df["position"]))
    lineups = [lu.player_ids for lu, _ in portfolio]
    if not lineups:
        return
    primary = []
    for lu in lineups:
        hitters = [team_of.get(p) for p in lu if pos_of.get(p) != "P" and team_of.get(p)]
        primary.append(pd.Series(hitters).value_counts().index[0] if hitters else None)
    n = len(lineups)
    all_pids = sorted({p for lu in lineups for p in lu})
    pid_pos = {p: i for i, p in enumerate(all_pids)}
    M = np.zeros((n, len(all_pids)), dtype=np.float32)
    for r, lu in enumerate(lineups):
        for p in lu:
            M[r, pid_pos[p]] = 1.0
    ov = M @ M.T
    g = np.array([p if p is not None else "?" for p in primary])
    iu = np.triu_indices(n, k=1)
    same = g[iu[0]] == g[iu[1]]
    v = ov[iu]
    within = float(v[same].mean()) if same.any() else float("nan")
    between = float(v[~same].mean()) if (~same).any() else float("nan")
    n_teams = len(set(p for p in primary if p is not None))
    print(f"  structure: {n_teams} primary teams, within-team overlap {within:.2f}, "
          f"between-team overlap {between:.2f} (reference: ShaidyAdvice ~20 teams, ~3.9/~0.9)")


def run_build_mode(
    archive_dir: Path, portfolio_size: int, implied_entries: float, sharpness: float,
    admit_n: int, field_size: int, n_sims: int, chunk: int, seed: int, entrants: list[str],
) -> pd.DataFrame:
    """Build a portfolio through the actual production path
    (ep.allocate_contests, ev_type="p_win") and grade it out-of-sample.

    The build must not see anything production wouldn't have at lock:
    `n_sims` is split into three EQUAL, DISJOINT thirds — build-A (the
    p_win stage-A cull draw), build-B (the stage-B select draw, exactly
    mirroring the two-stage winner's-curse guard `allocate_contests` uses
    in production), and eval (scored against the REAL field from the
    standings zip, never touched by the build). The two opponent fields
    used for build-A/build-B are independently generated
    (ContestSimulator.generate_field, ownership-sampled) — never the real
    field, which the build never sees.

    Reference portfolios, all evaluated on the identical eval slice/field:
    `prj_own` top-N (compute_prj_own_ev, matching the production ev_type
    of that name), a uniform random draw from the same pool, our persisted
    production portfolio (portfolio_sweep_draftkings.json, if present),
    and any --entrant handle (via evaluate()'s self-excluding scorer,
    since real entrants' lineups ARE part of the real field)."""
    players_df, grids, name_to_id = build_slate(archive_dir)
    cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    copula = EmpiricalCopula(str(PROJECT_ROOT / cfg["paths"]["copula"]))
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)
    np.random.seed(seed)
    n_third = n_sims // 3
    total = 3 * n_third
    print(f"  simulating {total:,} sims ({n_third:,} build-A / {n_third:,} build-B / "
          f"{n_third:,} eval, disjoint)")
    sim = engine.simulate(total)
    pid_index = {int(p): i for i, p in enumerate(sim.player_ids)}

    found = ep.discover_external_files(str(archive_dir))
    if not found["lineups_paths"]:
        raise FileNotFoundError(f"no lineups_*.csv in {archive_dir}")
    valid_ids = set(players_df["player_id"].astype(int))
    pool = ep.parse_lineup_pool(found["lineups_paths"], valid_ids)
    print(f"  pool: {len(pool.lineups):,} lineups")

    lineup_scores = ep.compute_lineup_scores(pool.lineups, sim)          # (M, total)
    corr = ep.compute_pool_corr(pool.lineups, sim, scores=lineup_scores)
    scores_A = lineup_scores[:, :n_third]
    scores_B = lineup_scores[:, n_third:2 * n_third]
    sims_A = sim.results_matrix[:n_third]
    sims_B = sim.results_matrix[n_third:2 * n_third]
    eval_sims = sim.results_matrix[2 * n_third:3 * n_third]

    own_vec = players_df["ownership"].astype(float).to_numpy()
    cs = ContestSimulator()
    field_n = field_size if field_size > 0 else 10_000
    field_A = cs.generate_field(players_df, own_vec, n_lineups=field_n, rng_seed=seed + 100)
    field_B = cs.generate_field(players_df, own_vec, n_lineups=field_n, rng_seed=seed + 101)
    print(f"  simulated opponent fields: {len(field_A):,} (A) / {len(field_B):,} (B) lineups")
    field_scores_A = cs.score_field(field_A, sims_A, pid_index)
    field_scores_B = cs.score_field(field_B, sims_B, pid_index)

    exponent = max(1.0, sharpness * implied_entries)
    p_win_cull = ep.compute_p_win(scores_A, field_scores_A, {"c0": exponent})
    p_win_select = ep.compute_p_win(scores_B, field_scores_B, {"c0": exponent})

    group = ContestGroup(
        contest_id="c0", contest_name="p_win build test",
        entry_fee_cents=400, prize_pool_cents=int(implied_entries * 400),
        single_entry_tag=False, roi_key="",
        entries=[(Path("x"), None)] * portfolio_size,
    )
    alloc = ep.allocate_contests(
        pool, corr, [group], risk=3.0, evw_base=0.10, evw_max=0.40,
        ev_type="p_win", p_win_cull=p_win_cull, p_win_select=p_win_select,
        p_win_admit_n=admit_n,
    )
    print(f"  built {len(alloc.portfolio)} entries ({len(alloc.unfilled)} unfilled)")
    _print_structure(alloc.portfolio, players_df)

    # --- reference portfolios, same pool, same pid_index space ----------
    proj_scores = ep.compute_pool_proj_scores(pool.lineups, players_df)
    own_scores = ep.compute_pool_ownership(pool.lineups, players_df)
    prj_own_ev = ep.compute_prj_own_ev(proj_scores, own_scores, implied_entries)
    topn_idx = np.argsort(-prj_own_ev)[:portfolio_size]

    rng = np.random.default_rng(seed)
    rand_idx = rng.choice(len(pool.lineups), portfolio_size, replace=False)

    matrices = {
        "p_win build": indicator_for([lu.player_ids for lu, _ in alloc.portfolio], pid_index),
        "prj_own top-N": indicator_for([pool.lineups[i].player_ids for i in topn_idx], pid_index),
        "pool random-N": indicator_for([pool.lineups[i].player_ids for i in rand_idx], pid_index),
    }
    sweep = archive_dir / "portfolio_sweep_draftkings.json"
    if sweep.exists():
        import json
        sw = json.load(open(sweep))["sweep"][0]
        lus = [[name_to_id.get(p["name"]) for p in lu["players"]] for lu in sw["lineups"]]
        O = indicator_for([lu for lu in lus if all(x is not None for x in lu)], pid_index)
        O = O[O.sum(axis=1) == 10]
        if len(O):
            matrices["ours (prod)"] = O
            print(f"  ours (prod risk 1): {len(O)} entries")

    # --- real field for the eval ------------------------------------------
    entries, _ = parse_standings(archive_dir)
    field_lineups, field_owner = [], []
    for names, handle in zip(entries["names"], entries["handle"]):
        ids = [name_to_id.get(n) for n in names]
        if not any(i is None for i in ids):
            field_lineups.append(ids)
            field_owner.append(handle)
    field_owner = np.array(field_owner)
    F = indicator_for(field_lineups, pid_index)
    keep = F.sum(axis=1) == 10
    F, field_owner = F[keep], field_owner[keep]
    n_field = len(F)
    print(f"  real field: {n_field:,} of {len(entries):,} entries modelled")

    acc, n_eval = _score_portfolios(eval_sims, F, matrices, chunk)
    rows = []
    for label, a in acc.items():
        n = matrices[label].shape[0]
        rows.append({
            "portfolio": label, "entries": n,
            "win_rate": a["win"] / n_eval,
            "lift": (a["win"] / n_eval) / (n / n_field),
            "top1_rate": a["top1"] / n_eval,
            "mean_pctl": a["pctl"] / n_eval,
        })

    # Real entrants are actual rows of F, so route them through evaluate()'s
    # self-excluding scorer over the same real field (a fresh, smaller sim
    # draw — the eval slice above was already consumed building `matrices`'
    # comparisons and reusing it here would be fine too, but evaluate() owns
    # its own sim draw and mixing accumulators would complicate the n_eval
    # denominator, so this is a second, independent read for entrants only).
    if entrants:
        ent_df = evaluate(archive_dir, entrants, n_sims=min(n_third, 5000), chunk=chunk,
                          seed=seed + 200)
        ent_df = ent_df[ent_df["portfolio"].isin(entrants)]
        rows.extend(ent_df.to_dict("records"))

    return pd.DataFrame(rows).sort_values("lift", ascending=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Post-contest simulation of whole portfolios against the real field.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--slate", required=True, help="Archive dir name, e.g. 07262026")
    p.add_argument("--entrant", action="append", default=[],
                   help="DK handle to evaluate; repeatable.")
    p.add_argument("--n-sims", type=int, default=10000)
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank-entrants", action="store_true",
                   help="Also print the whole field ranked by simulated P(win), to test "
                        "a 'tops in the field' claim from another simulator.")
    p.add_argument("--build", action="store_true",
                   help="Instead of grading existing portfolios, BUILD one through the "
                        "production p_win path (ep.allocate_contests) and grade it "
                        "out-of-sample: --n-sims is split into three equal disjoint thirds "
                        "(stage-A cull / stage-B select / eval against the real field). "
                        "Slower when combined with --entrant (a second, independent sim "
                        "draw scores real entrants via evaluate()'s self-excluding path).")
    p.add_argument("--portfolio-size", type=int, default=150,
                   help="--build only: entries in the built portfolio (default: 150).")
    p.add_argument("--implied-entries", type=float, default=10_000.0,
                   help="--build only: assumed contest size for the p_win exponent "
                        "(sharpness * implied_entries) and the prj_own reference "
                        "(default: 10000).")
    p.add_argument("--sharpness", type=float, default=1.0,
                   help="--build only: p_win exponent multiplier (default: 1.0 = literal "
                        "P(win); lower softens toward P(top X%%)).")
    p.add_argument("--admit-n", type=int, default=2000,
                   help="--build only: p_win stage-A cull size (default: 2000; 0 disables).")
    p.add_argument("--field-size", type=int, default=0,
                   help="--build only: simulated opponent field size for the p_win stages "
                        "(default: 0 -> 10000).")
    args = p.parse_args()
    logging.getLogger("src.api.external_pool").setLevel(logging.ERROR)

    d = PROJECT_ROOT / "archive" / args.slate
    print(f"=== {args.slate} ===")
    if args.build:
        out = run_build_mode(
            d, args.portfolio_size, args.implied_entries, args.sharpness, args.admit_n,
            args.field_size, args.n_sims, args.chunk, args.seed, args.entrant,
        )
    else:
        out = evaluate(d, args.entrant, args.n_sims, args.chunk, args.seed, args.rank_entrants)
    print()
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
