"""Generate optimal DraftKings lineups via iterative ILP (OR-Tools CBC).

Shared by the pipeline and bench scripts so both produce identical results.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.optimization.lineup import Lineup

POS_REQUIREMENTS: dict[str, int] = {
    "P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3,
}


def stratified_sim_sample(
    sim_matrix: np.ndarray, n_sample: int, rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample n_sample sim indices stratified across slate-total deciles, so
    quiet and explosive run environments are both represented.

    Returns [(sim_index, decile 1-10), ...], at most n_sample entries.
    """
    n_sims = sim_matrix.shape[0]
    n_sample = min(int(n_sample), n_sims)
    order = np.argsort(sim_matrix.sum(axis=1))
    sampled: list[tuple[int, int]] = []
    per_decile = max(1, int(np.ceil(n_sample / 10)))
    for dec_idx, dec in enumerate(np.array_split(order, 10)):
        take = min(per_decile, len(dec))
        sampled.extend(
            (int(s), dec_idx + 1) for s in rng.choice(dec, size=take, replace=False)
        )
    return sampled[:n_sample]


def generate_sim_optimal_lineups(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    sim_player_ids: list[int],
    sim_indices: list[int],
    min_stack: int = 4,
    salary_floor: Optional[float] = None,
    seen: Optional[set] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> list[Lineup]:
    """Per-sim optimal lineups: for each sim index, solve the roster ILP with
    that sim's *realized* player scores as the objective — the lineup that
    wins that particular simulated world.

    Unlike the projected-mean ILP seeding (one deterministic answer per team
    stack), these seeds are ceiling lineups by construction: each is a
    99.9th-percentile outcome in at least one world the simulator produced,
    capturing the correlation/variance structure a mean objective ignores.
    Sample sim_indices with stratified_sim_sample so the seeds span run
    environments.

    df is the candidate player pool (same contract as generate_optimal_lineups;
    the "mean" column is ignored and replaced per sim). Players missing from
    sim_player_ids score 0 and are never preferred. Duplicate solutions
    (across sims, and against `seen`) are dropped.

    Runs solves in a thread pool — CBC releases the GIL. Returns unique
    lineups in sim_indices order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    col_map = {int(pid): i for i, pid in enumerate(sim_player_ids)}
    pid_list = df["player_id"].astype(int).tolist()
    col_idx = np.array([col_map.get(p, -1) for p in pid_list], dtype=np.int64)

    def _solve(sim_idx: int) -> Optional[Lineup]:
        if stop_check is not None and stop_check():
            return None
        row = sim_matrix[sim_idx].astype(np.float64)
        scores = np.where(col_idx >= 0, row[np.clip(col_idx, 0, None)], 0.0)
        d = df.copy()
        d["mean"] = scores
        lineups = generate_optimal_lineups(
            d, n=1, min_uniques=1, min_stack=min_stack, salary_floor=salary_floor,
        )
        return lineups[0] if lineups else None

    results: list[Optional[Lineup]] = [None] * len(sim_indices)
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_solve, s): i for i, s in enumerate(sim_indices)}
        n_done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            n_done += 1
            if progress_cb is not None:
                progress_cb(n_done)

    seen_keys: set = set(seen or ())
    unique: list[Lineup] = []
    for lu in results:
        if lu is None:
            continue
        key = frozenset(int(p) for p in lu.player_ids)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(lu)
    return unique


def generate_optimal_lineups(
    df: pd.DataFrame,
    n: int = 100,
    min_uniques: int = 4,
    min_stack: int = 4,
    stack_team: Optional[str] = None,
    salary_floor: Optional[float] = None,
    seen: Optional[set] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    prior_lineups: Optional[list] = None,
    min_uniques_vs_prior: Optional[int] = None,
) -> list[Lineup]:
    """Return up to N optimal lineups by projected mean score.

    Uses iterative ILP (OR-Tools CBC) with no-good cuts to enumerate
    distinct lineups. Each new lineup must differ from every previously
    found lineup by at least *min_uniques* players.

    Parameters
    ----------
    df:
        Player pool. Required columns: player_id, mean, position,
        eligible_positions (list[str]), salary, team, opponent, game.
        Pass cand_players_df from the pipeline so exclusions are respected.
    n:
        Number of lineups to return.
    min_uniques:
        Minimum players that must differ from any single prior lineup.
    min_stack:
        Minimum batters from one team required (stack constraint).
    stack_team:
        If provided, forces this team to be the stack team (z[t]=1).
    salary_floor:
        Minimum total salary (optional).
    seen:
        set of frozenset[int] of player_ids already generated in other
        batches. Duplicate lineups are skipped (but their no-good cut is
        still added), so the solver won't revisit them.
    progress_cb:
        Called with the count of lineups found so far after each lineup.
    prior_lineups:
        Lineups from a previously solved batch. No-good cuts are seeded
        for each at the start of the solve, enforcing min_uniques_vs_prior
        uniqueness against each of them before any new lineup is generated.
    min_uniques_vs_prior:
        Min players that must differ from each lineup in prior_lineups.
        Defaults to min_uniques when not specified.
    """
    try:
        from ortools.linear_solver import pywraplp
    except ImportError as exc:
        raise ImportError(
            "ortools is required for optimal lineup generation. "
            "Install with: pip install ortools"
        ) from exc

    # --- Build metadata ---
    player_ids: list[int] = df["player_id"].astype(int).tolist()
    mean_map: dict[int, float] = {int(r.player_id): float(r.mean) for r in df.itertuples(index=False)}

    meta: dict[int, dict] = {}
    for r in df.itertuples(index=False):
        pid = int(r.player_id)
        ep = r.eligible_positions
        meta[pid] = {
            "position": r.position,
            "eligible_positions": list(ep) if ep is not None else [r.position],
            "salary": float(r.salary),
            "team": r.team,
            "opponent": r.opponent,
            "game": r.game,
        }

    # --- (player, pos) variable index structures ---
    xp_list: list[tuple[int, str]] = []
    player_to_js: dict[int, list[int]] = {pid: [] for pid in player_ids}
    pos_to_js: dict[str, list[int]] = {}

    for pid in player_ids:
        for pos in meta[pid]["eligible_positions"]:
            j = len(xp_list)
            xp_list.append((pid, pos))
            player_to_js[pid].append(j)
            pos_to_js.setdefault(pos, []).append(j)

    n_xp = len(xp_list)

    pitcher_pids = [pid for pid in player_ids if meta[pid]["position"] == "P"]
    batter_pids  = [pid for pid in player_ids if meta[pid]["position"] != "P"]
    batter_teams = sorted({meta[pid]["team"] for pid in batter_pids})
    T = len(batter_teams)
    team_idx = {tm: t for t, tm in enumerate(batter_teams)}

    team_batter_js: dict[str, list[int]] = {tm: [] for tm in batter_teams}
    game_js: dict[str, list[int]] = {}
    for j, (pid, _pos) in enumerate(xp_list):
        if meta[pid]["position"] != "P":
            team_batter_js[meta[pid]["team"]].append(j)
        g = meta[pid]["game"]
        if g:
            game_js.setdefault(g, []).append(j)

    pitcher_team_js: dict[str, list[int]] = {}
    for pp in pitcher_pids:
        for j in player_to_js[pp]:
            pitcher_team_js.setdefault(meta[pp]["team"], []).append(j)

    opp_of = {pid: meta[pid]["opponent"] for pid in pitcher_pids}

    # --- Create solver ---
    solver = pywraplp.Solver.CreateSolver("CBC")
    solver.SuppressOutput()

    xp = [solver.BoolVar(f"xp{j}") for j in range(n_xp)]
    z  = [solver.BoolVar(f"z{t}") for t in range(T)]

    # Objective: maximise sum(mean * xp)
    obj = solver.Objective()
    obj.SetMaximization()
    for j, (pid, _) in enumerate(xp_list):
        obj.SetCoefficient(xp[j], mean_map[pid])

    # C1: each multi-position player selected at most once
    for pid, js in player_to_js.items():
        if len(js) > 1:
            c = solver.Constraint(0, 1)
            for j in js:
                c.SetCoefficient(xp[j], 1)

    # C2: exact position slot counts
    for pos, count in POS_REQUIREMENTS.items():
        c = solver.Constraint(count, count)
        for j in pos_to_js.get(pos, []):
            c.SetCoefficient(xp[j], 1)

    # C3: salary cap <= 50,000
    c = solver.Constraint(0, 50_000)
    for j, (pid, _) in enumerate(xp_list):
        c.SetCoefficient(xp[j], meta[pid]["salary"])

    # C3b: salary floor (optional)
    if salary_floor is not None and salary_floor > 0:
        c = solver.Constraint(float(salary_floor), solver.infinity())
        for j, (pid, _) in enumerate(xp_list):
            c.SetCoefficient(xp[j], meta[pid]["salary"])

    # C4: <= 5 batters per team
    for tm in batter_teams:
        c = solver.Constraint(0, 5)
        for j in team_batter_js[tm]:
            c.SetCoefficient(xp[j], 1)

    # C5: pitcher-batter conflict (aggregate big-M per pitcher)
    #     sum(opp_batter_xp) + n_opp * sum(pitcher_xp) <= n_opp
    for pp in pitcher_pids:
        opp = opp_of[pp]
        opp_batter_js = [j for bp in batter_pids if meta[bp]["team"] == opp
                         for j in player_to_js[bp]]
        n_opp = len([bp for bp in batter_pids if meta[bp]["team"] == opp])
        if opp_batter_js and player_to_js[pp]:
            c = solver.Constraint(-solver.infinity(), float(n_opp))
            for j in opp_batter_js:
                c.SetCoefficient(xp[j], 1)
            for j in player_to_js[pp]:
                c.SetCoefficient(xp[j], float(n_opp))

    # C6: <= 1 pitcher per team
    for tm, js in pitcher_team_js.items():
        if len(js) > 1:
            c = solver.Constraint(0, 1)
            for j in js:
                c.SetCoefficient(xp[j], 1)

    # C7: sum(z[t]) >= 1 (at least one stack team)
    c = solver.Constraint(1, T)
    for t in range(T):
        c.SetCoefficient(z[t], 1)

    # C8: sum(batter_xp[team_t]) - min_stack * z[t] >= 0 (stack linkage)
    for t, tm in enumerate(batter_teams):
        c = solver.Constraint(0, solver.infinity())
        for j in team_batter_js[tm]:
            c.SetCoefficient(xp[j], 1)
        c.SetCoefficient(z[t], -float(min_stack))

    # C9: <= 9 players per game (ensures >= 2 games)
    for _g, js in game_js.items():
        c = solver.Constraint(0, 9)
        for j in js:
            c.SetCoefficient(xp[j], 1)

    # C9b: force a specific team to be the stack team (optional)
    if stack_team is not None and stack_team in team_idx:
        t_forced = team_idx[stack_team]
        c = solver.Constraint(1, 1)
        c.SetCoefficient(z[t_forced], 1)

    # --- Seed no-good cuts from prior batches ---
    if prior_lineups:
        _mu_prior = min_uniques_vs_prior if min_uniques_vs_prior is not None else min_uniques
        for _lu in prior_lineups:
            _prior_js = sorted({j for pid in _lu.player_ids for j in player_to_js.get(pid, [])})
            if _prior_js:
                _c = solver.Constraint(0, float(10 - _mu_prior))
                for j in _prior_js:
                    _c.SetCoefficient(xp[j], 1)

    # --- Iterative solve with no-good cuts ---
    lineups: list[Lineup] = []
    # Budget extra attempts to account for cross-batch duplicates skipped via `seen`
    max_attempts = n * 3 if seen is not None else n

    for _ in range(max_attempts):
        if len(lineups) >= n:
            break
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            break

        pids = [xp_list[j][0] for j in range(n_xp) if xp[j].solution_value() > 0.5]

        # No-good cut always runs so this lineup is never revisited
        cut_js = sorted({j for pid in pids for j in player_to_js[pid]})
        c = solver.Constraint(0, float(10 - min_uniques))
        for j in cut_js:
            c.SetCoefficient(xp[j], 1)

        # Skip if already generated in a prior batch
        key = frozenset(pids)
        if seen is not None and key in seen:
            continue

        lineups.append(Lineup(player_ids=pids))
        if seen is not None:
            seen.add(key)
        if progress_cb is not None:
            progress_cb(len(lineups))

    return lineups
