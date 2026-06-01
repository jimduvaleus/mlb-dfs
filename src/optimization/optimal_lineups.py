"""Generate optimal DraftKings lineups via iterative ILP (OR-Tools CBC).

Shared by the pipeline and bench scripts so both produce identical results.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from src.optimization.lineup import Lineup

POS_REQUIREMENTS: dict[str, int] = {
    "P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3,
}


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
