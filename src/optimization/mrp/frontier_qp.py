"""Haugh & Singal line 2, solved whole: the mean-variance frontier generator.

    w_lambda = argmax_{w in W}  w'mu + lambda (w'Sigma w - 2 w'sigma_dG)

`src/optimization/optimal_lineups.py` solves the lambda=0 case with CBC. CBC
cannot express a quadratic objective at all, which is why `sigma_frontier.py`
dropped `w'Sigma w` and kept only the cross-term. This module puts it back.

WHAT THE QUADRATIC ACTUALLY IS. From the paper's (14), with Y = w'delta - G:

    w'Sigma w - 2 w'sigma_dG  =  Var(w'delta - G)  -  Var(G)
                                 ^ margin variance    ^ constant in w

So line 2 is mean-variance in MARGIN space -- expected margin over the payout
cutoff plus lambda times the variance of that margin. Keeping only the
cross-term (leverage) or only the quadratic (ceiling) is half an object each.

CP-SAT, NOT CBC. Maximising w'Sigma w over binaries is a BQP; it is linearised
with one McCormick product variable per off-diagonal pair. Two things keep that
small enough to solve:

  * ONLY WITHIN-UNIT PAIRS. Cross-copula-unit covariance is ~0 by construction
    (see lineup_variance.py), so a 12-game slate needs ~1,000 product
    variables rather than the ~80,000 of a dense linearisation.
  * THE DIAGONAL IS FREE. y^2 = y for a binary, so Sigma_pp folds into the
    linear coefficient on y_p and needs no product variable.

PLAYER VARIABLES vs (PLAYER, POSITION) VARIABLES. `generate_optimal_lineups`
indexes on (player, position) pairs so a multi-position player can fill any of
his slots. The quadratic is over PLAYERS, so this adds y_p == sum of that
player's xp columns -- well defined because C1 already caps that sum at 1 --
and builds every product on y. No-good cuts are written on y too, which is the
same constraint as the xp form and easier to read.

LAMBDA IS NOT TUNED HERE. Algorithm 4's line 4 and Algorithm 6's line 6 both
exist to pick a single lambda, because the paper can only play what its
frontier hands it. dR evaluates the true objective over the whole pool, so we
hand it the ENTIRE frontier and let it choose. Lambda is a generation-diversity
knob, not a hyperparameter -- nothing in this file is fitted.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.optimization.lineup import Lineup, normalize_eligible_positions
from src.optimization.candidate_generator import CandidateGenerator
from src.optimization.mrp.lineup_variance import FrontierScorer, lineup_variance
from src.optimization.optimal_lineups import POS_REQUIREMENTS

ROSTER_SIZE = 10

_OBJ_SCALE = 10_000
"""CP-SAT objective coefficients must be integers. FPTS and FPTS^2 quantities
carry ~4 meaningful decimals at this scale, well inside int64 even at the
largest lambda the calibrator produces."""


def restrict_to_playable(
    df: pd.DataFrame,
    pool_pids: Optional[set] = None,
) -> tuple[pd.DataFrame, dict]:
    """Restrict the ILP's player universe to actual starters.

    A slate's players_df carries every listed player. For hitters that is
    already the starting nine -- `build_external_players_df` keeps only batters
    with a batting order, so `slot` runs 1-9 and there are exactly nine per
    team. For PITCHERS it is the entire staff: 482 rows across 18 teams on the
    08/18 slate, ~7 per team once low-projection rows are filtered.

    ONE STARTER PER TEAM, BY HIGHEST PROJECTED MEAN. This mirrors
    `parse_sabersim_projections` (external_pool.py:538-542,
    `groupby("team")["mean"].idxmax()`), which is the repo's established rule
    and exists because a pitcher row's Status column is not a reliable
    confirmed-starter signal. The data backs it: every team shows a clean gap
    between its starter and the rest of the staff (CHC 17.27 then 2.34, BOS
    15.36 then 2.15), and 94% of the external pool's pitcher slots are already
    filled by a team-top pitcher.

    WHY THIS IS CORRECTNESS, NOT SPEED. A threshold on projection or variance
    admits relievers -- they sit at 1-2 FPTS with non-trivial variance and low
    salary, which is exactly the shape a variance-MAXIMISING objective reaches
    for once lambda is large enough. A frontier lineup that rosters a reliever
    is unplayable, and no downstream stage would catch it. The speed win (pair
    variables fall ~82%, 11,280 -> ~2,100) is a side effect.

    `pool_pids` widens the HITTER side only -- a pool hitter with no batting
    order is still a real SaberSim pick. It deliberately does NOT widen the
    pitcher side: the pool's 5.9% non-top pitcher usage is the thing being
    excluded, not a reason to re-admit it.

    Returns `(restricted_df, diag)`.
    """
    is_p = (df["position"] == "P").to_numpy()
    in_pool = (df["player_id"].isin(pool_pids).to_numpy() if pool_pids
               else np.zeros(len(df), dtype=bool))

    pitchers = df[is_p]
    starters: set = set()
    if not pitchers.empty:
        valid = pitchers[pitchers["mean"].notna()]
        if not valid.empty:
            starters = {int(p) for p in
                        df.loc[valid.groupby("team")["mean"].idxmax(), "player_id"]}

    slot = df["slot"].to_numpy() if "slot" in df.columns else np.full(len(df), 1)
    keep_p = is_p & df["player_id"].isin(starters).to_numpy()
    keep_b = ~is_p & (((slot >= 1) & (slot <= 9)) | in_pool)
    keep = keep_p | keep_b

    diag = {
        "n_players_before": int(len(df)),
        "n_players_kept": int(keep.sum()),
        "n_pitchers_before": int(is_p.sum()),
        "n_pitchers_kept": int(keep_p.sum()),
        "n_batters_kept": int(keep_b.sum()),
        "n_teams": int(df["team"].nunique()),
    }
    return df[keep].copy(), diag


def _build_metadata(df: pd.DataFrame) -> dict:
    """The index structures `generate_optimal_lineups` builds, reused verbatim.

    Kept byte-for-byte compatible with the CBC path so the two solvers are
    genuinely solving the same feasible set -- `tests/test_mrp_frontier_qp.py`
    asserts they return the same lineup at lambda=0, which only means anything
    if the constraint set is identical.
    """
    if "eligible_positions" not in df.columns:
        raise ValueError(
            "frontier_qp needs `eligible_positions`; pass the same players_df "
            "generate_optimal_lineups takes (dk_slate provides the column)"
        )
    player_ids: list[int] = df["player_id"].astype(int).tolist()
    mean_map = {int(r.player_id): float(r.mean) for r in df.itertuples(index=False)}

    meta: dict[int, dict] = {}
    for r in df.itertuples(index=False):
        pid = int(r.player_id)
        ep = r.eligible_positions
        meta[pid] = {
            "position": r.position,
            "eligible_positions": normalize_eligible_positions(ep, r.position),
            "salary": float(r.salary),
            "team": r.team,
            "opponent": r.opponent,
            "game": r.game,
        }

    xp_list: list[tuple[int, str]] = []
    player_to_js: dict[int, list[int]] = {pid: [] for pid in player_ids}
    pos_to_js: dict[str, list[int]] = {}
    for pid in player_ids:
        for pos in meta[pid]["eligible_positions"]:
            j = len(xp_list)
            xp_list.append((pid, pos))
            player_to_js[pid].append(j)
            pos_to_js.setdefault(pos, []).append(j)

    pitcher_pids = [pid for pid in player_ids if meta[pid]["position"] == "P"]
    batter_pids = [pid for pid in player_ids if meta[pid]["position"] != "P"]
    batter_teams = sorted({meta[pid]["team"] for pid in batter_pids})

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

    return {
        "player_ids": player_ids, "mean_map": mean_map, "meta": meta,
        "xp_list": xp_list, "player_to_js": player_to_js, "pos_to_js": pos_to_js,
        "pitcher_pids": pitcher_pids, "batter_pids": batter_pids,
        "batter_teams": batter_teams, "team_batter_js": team_batter_js,
        "game_js": game_js, "pitcher_team_js": pitcher_team_js,
    }


def _build_model(
    md: dict,
    lam: float,
    var_by_pid: dict,
    cov_by_pair: dict,
    sigma_dG: dict,
    min_stack: int,
    salary_floor,
    min_secondary,
    stack_team,
    objective_floor: Optional[int] = None,
):
    """CP-SAT model for one lambda. Returns `(model, xp, y, xp_list)`.

    `objective_floor` (scaled by `_OBJ_SCALE`) switches the model from
    "maximise" to "any lineup at least this good", which is what makes bulk
    near-optimal enumeration possible.

    Constraints C1-C9b are a direct port of `generate_optimal_lineups`; the
    objective and the y/z machinery are what is new.
    """
    from ortools.sat.python import cp_model

    meta = md["meta"]
    xp_list = md["xp_list"]
    player_to_js = md["player_to_js"]
    batter_teams = md["batter_teams"]
    T = len(batter_teams)
    team_idx = {tm: t for t, tm in enumerate(batter_teams)}

    model = cp_model.CpModel()
    xp = [model.NewBoolVar(f"xp{j}") for j in range(len(xp_list))]
    z_team = [model.NewBoolVar(f"z{t}") for t in range(T)]

    # C1: each multi-position player selected at most once.
    for pid, js in player_to_js.items():
        if len(js) > 1:
            model.Add(sum(xp[j] for j in js) <= 1)

    # C2: exact position slot counts.
    for pos, count in POS_REQUIREMENTS.items():
        js = md["pos_to_js"].get(pos, [])
        model.Add(sum(xp[j] for j in js) == count)

    # C3 / C3b: salary cap and optional floor.
    sal = [int(round(meta[pid]["salary"])) for pid, _ in xp_list]
    model.Add(sum(sal[j] * xp[j] for j in range(len(xp_list))) <= 50_000)
    if salary_floor is not None and salary_floor > 0:
        model.Add(sum(sal[j] * xp[j] for j in range(len(xp_list))) >= int(round(salary_floor)))

    # C4: <= 5 batters per team.
    for tm in batter_teams:
        model.Add(sum(xp[j] for j in md["team_batter_js"][tm]) <= 5)

    # C5: pitcher-batter conflict (aggregate big-M per pitcher).
    for pp in md["pitcher_pids"]:
        opp = meta[pp]["opponent"]
        opp_batter_js = [j for bp in md["batter_pids"] if meta[bp]["team"] == opp
                         for j in player_to_js[bp]]
        n_opp = len([bp for bp in md["batter_pids"] if meta[bp]["team"] == opp])
        if opp_batter_js and player_to_js[pp]:
            model.Add(
                sum(xp[j] for j in opp_batter_js)
                + n_opp * sum(xp[j] for j in player_to_js[pp]) <= n_opp
            )

    # C6: <= 1 pitcher per team.
    for _tm, js in md["pitcher_team_js"].items():
        if len(js) > 1:
            model.Add(sum(xp[j] for j in js) <= 1)

    # C7 / C8: at least one team carries a >= min_stack batter stack.
    model.Add(sum(z_team) >= 1)
    for t, tm in enumerate(batter_teams):
        model.Add(sum(xp[j] for j in md["team_batter_js"][tm]) >= min_stack * z_team[t])

    # C8b: secondary-stack shape -- two teams must each carry >= min_secondary.
    if min_secondary is not None and min_secondary >= 1:
        w_sec = [model.NewBoolVar(f"w{t}") for t in range(T)]
        for t, tm in enumerate(batter_teams):
            model.Add(sum(xp[j] for j in md["team_batter_js"][tm]) >= min_secondary * w_sec[t])
        model.Add(sum(w_sec) >= 2)

    # C9: <= 9 players per game (forces >= 2 games).
    for _g, js in md["game_js"].items():
        model.Add(sum(xp[j] for j in js) <= 9)

    # C9b: force a specific stack team.
    if stack_team is not None and stack_team in team_idx:
        model.Add(z_team[team_idx[stack_team]] == 1)

    # --- Player-level indicators, for the quadratic and the cuts -----------
    y = {pid: model.NewBoolVar(f"y{pid}") for pid in md["player_ids"]}
    for pid, js in player_to_js.items():
        model.Add(y[pid] == sum(xp[j] for j in js))

    # --- Objective ---------------------------------------------------------
    # Linear part: mu_p + lambda*Sigma_pp - 2*lambda*sigma_dG_p, the diagonal
    # folded in because y^2 = y for a binary.
    terms = []
    for pid in md["player_ids"]:
        coef = (md["mean_map"][pid]
                + lam * float(var_by_pid.get(pid, 0.0))
                - 2.0 * lam * float(sigma_dG.get(pid, 0.0)))
        c = int(round(coef * _OBJ_SCALE))
        if c:
            terms.append(c * y[pid])

    # Quadratic part: 2*lambda*Sigma_ab per within-unit pair, via McCormick.
    # All three inequalities are added rather than only the binding one -- the
    # coefficient's sign varies (same-team batters positive, batter-vs-opposing
    # -pitcher negative) and at ~1,000 pairs the extra rows cost nothing, while
    # a true product keeps the reported objective interpretable.
    if lam > 0:
        for (pa, pb), cov in cov_by_pair.items():
            if pa not in y or pb not in y:
                continue
            c = int(round(2.0 * lam * float(cov) * _OBJ_SCALE))
            if not c:
                continue
            zab = model.NewBoolVar(f"z_{pa}_{pb}")
            model.Add(zab <= y[pa])
            model.Add(zab <= y[pb])
            model.Add(zab >= y[pa] + y[pb] - 1)
            terms.append(c * zab)

    if objective_floor is None:
        model.Maximize(sum(terms))
    else:
        # NEAR-OPTIMAL ENUMERATION MODE. With no objective the solve is pure
        # SAT search -- no bound to prove, no optimality gap to close -- and
        # CP-SAT can stream out solutions from a single tree instead of paying
        # a fresh proof per lineup. The floor is what keeps them near-optimal.
        model.Add(sum(terms) >= int(objective_floor))
    return model, xp, y, xp_list


def solve_lambda(
    md: dict,
    lam: float,
    var_by_pid: dict,
    cov_by_pair: dict,
    sigma_dG: dict,
    n: int = 1,
    min_uniques: int = 3,
    min_stack: int = 4,
    salary_floor=None,
    min_secondary=None,
    stack_team=None,
    seen: Optional[set] = None,
    timeout_s: float = 10.0,
    seed: int = 42,
    workers: int = 8,
) -> list[Lineup]:
    """Up to `n` distinct lineups at one lambda, via iterative no-good cuts.

    FEASIBLE is accepted alongside OPTIMAL: CP-SAT re-solves from scratch after
    each cut (there is no incremental path as there is in CBC), so a proof of
    optimality on every one of `n` solves is not worth its wall-clock. A
    near-optimal frontier lineup is still a frontier lineup, and dR re-ranks
    everything downstream anyway.
    """
    from ortools.sat.python import cp_model

    model, xp, y, xp_list = _build_model(
        md, lam, var_by_pid, cov_by_pair, sigma_dG,
        min_stack, salary_floor, min_secondary, stack_team,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(timeout_s)
    # Fixed seed AND fixed worker count: CP-SAT's parallel portfolio is
    # deterministic only when both are pinned, and a portfolio that changes
    # between identical runs is not debuggable.
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = int(workers)

    out: list[Lineup] = []
    max_attempts = n * 3 if seen is not None else n
    for _ in range(max_attempts):
        if len(out) >= n:
            break
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        pids = sorted({xp_list[j][0] for j in range(len(xp_list))
                       if solver.Value(xp[j]) > 0})
        if len(pids) != ROSTER_SIZE:
            break
        # No-good cut always runs, so this lineup is never revisited even when
        # it is skipped as a cross-lambda duplicate below.
        model.Add(sum(y[pid] for pid in pids) <= ROSTER_SIZE - min_uniques)

        key = frozenset(pids)
        if seen is not None and key in seen:
            continue
        if seen is not None:
            seen.add(key)
        out.append(Lineup(player_ids=list(pids)))
    return out


def lambda_search_grid(
    scorer: "FrontierScorer",
    rows: np.ndarray,
    sigma_dG: dict,
    n_lambdas: int = 12,
    span_lo: float = 0.125,
    span_hi: float = 4.0,
) -> list[float]:
    """Candidate lambdas for line 4 to choose from.

    The seed is the only scale-free anchor available: `w'mu / w'Sigma w` on the
    lambda=0 winner is the lambda at which the two objective terms are exactly
    equal. The grid spans a fixed multiplicative window around it, wide enough
    to bracket the optimum from either side.

    This is a SEARCH RANGE, not a calibration. The earlier build picked the
    grid's endpoint with an invented rule -- raise lambda until the top lineup
    moves N players -- which has no connection to the payout structure, the
    field size, or anything else in the model, and put half the grid in a
    region where the objective had gone projection-blind (measured: rho with
    projection 0.377 at the grid's top, negative just beyond). Line 4 picks the
    operating point now; this only has to contain it.
    """
    base = scorer.score(rows, 0.0, sigma_dG)
    one = np.array([rows[int(np.argmax(base))]])
    mean0 = float(scorer.score(one, 0.0, None)[0])
    var0 = float(scorer.score(one, 1.0, None)[0] - mean0)
    if not np.isfinite(var0) or var0 <= 0 or mean0 <= 0:
        return [0.0]
    seed = mean0 / var0
    # lambda=0 -- pure projection, no variance seeking -- is a legitimate
    # operating point and MUST be choosable: a contest whose paying bar is
    # reachable on projection alone should be allowed to say so. A geomspace
    # can never reach 0, so it is prepended explicitly. Line 4 decides.
    return [0.0] + list(np.geomspace(seed * span_lo, seed * span_hi,
                                     max(int(n_lambdas) - 1, 2)))


def line4_lambda_star(
    scorer: "FrontierScorer",
    rows: np.ndarray,
    sigma_dG: dict,
    grid,
    contest_specs: list,
    sim_matrix: np.ndarray,
    col_map: dict,
    top_k: int = 20,
) -> tuple[dict, dict]:
    """Algorithm 4 line 4, per contest.

        lambda* = argmax_lambda  sum_d (R_d - R_{d+1}) P{w_lambda'delta > G^(r_d)}

    The paper picks ONE lambda this way and plays it. We were skipping the step
    entirely -- generating uniformly across a lambda grid whose endpoint came
    from an invented rule -- on the argument that dR re-ranks by marginal
    dollars anyway. That argument controls SELECTION from what we generate; it
    cannot fix a grid that spends half its budget where the objective no longer
    tracks projection.

    Restoring it makes generation CONTEST-AWARE, which is the property that was
    missing. Measured on the 08/24 slate, lambda* rises monotonically with
    field size -- ~0.037 for a 496-entrant contest, ~0.056 for an 8,000 -- so a
    small field is served a higher-projection, less contrarian build and a
    large field a more extreme one, without either being hand-tuned.

    `contest_specs` entries are `(key, thr, steps, k)`: `thr` is the (S, T)
    field order statistic at each paying tier's last rank (what
    `field_order_statistics` returns) and `steps` the matching R_d - R_{d+1}.
    Only the order statistics are needed, not the full sorted field, so this
    costs a few (S, ~20) arrays rather than the multi-GB sorted-field ones.

    Returns `(lambda_star_by_key, ev_curve_by_key)`.
    """
    # One score pass over the union of every lambda's top-K, so the (n, S)
    # array stays at a few hundred rows rather than the whole sample pool.
    tops = {}
    for lam in grid:
        tops[float(lam)] = np.argsort(scorer.score(rows, float(lam), sigma_dG))[::-1][:top_k]
    uniq = sorted({int(j) for v in tops.values() for j in v})
    pos = {j: i for i, j in enumerate(uniq)}
    cols = np.array([[col_map[int(p)] for p in rows[j]] for j in uniq], dtype=np.int64)
    S = sim_matrix.shape[0]
    cand = sim_matrix[:, cols.ravel()].reshape(S, len(uniq), cols.shape[1]).sum(axis=2).T

    lam_star, curves = {}, {}
    for key, thr, steps, _k in contest_specs:
        ev = []
        for lam in grid:
            idx = [pos[int(j)] for j in tops[float(lam)]]
            v = cand[idx]                                   # (top_k, S)
            ev.append(float(sum(
                st * (v > thr[:, d][None, :]).mean()
                for d, st in enumerate(steps) if st != 0.0)))
        curves[key] = ev
        lam_star[key] = float(grid[int(np.argmax(ev))])
    return lam_star, curves


def calibrate_lambda_grid(
    md: dict,
    var_by_pid: dict,
    cov_by_pair: dict,
    sigma_dG: dict,
    n_lambdas: int = 10,
    target_uniques: int = 5,
    max_doublings: int = 8,
    **solve_kw,
) -> list[float]:
    """Solver-based grid calibration. Superseded in the production path by
    `calibrate_lambda_grid_from_pool` (same bracket, no solves); kept because
    it is the reference the solver-only tests exercise."""
    base = solve_lambda(md, 0.0, var_by_pid, cov_by_pair, sigma_dG, n=1, **solve_kw)
    if not base:
        return [0.0]
    base_ids = set(base[0].player_ids)

    mean0 = sum(md["mean_map"][p] for p in base_ids)
    var0 = lineup_variance(base_ids, var_by_pid, cov_by_pair)
    if var0 <= 0 or mean0 <= 0:
        return [0.0]
    seed = mean0 / var0

    def _moves(lam: float) -> bool:
        got = solve_lambda(md, lam, var_by_pid, cov_by_pair, sigma_dG, n=1, **solve_kw)
        return bool(got) and len(base_ids - set(got[0].player_ids)) >= target_uniques

    if _moves(seed):
        hi, lo = seed, seed
        for _ in range(max_doublings):
            lo = lo / 2.0
            if not _moves(lo):
                break
            hi = lo
    else:
        lo, hi = seed, seed
        for _ in range(max_doublings):
            hi = hi * 2.0
            if _moves(hi):
                break
            lo = hi

    if n_lambdas <= 2 or hi <= lo:
        return [0.0, hi]
    return [0.0] + list(np.geomspace(lo, hi, n_lambdas - 1))


def _core_keys(pids):
    """The 10 nine-player subsets of a lineup.

    Two lineups overlap in exactly 9 players iff they share one of these, so
    the near-duplicate test is 10 set lookups rather than a comparison against
    every lineup kept so far. That is what lets it run INSIDE the per-team
    selection loop instead of as a separate O(n^2) pass afterwards.
    """
    s = sorted(int(x) for x in pids)
    return [frozenset(s[:k] + s[k + 1:]) for k in range(len(s))]


def _primary_stack(pids, team_of: dict, pos_of: dict):
    """The team supplying the most BATTERS -- a lineup's stack identity."""
    counts: dict = {}
    for p in pids:
        p = int(p)
        if pos_of.get(p) != "P":
            t = team_of.get(p)
            if t is not None:
                counts[t] = counts.get(t, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else None


def frontier_lineups(
    df: pd.DataFrame,
    var_by_pid: dict,
    cov_by_pair: dict,
    sigma_dG: dict,
    contest_specs: list,
    sim_matrix: np.ndarray,
    col_map: dict,
    n_lambdas: int = 12,
    target_lineups: int = 4_000,
    min_per_team: int = 4,
    sample_n: int = 30_000,
    n_anchors: int = 2,
    n_generations: int = 2,
    mutants_per_parent: int = 4,
    min_uniques: int = 3,
    min_stack: int = 4,
    salary_floor=None,
    min_secondary=None,
    timeout_s: float = 8.0,
    seed: int = 42,
    mutant_workers: int = 0,
    progress_cb: Optional[Callable[[int, int, int], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> tuple[list[Lineup], list[float], dict]:
    """Generate at each contest's OWN line-4 lambda*, ranked by the exact objective.

    Three pieces, in order:

    1. SAMPLE. `CandidateGenerator.generate` round-robins teams with a per-team
       quota, so team coverage is structural. 30k costs ~10s. (An earlier build
       solved one exact anchor per lambda and mutated it; because
       `generate_shape_mutants` preserves the stack profile exactly, 100% of
       733 generated lineups came out on a single team.)

    2. CHOOSE LAMBDA PER CONTEST -- Algorithm 4 line 4. See
       `line4_lambda_star`. Small fields get a higher-projection build, large
       fields a more contrarian one, because that is what the payout maths
       says, not because anyone tuned it.

    3. KEEP THE BEST PER (lambda*, TEAM). `per_team` bounds any one team's
       share so the objective's near-indifference between stacks -- at the top
       of the old grid, every team scored within 3.4% -- cannot become a
       one-team pool.

    Scoring a fixed sample pool is what makes all of this affordable: a lambda
    is a re-rank (0.03s per 20k lineups), not a solve.

    BUDGET IS A TOTAL, NOT A PER-TEAM RATE. Output is
    `n_lambda_star x n_teams x per_team`, and `n_lambda_star` is EMERGENT --
    whatever line 4 returns once it has seen the contests. Setting the per-team
    rate directly therefore made pool size swing with how much the slate's
    contests happened to agree: the same rate of 40 gave 4,320 lineups on a
    6-operating-point slate and 1,920 on a 2-point one. `target_lineups` is
    divided by the actual `n_lambda_star x n_teams` instead, so output lands
    near the target however line 4 resolves, and the memory projection becomes
    stable enough to plan against.

    `min_per_team` floors the result: the per-team cap is what stops one team
    crowding out the rest, and that guarantee has to survive the division.

    `mutant_workers` spreads step 3's mutation over processes (0 = auto,
    1 = serial). Mutation is the largest single block of this function's wall
    clock -- 21.8s of 39.7s on the 08/25 slate -- and it is a pure-Python
    per-parent loop, so processes are the only thing that helps: the same
    slate ran the stage in 22.1s on auto. Output does not depend on the
    count, because `generate_shape_mutants` chunks parents at a fixed size
    and seeds each chunk from its own SeedSequence child.

    Returns `(lineups, lambda_per_lineup, diag)`.
    """
    from src.optimization.mrp.lineup_variance import FrontierScorer

    team_of = {int(r.player_id): r.team for r in df.itertuples()}
    pos_of = {int(r.player_id): r.position for r in df.itertuples()}
    own = (df["ownership"].to_numpy(dtype=np.float64)
           if "ownership" in df.columns else np.ones(len(df)))
    generator = CandidateGenerator(df, own, rng_seed=seed, salary_floor=salary_floor)
    scorer = FrontierScorer(df, var_by_pid, cov_by_pair)

    sampled = generator.generate(n_candidates=int(sample_n), stop_check=stop_check)
    if not sampled:
        return [], [], {"skipped": "sampler returned no lineups"}
    rows = np.array([l.player_ids for l in sampled], dtype=np.int64)

    search = lambda_search_grid(scorer, rows, sigma_dG, n_lambdas=n_lambdas)
    lam_star, curves = line4_lambda_star(
        scorer, rows, sigma_dG, search, contest_specs, sim_matrix, col_map,
    )
    # Distinct operating points only: contests that agree share their lineups
    # instead of generating the same pool twice.
    lambdas_used = sorted(set(lam_star.values())) or [search[0]]
    diag = {
        "lambda_search_lo": float(min(search)), "lambda_search_hi": float(max(search)),
        "lambda_star_by_contest": {str(k): v for k, v in lam_star.items()},
        "n_lambda_star": len(lambdas_used),
        # An optimum at the search boundary means the true one may lie outside.
        # Edge check over the geometric part only: lambda=0 is a deliberate
        # endpoint, not evidence the search range was too narrow.
        "lambda_star_at_edge": bool(
            lambdas_used and [x for x in search if x > 0]
            and (min([x for x in lambdas_used if x > 0], default=float("inf"))
                 <= min(x for x in search if x > 0) * 1.001
                 or max(lambdas_used) >= max(search) * 0.999)),
    }

    # Publish the real operating-point count the moment line 4 has decided it.
    # Everything above -- sampling, scoring, the lambda* search -- is a single
    # opaque phase to a progress bar, and without this the UI has no total to
    # divide by until the FIRST lambda* has already finished generating.
    if progress_cb is not None:
        progress_cb(0, len(lambdas_used), 0)

    md = None
    anchors: dict = {}
    if n_anchors > 0 and lambdas_used:
        md = _build_metadata(df)
        picks = np.unique(np.linspace(0, len(lambdas_used) - 1,
                                      min(n_anchors, len(lambdas_used))).astype(int))
        for gi in picks:
            if stop_check is not None and stop_check():
                break
            got = solve_lambda(md, float(lambdas_used[gi]), var_by_pid, cov_by_pair,
                               sigma_dG, n=1, min_uniques=min_uniques,
                               min_stack=min_stack, salary_floor=salary_floor,
                               min_secondary=min_secondary, timeout_s=timeout_s, seed=seed)
            if got:
                anchors.setdefault(float(lambdas_used[gi]), []).extend(got)

    # Derived only once lambda* is known -- see the budget note above.
    n_teams = len({t for pid, t in team_of.items() if pos_of.get(pid) != "P"})
    per_team = max(int(min_per_team),
                   int(target_lineups) // max(len(lambdas_used) * max(n_teams, 1), 1))
    diag.update(n_teams=n_teams, per_team_derived=per_team,
                target_lineups=int(target_lineups))

    seen: set = set()
    lineups: list[Lineup] = []
    lambdas: list[float] = []

    def _cap(cands: list[Lineup], sc: np.ndarray) -> list[Lineup]:
        """Top `per_team` per team, skipping 9/10 siblings of what is kept."""
        taken: dict = {}
        kept_cores: set = set()
        out: list[Lineup] = []
        for j in np.argsort(sc)[::-1]:
            lu = cands[int(j)]
            t = _primary_stack(lu.player_ids, team_of, pos_of)
            if taken.get(t, 0) >= per_team:
                continue
            ck = _core_keys(lu.player_ids)
            if any(c in kept_cores for c in ck):
                continue
            kept_cores.update(ck)
            taken[t] = taken.get(t, 0) + 1
            out.append(lu)
        return out

    for i, lam in enumerate(lambdas_used):
        if stop_check is not None and stop_check():
            break
        scores = scorer.score(rows, float(lam), sigma_dG)
        taken: dict = {}
        picked: list[Lineup] = []
        cores: set = set()
        for j in np.argsort(scores)[::-1]:
            t = _primary_stack(rows[j], team_of, pos_of)
            if taken.get(t, 0) >= per_team:
                continue
            key = frozenset(int(x) for x in rows[j])
            if key in seen:
                continue
            # A team's per_team slots are a DIVERSITY BUDGET, so spend them on
            # distinct shapes. Measured: 37% of generated lineups shared a
            # 9-player core with another, so a 40-slot team really offered ~25
            # shapes. Skipping a 1-swap sibling here does NOT shrink the pool --
            # the slot refills with the next-best distinct lineup -- which is
            # the whole difference between this and culling afterwards.
            ck = _core_keys(rows[j])
            if any(c in cores for c in ck):
                continue
            seen.add(key)
            cores.update(ck)
            taken[t] = taken.get(t, 0) + 1
            picked.append(sampled[int(j)])

        for a in anchors.get(float(lam), []):
            key = frozenset(a.player_ids)
            if key not in seen:
                seen.add(key)
                picked.append(a)

        # Mutation deepens each team's neighbourhood; the per-team cap is
        # re-applied afterwards, or 2 generations x 4 mutants turns the picks
        # into tens of thousands of lineups and the per-contest (M x S) rank
        # arrays blow the memory budget CLAUDE.md sets.
        if n_generations > 0 and picked:
            parents = list(picked)
            for g in range(int(n_generations)):
                if stop_check is not None and stop_check():
                    break
                kids = generator.generate_shape_mutants(
                    parents, n_per_parent=int(mutants_per_parent), seen=seen,
                    rng_seed=seed + g, salary_floor=salary_floor,
                    n_workers=mutant_workers,
                )
                if not kids:
                    break
                picked.extend(kids)
                parents = kids
            picked = _cap(picked, scorer.score(
                np.array([l.player_ids for l in picked], dtype=np.int64),
                float(lam), sigma_dG))

        lineups.extend(picked)
        lambdas.extend([float(lam)] * len(picked))
        if progress_cb is not None:
            progress_cb(i + 1, len(lambdas_used), len(lineups))

    return lineups, lambdas, diag
