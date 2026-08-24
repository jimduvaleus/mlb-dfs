"""Sigma_delta, in the only form the frontier solver actually needs.

Haugh & Singal's line-2 objective carries `w'Sigma_delta w`, the variance of
our own lineup's score. Written as a P x P matrix that is ~400x400 of mostly
zeros; written as the pairs a solver can consume it is ~1,000 numbers.

WHY PAIRS, NOT A MATRIX. The quadratic enters the ILP through McCormick
product variables, one per off-diagonal pair with a non-zero coefficient. The
diagonal never needs one: `y^2 = y` for a binary, so `Sigma_pp` folds into the
linear coefficient on `y_p`. So the solver wants exactly `(variance_by_pid,
covariance_by_pair)` and never wants a matrix.

WHICH PAIRS. Only within-COPULA-UNIT pairs carry mass. CLAUDE.md's unit is the
9 batters of one team plus their opposing pitcher, and the sim-time overlay
gives the two units of a game independent factors, so cross-unit covariance is
~0 by construction rather than by approximation. On a 12-game slate that is
~1,000 pairs instead of the ~80,000 a dense linearisation would carry -- the
difference between a solve and a hang.

ESTIMATED FROM THE SIM, not from a parametric Sigma: `sim_matrix` already
carries the copula, the env overlay and the marginals, so the covariance it
implies is the one every other part of MRP is denominated in.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def unit_player_groups(players_df) -> dict[tuple[str, str], list[int]]:
    """`(team, opponent) -> [player_id]` for each copula unit on the slate.

    The unit of `(T, O)` is T's batters plus O's pitcher(s) -- i.e. the
    pitcher those batters face. That is CLAUDE.md's 10-player unit, and it is
    the grouping the empirical copula's rows were built on, so it is also the
    grouping across which the sampled dependence is non-zero.

    Pitchers are matched by `(team == O) & (opponent == T)` rather than by
    position alone, so a slate carrying more than one listed pitcher per team
    puts each in the unit it actually faces.
    """
    is_p = players_df["position"].to_numpy() == "P"
    teams = players_df["team"].to_numpy()
    opps = players_df["opponent"].to_numpy()
    pids = players_df["player_id"].to_numpy(dtype=np.int64)

    groups: dict[tuple[str, str], list[int]] = {}
    for t, o in {(str(a), str(b)) for a, b in zip(teams, opps) if a and b}:
        batters = pids[(~is_p) & (teams == t)]
        pitchers = pids[is_p & (teams == o) & (opps == t)]
        members = [int(p) for p in batters] + [int(p) for p in pitchers]
        if len(members) > 1:
            groups[(t, o)] = members
    return groups


def unit_covariance_pairs(
    sim_matrix: np.ndarray,
    sim_player_ids,
    players_df,
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """`(variance_by_pid, covariance_by_pair)` over within-unit pairs.

    `sim_matrix` is `(n_sims, n_players)` and `sim_player_ids` gives its column
    order -- the `SimulationResults` contract.

    Pair keys are `(min_pid, max_pid)` so a lookup never has to try both
    orders. Only pairs inside one unit appear; everything else is treated as
    exactly zero, which is what the overlay makes it.
    """
    col_of = {int(p): i for i, p in enumerate(sim_player_ids)}
    X = np.asarray(sim_matrix)

    var_by_pid: dict[int, float] = {}
    cov_by_pair: dict[tuple[int, int], float] = {}

    for members in unit_player_groups(players_df).values():
        present = [p for p in members if p in col_of]
        if len(present) < 2:
            continue
        cols = [col_of[p] for p in present]
        # (u, u) block only -- u is ~10, so this is a trivially small copy
        # even though the source array is (n_sims x n_players).
        block = np.cov(X[:, cols], rowvar=False)
        block = np.atleast_2d(np.asarray(block, dtype=np.float64))
        for a in range(len(present)):
            pa = present[a]
            # A player can sit in only one unit, so this never overwrites a
            # different unit's estimate of the same variance.
            var_by_pid[pa] = float(block[a, a])
            for b in range(a + 1, len(present)):
                pb = present[b]
                key = (pa, pb) if pa < pb else (pb, pa)
                cov_by_pair[key] = float(block[a, b])

    # Players with no unit (no opponent, lone pitcher) still need a variance:
    # they contribute their own diagonal to w'Sigma w even with no partner.
    for pid, col in col_of.items():
        if pid not in var_by_pid:
            var_by_pid[pid] = float(np.var(X[:, col]))

    return var_by_pid, cov_by_pair


def lineup_variance(
    player_ids,
    var_by_pid: dict[int, float],
    cov_by_pair: dict[tuple[int, int], float],
) -> float:
    """`w'Sigma w` for one lineup, from the pair form.

    The reference implementation for the solver's objective: if the McCormick
    linearisation is correct, the solver's reported quadratic term equals this
    exactly (`tests/test_mrp_frontier_qp.py`).
    """
    pids = [int(p) for p in player_ids]
    total = sum(var_by_pid.get(p, 0.0) for p in pids)
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = pids[i], pids[j]
            key = (a, b) if a < b else (b, a)
            total += 2.0 * cov_by_pair.get(key, 0.0)
    return float(total)


def margin_variance(cand_scores: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """(M,) `Var_s(w'delta - G)` -- the quantity line 2 actually maximises.

    Equation (14) gives `w'Sigma w + Var(G) - 2 w'sigma_dG`, so line 2's
    lambda-term is this minus a w-independent constant. Diagnostics only: the
    solver works from the pair form above, since it cannot evaluate a variance
    over sim worlds inside an ILP.

    `cand_scores` is (M, S); `threshold` is (S,).
    """
    a = np.asarray(cand_scores, dtype=np.float64)
    g = np.asarray(threshold, dtype=np.float64).reshape(1, -1)
    return np.var(a - g, axis=1)


class FrontierScorer:
    """Vectorised exact line-2 objective over many lineups at once.

    `w'mu + lambda(w'Sigma w - 2 w'sigma_dG)` for an (M, 10) array of player
    ids, in one pass. Measured at 0.02s for 20,000 lineups, which is what makes
    generate-and-rank viable: the solver is needed to find the frontier's
    anchor points, but ranking everything around them is free.

    The covariance is densified to (P, P) here rather than kept as pairs. That
    is the opposite of `unit_covariance_pairs`' choice and deliberately so: a
    SOLVER wants pairs (one product variable each), a SCORER wants a matrix
    (one fancy-index per slot pair). P is the restricted starter set, ~180
    players, so the matrix is ~260 KB -- not the ~400x400 of the full slate,
    and nothing like the memory rule's concern.
    """

    def __init__(self, players_df, var_by_pid: dict, cov_by_pair: dict):
        self.pids = players_df["player_id"].to_numpy(dtype=np.int64)
        self.idx = {int(p): i for i, p in enumerate(self.pids)}
        n = len(self.pids)
        self.mu = players_df["mean"].to_numpy(dtype=np.float64)
        self.var = np.array([float(var_by_pid.get(int(p), 0.0)) for p in self.pids])
        self.cov = np.zeros((n, n), dtype=np.float64)
        for (a, b), v in cov_by_pair.items():
            ia, ib = self.idx.get(a), self.idx.get(b)
            if ia is not None and ib is not None:
                self.cov[ia, ib] = v
                self.cov[ib, ia] = v

    def columns(self, rows) -> np.ndarray:
        """(M, 10) player ids -> (M, 10) column indices, -1 where unknown."""
        arr = np.asarray(rows, dtype=np.int64)
        out = np.full(arr.shape, -1, dtype=np.int64)
        for pid, i in self.idx.items():
            out[arr == pid] = i
        return out

    def score(self, rows, lam: float, sigma_dG: Optional[dict] = None) -> np.ndarray:
        """(M,) objective value. Rows containing an unknown player score -inf."""
        cols = self.columns(rows)
        bad = (cols < 0).any(axis=1)
        safe = np.where(cols < 0, 0, cols)

        sg = np.zeros(len(self.pids))
        if sigma_dG:
            sg = np.array([float(sigma_dG.get(int(p), 0.0)) for p in self.pids])

        lin = (self.mu + lam * self.var - 2.0 * lam * sg)[safe].sum(axis=1)
        quad = np.zeros(len(safe), dtype=np.float64)
        k = safe.shape[1]
        for i in range(k):
            for j in range(i + 1, k):
                quad += self.cov[safe[:, i], safe[:, j]]
        out = lin + 2.0 * lam * quad
        return np.where(bad, -np.inf, out)
