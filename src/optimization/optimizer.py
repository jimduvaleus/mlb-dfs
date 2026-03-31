"""
Basin-Hopping optimizer for DraftKings MLB lineup construction.

Phase 3 implementation: maximizes P(lineup total >= target) over Monte Carlo
simulation results, subject to DK Classic salary and position constraints.
"""
import logging
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Tuple

import os

import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads as _numba_set_num_threads

from src.optimization.lineup import (
    Lineup,
    PlayerMeta,
    ROSTER_REQUIREMENTS,
    SALARY_CAP,
    MAX_HITTERS_PER_TEAM,
    MIN_GAMES,
    SLOTS,
)
from src.simulation.results import SimulationResults

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _build_player_meta(players_df: pd.DataFrame) -> PlayerMeta:
    """Convert a players DataFrame to a fast dict-based lookup."""
    has_game = 'game' in players_df.columns
    has_eligible = 'eligible_positions' in players_df.columns
    meta: PlayerMeta = {}
    for _, row in players_df.iterrows():
        pos = row['position']
        elig = list(row['eligible_positions']) if has_eligible else []
        if not elig:
            elig = [pos]
        team = row['team']
        game_str = str(row['game']) if has_game else ''
        if has_game and '@' in game_str:
            away, home = game_str.split('@', 1)
            opponent = home if team == away else away
        else:
            opponent = ''
        meta[int(row['player_id'])] = {
            'position': pos,
            'eligible_positions': elig,
            'salary': float(row['salary']),
            'team': team,
            'opponent': opponent,
            'game': game_str,
        }
    return meta


# ------------------------------------------------------------------ #
#  Objective functions                                                  #
# ------------------------------------------------------------------ #

# Supported objective names (used in config and API).
OBJECTIVES = ("p_hit", "expected_surplus", "marginal_payout")


def _score_totals_p_hit(totals: np.ndarray, target: float) -> float:
    """P(lineup total >= target)."""
    return float((totals >= target).mean())


def _score_totals_surplus(totals: np.ndarray, target: float) -> float:
    """E[max(lineup total - target, 0)]."""
    return float(np.maximum(totals - target, 0.0).mean())


@njit(cache=True)
def _score_swap_candidates_p_hit(
    sim_matrix: np.ndarray,
    totals: np.ndarray,
    col_out: int,
    cand_cols: np.ndarray,
    target: float,
) -> np.ndarray:
    """Score all swap candidates using P(hit) objective (Numba-accelerated)."""
    n_sims = totals.shape[0]
    n_cands = cand_cols.shape[0]
    cand_scores = np.empty(n_cands, dtype=np.float64)
    for j in range(n_cands):
        count = 0.0
        col_in = cand_cols[j]
        for i in range(n_sims):
            if totals[i] - sim_matrix[i, col_out] + sim_matrix[i, col_in] >= target:
                count += 1.0
        cand_scores[j] = count / n_sims
    return cand_scores


@njit(cache=True)
def _score_swap_candidates_surplus(
    sim_matrix: np.ndarray,
    totals: np.ndarray,
    col_out: int,
    cand_cols: np.ndarray,
    target: float,
) -> np.ndarray:
    """Score all swap candidates using expected surplus objective (Numba-accelerated)."""
    n_sims = totals.shape[0]
    n_cands = cand_cols.shape[0]
    cand_scores = np.empty(n_cands, dtype=np.float64)
    for j in range(n_cands):
        surplus = 0.0
        col_in = cand_cols[j]
        for i in range(n_sims):
            val = totals[i] - sim_matrix[i, col_out] + sim_matrix[i, col_in]
            if val > target:
                surplus += val - target
        cand_scores[j] = surplus / n_sims
    return cand_scores


def _score_totals_payout(
    totals: np.ndarray, cash_line: float, best_scores: np.ndarray, beta: float,
) -> float:
    """E[payout(max(best_scores, lineup_scores))] using power-law payout."""
    effective = np.maximum(totals, best_scores)
    diff = np.maximum(effective - cash_line, 0.0)
    return float(np.mean(diff ** beta))


@njit(parallel=True, cache=True)
def _score_swap_candidates_payout(
    sim_matrix: np.ndarray,
    totals: np.ndarray,
    col_out: int,
    cand_cols: np.ndarray,
    cash_line: float,
    best_scores: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Score all swap candidates using marginal payout objective.

    For each candidate, computes E[P(max(best_scores, new_total))] where
    P(s) = max(0, s - cash_line)^beta.

    Candidates are scored in parallel via Numba prange. In worker processes,
    numba thread count is capped to cpu_count // n_workers to prevent
    over-subscription; in single-process mode all threads are available.
    """
    n_sims = totals.shape[0]
    n_cands = cand_cols.shape[0]
    cand_scores = np.empty(n_cands, dtype=np.float64)
    for j in prange(n_cands):
        total_payout = 0.0
        col_in = cand_cols[j]
        for i in range(n_sims):
            new_total = totals[i] - sim_matrix[i, col_out] + sim_matrix[i, col_in]
            effective = new_total if best_scores[i] <= new_total else best_scores[i]
            diff = effective - cash_line
            if diff > 0.0:
                total_payout += diff ** beta
        cand_scores[j] = total_payout / n_sims
    return cand_scores


# ------------------------------------------------------------------ #
#  Slot assignment helpers (fast path for _local_search)              #
# ------------------------------------------------------------------ #

def _compute_slot_assignment(
    ids: List[int],
    player_meta: PlayerMeta,
) -> Tuple[List[int], List[int]]:
    """Compute a valid player→slot bipartite matching via augmenting-path DFS.

    Returns ``(slot_to_pidx, pidx_to_slot)`` where:
      ``slot_to_pidx[j]`` = index into *ids* matched to SLOTS[j]  (-1 = free)
      ``pidx_to_slot[i]`` = slot index matched to ids[i]          (-1 = unmatched)

    Raises RuntimeError if no full matching exists — only call with lineups
    that have already passed Lineup.is_valid().
    """
    n = len(ids)
    slot_to_pidx: List[int] = [-1] * len(SLOTS)
    pidx_to_slot: List[int] = [-1] * n

    def _elig(pidx: int) -> List[int]:
        meta = player_meta[ids[pidx]]
        ep = meta.get('eligible_positions') or [meta['position']]
        ep_set = set(ep)
        return [j for j, s in enumerate(SLOTS) if s in ep_set]

    def _dfs(pidx: int, visited: set) -> bool:
        for j in _elig(pidx):
            if j not in visited:
                visited.add(j)
                occ = slot_to_pidx[j]
                if occ == -1 or _dfs(occ, visited):
                    slot_to_pidx[j] = pidx
                    pidx_to_slot[pidx] = j
                    return True
        return False

    for i in range(n):
        if not _dfs(i, set()):
            raise RuntimeError(
                f"_compute_slot_assignment: could not match player index {i} "
                f"(id={ids[i]}); lineup appears invalid."
            )
    return slot_to_pidx, pidx_to_slot


def _try_augment_swap(
    pidx: int,
    ids: List[int],
    player_meta: PlayerMeta,
    slot_to_pidx: List[int],
    pidx_to_slot: List[int],
) -> bool:
    """Attempt to place the player at *pidx* into a slot via one augmenting-path DFS.

    **Pre-condition**: the old player's slot has already been freed in the
    passed arrays (``slot_to_pidx[old_slot] = -1``, ``pidx_to_slot[pidx] = -1``),
    and ``ids[pidx]`` holds the *new* player's id.

    Modifies *slot_to_pidx* and *pidx_to_slot* in-place on success only
    (a failed DFS leaves them unchanged). Returns True iff placement succeeded.
    """
    def _elig(p: int) -> List[int]:
        meta = player_meta[ids[p]]
        ep = meta.get('eligible_positions') or [meta['position']]
        ep_set = set(ep)
        return [j for j, s in enumerate(SLOTS) if s in ep_set]

    def _dfs(p: int, visited: set) -> bool:
        for j in _elig(p):
            if j not in visited:
                visited.add(j)
                occ = slot_to_pidx[j]
                if occ == -1 or _dfs(occ, visited):
                    slot_to_pidx[j] = p
                    pidx_to_slot[p] = j
                    return True
        return False

    return _dfs(pidx, set())


# ------------------------------------------------------------------ #
#  Chain runner (self-contained for multiprocessing pickling)          #
# ------------------------------------------------------------------ #

class _ChainRunner:
    """Runs a single Basin-Hopping chain.

    Kept as a standalone class so it can be pickled and sent to worker
    processes via ProcessPoolExecutor.
    """

    def __init__(
        self,
        sim_matrix: np.ndarray,
        player_meta: PlayerMeta,
        col_map: Dict[int, int],
        players_by_pos: Dict[str, List[int]],
        target: float,
        temperature: float,
        n_steps: int,
        niter_success: int = 25,
        salary_floor: Optional[float] = None,
        objective: str = "expected_surplus",
        best_scores: Optional[np.ndarray] = None,
        payout_beta: float = 2.5,
        payout_cash_line: Optional[float] = None,
    ):
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective '{objective}'. Must be one of {OBJECTIVES}")
        self.sim_matrix = sim_matrix
        self.player_meta = player_meta
        self.col_map = col_map
        self.players_by_pos = players_by_pos
        self.target = target
        self.temperature = temperature
        self.n_steps = n_steps
        self.niter_success = niter_success
        self.salary_floor = salary_floor
        self.objective = objective

        # Bind the right scoring functions based on objective
        if objective == "p_hit":
            self._score_totals = _score_totals_p_hit
            self._score_swap_candidates = _score_swap_candidates_p_hit
        elif objective == "marginal_payout":
            if best_scores is None:
                best_scores = np.zeros(sim_matrix.shape[0], dtype=np.float64)
            self._best_scores = best_scores
            self._payout_beta = payout_beta
            self._payout_cash_line = payout_cash_line if payout_cash_line is not None else target
            self._score_totals = lambda totals, tgt: _score_totals_payout(
                totals, self._payout_cash_line, self._best_scores, self._payout_beta
            )
            self._score_swap_candidates = lambda sm, t, co, cc, tgt: _score_swap_candidates_payout(
                sm, t, co, cc, self._payout_cash_line, self._best_scores, self._payout_beta
            )
        else:
            self._score_totals = _score_totals_surplus
            self._score_swap_candidates = _score_swap_candidates_surplus

        # Pre-built parallel arrays for each position pool so _local_search
        # can collect cand_ids and cand_cols in one pass without a second
        # col_map lookup loop.
        self._pos_col_arr: Dict[str, np.ndarray] = {
            pos: np.array([col_map[pid] for pid in pids], dtype=np.int32)
            for pos, pids in players_by_pos.items()
        }

    # ---- public entry point ---------------------------------------- #

    def run(self, seed: int, initial_lineup: Optional[Lineup] = None) -> Tuple[Lineup, float]:
        rng = np.random.default_rng(seed)
        if initial_lineup is not None and all(pid in self.col_map for pid in initial_lineup.player_ids):
            lineup = initial_lineup
        else:
            lineup = self._random_valid_lineup(rng)
        cols = [self.col_map[pid] for pid in lineup.player_ids]
        totals = self.sim_matrix[:, cols].sum(axis=1)
        score = self._score_totals(totals, self.target)

        best_lineup = lineup
        best_score = score
        steps_since_improvement = 0

        for _ in range(self.n_steps):
            # Perturbation
            candidate = self._mutate(lineup, rng)
            cand_cols = [self.col_map[pid] for pid in candidate.player_ids]
            cand_totals = self.sim_matrix[:, cand_cols].sum(axis=1)

            # Local search (greedy hill-climbing)
            candidate, cand_totals = self._local_search(candidate, cand_totals, rng)
            cand_score = self._score_totals(cand_totals, self.target)

            # Metropolis acceptance
            delta = cand_score - score
            if delta >= 0 or rng.random() < np.exp(delta / self.temperature):
                lineup = candidate
                totals = cand_totals
                score = cand_score

            if score > best_score:
                best_score = score
                best_lineup = lineup
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                if steps_since_improvement >= self.niter_success:
                    break

        return best_lineup, best_score

    # ---- lineup construction --------------------------------------- #

    # GPP-oriented stack templates: (primary_stack, secondary_stack).
    # Remaining hitters are filled from other teams.
    STACK_TEMPLATES = [(5, 3), (5, 2), (4, 4), (4, 3)]

    def _random_valid_lineup(
        self, rng: np.random.Generator, max_attempts: int = 1000
    ) -> Lineup:
        """Sample a random valid lineup, ~80% seeded with a stack template."""
        if rng.random() < 0.8:
            lineup = self._stacked_lineup(rng, max_attempts=max_attempts // 2)
            if lineup is not None:
                return lineup
        return self._unstacked_lineup(rng, max_attempts=max_attempts)

    def _unstacked_lineup(
        self, rng: np.random.Generator, max_attempts: int = 1000
    ) -> Lineup:
        """Sample a random valid lineup by filling positions greedily."""
        for _ in range(max_attempts):
            player_ids: List[int] = []
            used: set = set()
            ok = True
            for pos, count in ROSTER_REQUIREMENTS.items():
                pool = [p for p in self.players_by_pos[pos] if p not in used]
                if len(pool) < count:
                    ok = False
                    break
                if self.salary_floor is not None:
                    weights = np.array(
                        [self.player_meta[p]['salary'] for p in pool], dtype=float
                    )
                    weights /= weights.sum()
                    chosen = rng.choice(pool, size=count, replace=False, p=weights).tolist()
                else:
                    chosen = rng.choice(pool, size=count, replace=False).tolist()
                player_ids.extend(chosen)
                used.update(chosen)
            if ok:
                lineup = Lineup(player_ids)
                if lineup.is_valid(self.player_meta, salary_floor=self.salary_floor):
                    return lineup
        raise RuntimeError(
            f"Could not build a valid lineup after {max_attempts} attempts. "
            "Check that enough players of each position are in the simulation results."
        )

    def _stacked_lineup(
        self, rng: np.random.Generator, max_attempts: int = 500
    ) -> Optional[Lineup]:
        """Build a lineup seeded with a GPP-style stack template (e.g. 5-3)."""
        # Group non-pitcher players by team
        team_batters: Dict[str, List[int]] = {}
        for pid, meta in self.player_meta.items():
            if meta['position'] != 'P':
                team_batters.setdefault(meta['team'], []).append(pid)

        # Only consider teams with enough batters
        teams_with_enough: Dict[int, List[str]] = {}
        for team, pids in team_batters.items():
            n = len(pids)
            for size in (3, 4, 5):
                if n >= size:
                    teams_with_enough.setdefault(size, []).append(team)

        template = self.STACK_TEMPLATES[int(rng.integers(len(self.STACK_TEMPLATES)))]
        primary_size, secondary_size = template

        primary_teams = teams_with_enough.get(primary_size, [])
        if not primary_teams:
            return None

        for _ in range(max_attempts):
            # Pick primary stack team
            primary_team = primary_teams[int(rng.integers(len(primary_teams)))]

            # Pick secondary stack team (different team, different game preferred)
            secondary_teams = [
                t for t in teams_with_enough.get(secondary_size, [])
                if t != primary_team
            ]
            if not secondary_teams:
                continue
            secondary_team = secondary_teams[int(rng.integers(len(secondary_teams)))]

            # Select stack batters
            primary_pool = team_batters[primary_team]
            secondary_pool = team_batters[secondary_team]

            primary_chosen = rng.choice(
                primary_pool, size=primary_size, replace=False
            ).tolist()
            secondary_chosen = rng.choice(
                secondary_pool, size=secondary_size, replace=False
            ).tolist()

            stack_ids = set(primary_chosen + secondary_chosen)

            # Fill pitchers
            pitcher_pool = [
                p for p in self.players_by_pos['P'] if p not in stack_ids
            ]
            if len(pitcher_pool) < 2:
                continue
            pitchers = rng.choice(pitcher_pool, size=2, replace=False).tolist()

            # Determine which batter positions are already covered by the stack
            needed: Dict[str, int] = {}
            for pos in ('C', '1B', '2B', '3B', 'SS', 'OF'):
                needed[pos] = ROSTER_REQUIREMENTS[pos]

            for pid in list(stack_ids):
                pos = self.player_meta[pid]['position']
                elig = self.player_meta[pid]['eligible_positions']
                # Try to fill a needed slot
                placed = False
                for e in elig:
                    if e != 'P' and needed.get(e, 0) > 0:
                        needed[e] -= 1
                        placed = True
                        break
                if not placed:
                    # Player doesn't fill a remaining needed slot — check if
                    # any slot still has room (bipartite will sort it out)
                    pass

            # Fill remaining positions from other teams
            used = set(pitchers) | stack_ids
            remaining_ids: List[int] = []
            total_hitters_needed = 8 - primary_size - secondary_size
            if total_hitters_needed < 0:
                continue

            ok = True
            for pos, count in ROSTER_REQUIREMENTS.items():
                if pos == 'P':
                    continue
                fill_count = needed.get(pos, 0)
                if fill_count <= 0:
                    continue
                pool = [
                    p for p in self.players_by_pos[pos]
                    if p not in used
                ]
                if len(pool) < fill_count:
                    ok = False
                    break
                chosen = rng.choice(pool, size=fill_count, replace=False).tolist()
                remaining_ids.extend(chosen)
                used.update(chosen)

            if not ok:
                continue

            all_ids = pitchers + list(stack_ids) + remaining_ids
            # Deduplicate (stack players may overlap with remaining)
            if len(set(all_ids)) != len(all_ids) or len(all_ids) != 10:
                continue

            lineup = Lineup(all_ids)
            if lineup.is_valid(self.player_meta, salary_floor=self.salary_floor):
                return lineup

        return None

    def _mutate(
        self,
        lineup: Lineup,
        rng: np.random.Generator,
        n_swaps: int = 3,
        max_attempts: int = 50,
    ) -> Lineup:
        """Mutate a lineup — 40% chance of stack-swap, 60% random swaps."""
        if rng.random() < 0.4:
            result = self._stack_swap(lineup, rng, max_attempts)
            if result is not None:
                return result
        return self._random_swap(lineup, rng, n_swaps, max_attempts)

    def _random_swap(
        self,
        lineup: Lineup,
        rng: np.random.Generator,
        n_swaps: int = 3,
        max_attempts: int = 50,
    ) -> Lineup:
        """Randomly replace n_swaps players while maintaining validity."""
        for _ in range(max_attempts):
            ids = list(lineup.player_ids)
            used = set(ids)
            indices = rng.choice(len(ids), size=n_swaps, replace=False).tolist()

            new_ids = list(ids)
            ok = True
            for idx in indices:
                old_id = new_ids[idx]
                elig = self.player_meta[old_id]['eligible_positions']
                pool_set: set = set()
                for pos in elig:
                    pool_set.update(self.players_by_pos[pos])
                pool = [p for p in pool_set if p not in used]
                if not pool:
                    ok = False
                    break
                new_id = int(rng.choice(pool))
                used.discard(old_id)
                used.add(new_id)
                new_ids[idx] = new_id

            if ok:
                cand = Lineup(new_ids)
                if cand.is_valid(self.player_meta, salary_floor=self.salary_floor):
                    return cand

        return lineup  # fallback: return original if no valid mutation found

    def _stack_swap(
        self,
        lineup: Lineup,
        rng: np.random.Generator,
        max_attempts: int = 50,
    ) -> Optional[Lineup]:
        """Replace a mini-stack (2-3 hitters from one team) with same-sized
        group from a different team, preserving position coverage."""
        ids = list(lineup.player_ids)
        # Find teams with 2+ hitters in the lineup
        team_indices: Dict[str, List[int]] = {}
        for i, pid in enumerate(ids):
            meta = self.player_meta[pid]
            if meta['position'] != 'P':
                team_indices.setdefault(meta['team'], []).append(i)

        swappable = [(t, idxs) for t, idxs in team_indices.items() if len(idxs) >= 2]
        if not swappable:
            return None

        for _ in range(max_attempts):
            # Pick a team to swap out
            team_out, out_indices = swappable[int(rng.integers(len(swappable)))]
            swap_size = min(int(rng.integers(2, min(len(out_indices), 3) + 1)), len(out_indices))
            chosen_indices = rng.choice(out_indices, size=swap_size, replace=False).tolist()

            # Pick a different team to swap in
            team_batters: Dict[str, List[int]] = {}
            used = set(ids)
            for pid, meta in self.player_meta.items():
                if meta['position'] != 'P' and meta['team'] != team_out and pid not in used:
                    team_batters.setdefault(meta['team'], []).append(pid)

            eligible_teams = [
                (t, pids) for t, pids in team_batters.items()
                if len(pids) >= swap_size
            ]
            if not eligible_teams:
                continue

            team_in, in_pool = eligible_teams[int(rng.integers(len(eligible_teams)))]
            new_players = rng.choice(in_pool, size=swap_size, replace=False).tolist()

            new_ids = list(ids)
            for i, idx in enumerate(chosen_indices):
                new_ids[idx] = new_players[i]

            if len(set(new_ids)) != len(new_ids):
                continue

            cand = Lineup(new_ids)
            if cand.is_valid(self.player_meta, salary_floor=self.salary_floor):
                return cand

        return None

    def _local_search(
        self,
        lineup: Lineup,
        totals: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[Lineup, np.ndarray]:
        """One greedy pass: for each slot, accept the best single-player swap.

        Stack-aware: players who are part of a stack (3+ teammates in the
        lineup) can only be swapped for same-team replacements, preserving the
        correlation structure that the objective function rewards.

        Uses an incremental fast-path validity check instead of rebuilding the
        full bipartite match from scratch on every candidate:
          1. O(1) salary check (eliminates most expensive replacements early)
          2. O(1) team hitter-cap check
          3. O(1) pitcher-batter conflict check
          4. O(1) min-games check (rarely triggered)
          5. Single augmenting-path DFS for position eligibility — O(n) worst
             case but O(1) in the common same-position swap, and always correct
             for multi-position eligible players.
        """
        ids = list(lineup.player_ids)
        lineup_set = set(ids)
        current_score = self._score_totals(totals, self.target)

        # ── incremental validity state ──────────────────────────────── #
        slot_to_pidx, pidx_to_slot = _compute_slot_assignment(ids, self.player_meta)

        running_salary: float = sum(self.player_meta[pid]['salary'] for pid in ids)

        team_counts: Dict[str, int] = {}
        for pid in ids:
            if self.player_meta[pid]['position'] != 'P':
                t = self.player_meta[pid]['team']
                team_counts[t] = team_counts.get(t, 0) + 1
        stacked_teams = {t for t, c in team_counts.items() if c >= 3}

        game_counts: Dict[str, int] = {}
        for pid in ids:
            g = self.player_meta[pid].get('game', '')
            if g:
                game_counts[g] = game_counts.get(g, 0) + 1

        pitcher_opponents: set = {
            self.player_meta[pid]['opponent']
            for pid in ids
            if self.player_meta[pid]['position'] == 'P'
            and self.player_meta[pid].get('opponent')
        }
        batter_teams: set = {
            self.player_meta[pid]['team']
            for pid in ids
            if self.player_meta[pid]['position'] != 'P'
        }

        # ── main greedy loop ─────────────────────────────────────────── #
        for idx in rng.permutation(len(ids)).tolist():
            pid = ids[idx]
            meta = self.player_meta[pid]
            elig = meta['eligible_positions']
            col_out = self.col_map[pid]
            old_salary = meta['salary']
            old_team = meta['team']
            old_game = meta.get('game', '')
            old_is_pitcher = meta['position'] == 'P'

            # Salary window for this slot.
            sal_max = SALARY_CAP - running_salary + old_salary
            sal_min = (
                (self.salary_floor - running_salary + old_salary)
                if self.salary_floor is not None else 0.0
            )

            # Build candidate pool in one pass using the pre-built parallel
            # arrays, collecting ids and col indices together so no second
            # col_map lookup loop is needed.
            is_stacked = not old_is_pitcher and old_team in stacked_teams
            cand_ids: List[int] = []
            cand_col_list: List[int] = []
            seen: set = set()
            for pos in elig:
                for c, col in zip(self.players_by_pos[pos], self._pos_col_arr[pos]):
                    if c in seen or c in lineup_set:
                        continue
                    sal = self.player_meta[c]['salary']
                    if sal < sal_min or sal > sal_max:
                        continue
                    if is_stacked and self.player_meta[c]['team'] != old_team:
                        continue
                    seen.add(c)
                    cand_ids.append(c)
                    cand_col_list.append(int(col))
            if not cand_ids:
                continue

            cand_cols = np.array(cand_col_list, dtype=np.int32)
            cand_scores = self._score_swap_candidates(
                self.sim_matrix, totals, col_out, cand_cols, self.target
            )

            order = np.argsort(-cand_scores)
            best_new_id: Optional[int] = None
            best_new_totals: Optional[np.ndarray] = None
            best_score = current_score

            for i in order:
                if cand_scores[i] <= best_score:
                    break
                new_id = cand_ids[i]
                new_meta = self.player_meta[new_id]
                new_salary = new_meta['salary']
                new_team = new_meta['team']
                new_game = new_meta.get('game', '')
                new_is_pitcher = new_meta['position'] == 'P'

                # 1. Team hitter cap: only check when new_team gains a hitter
                if not new_is_pitcher and (old_is_pitcher or old_team != new_team):
                    if team_counts.get(new_team, 0) + 1 > MAX_HITTERS_PER_TEAM:
                        continue

                # 3. Pitcher-batter conflict
                if new_is_pitcher:
                    new_opp = new_meta.get('opponent', '')
                    if new_opp:
                        # Swapping out a hitter may shrink batter_teams.
                        if not old_is_pitcher and team_counts.get(old_team, 0) == 1:
                            eff_batter_teams = batter_teams - {old_team}
                        else:
                            eff_batter_teams = batter_teams
                        if new_opp in eff_batter_teams:
                            continue
                elif old_is_pitcher:
                    # Pitcher → hitter: remaining pitcher opponents shrink by old opp.
                    old_opp = meta.get('opponent', '')
                    eff_pitcher_opps = pitcher_opponents - ({old_opp} if old_opp else set())
                    if new_team in eff_pitcher_opps:
                        continue
                else:
                    # Hitter → hitter: pitcher opponents unchanged.
                    if new_team in pitcher_opponents:
                        continue

                # 4. Min games (only when removing a player unique to their game)
                if old_game and old_game != new_game and game_counts.get(old_game, 0) == 1:
                    n_games_after = sum(
                        1 for g, c in game_counts.items() if g != old_game and c > 0
                    )
                    if new_game and new_game not in game_counts:
                        n_games_after += 1
                    if n_games_after < MIN_GAMES:
                        continue

                # 5. Position eligibility: incremental augmenting-path on copies.
                #    A failed attempt leaves the originals unchanged.
                st_copy = list(slot_to_pidx)
                ps_copy = list(pidx_to_slot)
                old_slot = ps_copy[idx]
                st_copy[old_slot] = -1
                ps_copy[idx] = -1

                ids_test = list(ids)
                ids_test[idx] = new_id

                if not _try_augment_swap(idx, ids_test, self.player_meta, st_copy, ps_copy):
                    continue

                # Valid swap — commit
                best_score = float(cand_scores[i])
                best_new_id = new_id
                best_new_totals = (
                    totals
                    - self.sim_matrix[:, col_out]
                    + self.sim_matrix[:, self.col_map[new_id]]
                )
                slot_to_pidx[:] = st_copy
                pidx_to_slot[:] = ps_copy
                break

            if best_new_id is not None:
                new_meta = self.player_meta[best_new_id]
                new_team = new_meta['team']
                new_game = new_meta.get('game', '')
                new_is_pitcher = new_meta['position'] == 'P'

                ids[idx] = best_new_id
                lineup_set.discard(pid)
                lineup_set.add(best_new_id)
                totals = best_new_totals
                current_score = best_score

                # Salary
                running_salary += new_meta['salary'] - old_salary

                # Team counts and stacked_teams
                if old_is_pitcher != new_is_pitcher or old_team != new_team:
                    if not old_is_pitcher:
                        team_counts[old_team] = team_counts.get(old_team, 1) - 1
                    if not new_is_pitcher:
                        team_counts[new_team] = team_counts.get(new_team, 0) + 1
                    stacked_teams = {t for t, c in team_counts.items() if c >= 3}

                # Game counts
                if old_game:
                    game_counts[old_game] = game_counts.get(old_game, 1) - 1
                if new_game:
                    game_counts[new_game] = game_counts.get(new_game, 0) + 1

                # Pitcher opponents and batter teams
                if old_is_pitcher:
                    old_opp = meta.get('opponent', '')
                    if old_opp:
                        pitcher_opponents.discard(old_opp)
                else:
                    if team_counts.get(old_team, 0) == 0:
                        batter_teams.discard(old_team)
                if new_is_pitcher:
                    new_opp = new_meta.get('opponent', '')
                    if new_opp:
                        pitcher_opponents.add(new_opp)
                else:
                    batter_teams.add(new_team)

        return Lineup(ids), totals


# Module-level worker function required for ProcessPoolExecutor pickling
def _chain_worker(args: tuple) -> Tuple[Lineup, float]:
    seed, shm_name, shm_shape, shm_dtype, player_meta, col_map, players_by_pos, target, temperature, n_steps, niter_success, salary_floor, objective, best_scores, payout_beta, payout_cash_line, n_workers, initial_lineup = args
    # Cap Numba intra-op threads so that n_workers processes don't collectively
    # over-subscribe the CPU.  Single-process mode (n_workers=1) leaves Numba
    # free to use all available cores for prange parallelism.
    cpu_count = os.cpu_count() or 1
    _numba_set_num_threads(max(1, cpu_count // n_workers))
    shm = SharedMemory(name=shm_name)
    # Do NOT unregister here: with fork start method all workers share the
    # parent's resource tracker subprocess, so the first unregister removes
    # the name from the set and every subsequent call gets KeyError.
    # The main process unlinks (and thus unregisters) via shm.unlink() below.
    sim_matrix = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
    try:
        runner = _ChainRunner(
            sim_matrix=sim_matrix,
            player_meta=player_meta,
            col_map=col_map,
            players_by_pos=players_by_pos,
            target=target,
            temperature=temperature,
            n_steps=n_steps,
            niter_success=niter_success,
            salary_floor=salary_floor,
            objective=objective,
            best_scores=best_scores,
            payout_beta=payout_beta,
            payout_cash_line=payout_cash_line,
        )
        return runner.run(seed, initial_lineup=initial_lineup)
    finally:
        shm.close()  # detach; do NOT unlink from worker


# ------------------------------------------------------------------ #
#  Public optimizer                                                    #
# ------------------------------------------------------------------ #

class BasinHoppingOptimizer:
    """
    Basin-Hopping optimizer for DFS lineup construction.

    Maximizes P(lineup total score >= target) by running m independent
    chains of perturb → local-search → Metropolis-accept.

    Args:
        sim_results:  SimulationResults from the Monte Carlo engine.
        players_df:   DataFrame with columns [player_id, position, salary, team]
                      and optionally [game].  Only players present in
                      sim_results are used.
        target:       Score threshold for the objective function.
        n_chains:     Number of independent Basin-Hopping chains.
        temperature:  Metropolis temperature T (controls acceptance of worse).
        n_steps:      Number of perturbation steps per chain.
        n_workers:    Worker processes for parallel chains (1 = sequential).
        rng_seed:     Base seed for reproducibility; chain i uses seed+i.
    """

    def __init__(
        self,
        sim_results: SimulationResults,
        players_df: pd.DataFrame,
        target: float,
        n_chains: int = 250,
        temperature: float = 0.1,
        n_steps: int = 100,
        niter_success: int = 25,
        n_workers: int = 1,
        rng_seed: Optional[int] = None,
        early_stopping_window: int = 25,
        early_stopping_threshold: float = 0.001,
        salary_floor: Optional[float] = None,
        objective: str = "expected_surplus",
        best_scores: Optional[np.ndarray] = None,
        payout_beta: float = 2.5,
        payout_cash_line: Optional[float] = None,
    ):
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective '{objective}'. Must be one of {OBJECTIVES}")

        # Restrict to players that appear in the simulation results
        sim_ids = set(sim_results.player_ids)
        meta_df = players_df[players_df['player_id'].isin(sim_ids)].copy()

        self.sim_matrix = sim_results.results_matrix
        self.col_map: Dict[int, int] = {
            pid: i for i, pid in enumerate(sim_results.player_ids)
        }
        self.player_meta: PlayerMeta = _build_player_meta(meta_df)
        self.target = target
        self.n_chains = n_chains
        self.temperature = temperature
        self.n_steps = n_steps
        self.niter_success = niter_success
        self.n_workers = n_workers
        self.rng_seed = rng_seed
        self.early_stopping_window = early_stopping_window
        self.early_stopping_threshold = early_stopping_threshold
        self.salary_floor = salary_floor
        self.objective = objective
        self.best_scores = best_scores
        self.payout_beta = payout_beta
        self.payout_cash_line = payout_cash_line

        # Pre-group player IDs by position for fast pool queries
        self._players_by_pos: Dict[str, List[int]] = {
            pos: [] for pos in ROSTER_REQUIREMENTS
        }
        for pid, meta in self.player_meta.items():
            for pos in meta['eligible_positions']:
                if pos in self._players_by_pos:
                    self._players_by_pos[pos].append(pid)

        self._runner = _ChainRunner(
            sim_matrix=self.sim_matrix,
            player_meta=self.player_meta,
            col_map=self.col_map,
            players_by_pos=self._players_by_pos,
            target=self.target,
            temperature=self.temperature,
            n_steps=self.n_steps,
            niter_success=self.niter_success,
            salary_floor=self.salary_floor,
            objective=self.objective,
            best_scores=self.best_scores,
            payout_beta=self.payout_beta,
            payout_cash_line=self.payout_cash_line,
        )

    def _run_chains(
        self,
        executor: Optional[ProcessPoolExecutor] = None,
        seed_lineups: Optional[List[Lineup]] = None,
    ) -> List[Tuple[Lineup, float]]:
        """Run all chains and return a (Lineup, score) pair for each.

        Applies cross-chain early stopping: if the global best score has not
        improved by at least ``early_stopping_threshold`` within the last
        ``early_stopping_window`` completed chains, remaining chains are
        skipped (sequential) or cancelled (parallel).

        Parameters
        ----------
        executor:
            An already-running ``ProcessPoolExecutor`` to reuse. When provided
            the executor is *not* shut down by this method — the caller owns its
            lifecycle.  If ``None`` and ``n_workers > 1``, a new executor is
            created and destroyed for this call.
        """
        seeds = [
            (self.rng_seed + i if self.rng_seed is not None else i)
            for i in range(self.n_chains)
        ]
        results: List[Tuple[Lineup, float]] = []
        window = self.early_stopping_window
        threshold = self.early_stopping_threshold

        def _should_stop_early(best_so_far: float, scores_since_last_improvement: int) -> bool:
            return scores_since_last_improvement >= window and best_so_far > 0.0

        if self.n_workers > 1:
            mat = np.ascontiguousarray(self.sim_matrix)
            shm = SharedMemory(create=True, size=mat.nbytes)
            shm_array = np.ndarray(mat.shape, dtype=mat.dtype, buffer=shm.buf)
            shm_array[:] = mat
            chain_args = []
            for i, seed in enumerate(seeds):
                initial = seed_lineups[i] if (seed_lineups and i < len(seed_lineups)) else None
                chain_args.append((
                    seed,
                    shm.name,
                    mat.shape,
                    mat.dtype,
                    self.player_meta,
                    self.col_map,
                    self._players_by_pos,
                    self.target,
                    self.temperature,
                    self.n_steps,
                    self.niter_success,
                    self.salary_floor,
                    self.objective,
                    self.best_scores,
                    self.payout_beta,
                    self.payout_cash_line,
                    self.n_workers,
                    initial,
                ))
            owned_executor = executor is None
            try:
                if owned_executor:
                    executor = ProcessPoolExecutor(max_workers=self.n_workers)
                futures: List[Future] = [executor.submit(_chain_worker, a) for a in chain_args]
                best_so_far = -1.0
                steps_since_improvement = 0
                for fut in as_completed(futures):
                    lineup, score = fut.result()
                    logger.debug("Chain completed score=%.4f", score)
                    results.append((lineup, score))
                    if score >= best_so_far + threshold:
                        best_so_far = score
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1
                    if _should_stop_early(best_so_far, steps_since_improvement):
                        logger.debug(
                            "Early stopping: best=%.4f unchanged for %d chains",
                            best_so_far, steps_since_improvement,
                        )
                        for f in futures:
                            f.cancel()
                        break
            finally:
                shm.close()
                shm.unlink()  # internally calls resource_tracker.unregister once
                if owned_executor and executor is not None:
                    executor.shutdown(wait=False)
        else:
            best_so_far = -1.0
            steps_since_improvement = 0
            for i, seed in enumerate(seeds):
                initial = seed_lineups[i] if (seed_lineups and i < len(seed_lineups)) else None
                lineup, score = self._runner.run(seed, initial_lineup=initial)
                logger.debug("Chain seed=%d score=%.4f", seed, score)
                results.append((lineup, score))
                if score >= best_so_far + threshold:
                    best_so_far = score
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
                if _should_stop_early(best_so_far, steps_since_improvement):
                    logger.debug(
                        "Early stopping: best=%.4f unchanged for %d chains",
                        best_so_far, steps_since_improvement,
                    )
                    break

        return results

    def optimize(
        self,
        executor: Optional[ProcessPoolExecutor] = None,
        seed_lineups: Optional[List[Lineup]] = None,
    ) -> Tuple[Lineup, float]:
        """Run all chains and return the best (Lineup, score) found.

        Parameters
        ----------
        executor:
            Optional pre-created ``ProcessPoolExecutor`` to reuse across
            multiple calls. The caller is responsible for shutting it down.
        seed_lineups:
            Optional list of ``Lineup`` objects to use as warm-start initial
            points for the first ``len(seed_lineups)`` chains. Chains beyond
            that index start from a random valid lineup as usual.
        """
        results = self._run_chains(executor=executor, seed_lineups=seed_lineups)
        return max(results, key=lambda x: x[1])

    def optimize_top_k(self, k: int, executor: Optional[ProcessPoolExecutor] = None, seed_lineups: Optional[List[Lineup]] = None) -> List[Tuple[Lineup, float]]:
        """Run all chains and return the top-k distinct (Lineup, score) pairs.

        Lineups are deduplicated by player set; the highest-scoring instance of
        each distinct lineup is kept.  If fewer than ``k`` distinct lineups are
        found across all chains, all available distinct lineups are returned.

        Parameters
        ----------
        k : int
            Maximum number of distinct lineups to return.

        Returns
        -------
        List[Tuple[Lineup, float]]
            Up to ``k`` (Lineup, score) pairs sorted best-first.
        """
        results = self._run_chains(executor=executor, seed_lineups=seed_lineups)
        results.sort(key=lambda x: -x[1])
        seen: set = set()
        top_k: List[Tuple[Lineup, float]] = []
        for lineup, score in results:
            key = frozenset(lineup.player_ids)
            if key not in seen:
                seen.add(key)
                top_k.append((lineup, score))
            if len(top_k) == k:
                break
        return top_k
