"""
Basin-Hopping optimizer for DraftKings MLB lineup construction.

Phase 3 implementation: maximizes P(lineup total >= target) over Monte Carlo
simulation results, subject to DK Classic salary and position constraints.
"""
import logging
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit

from src.optimization.lineup import (
    Lineup,
    PlayerMeta,
    ROSTER_REQUIREMENTS,
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
        meta[int(row['player_id'])] = {
            'position': pos,
            'eligible_positions': elig,
            'salary': float(row['salary']),
            'team': row['team'],
            'game': str(row['game']) if has_game else '',
        }
    return meta


# ------------------------------------------------------------------ #
#  Objective functions                                                  #
# ------------------------------------------------------------------ #

# Supported objective names (used in config and API).
OBJECTIVES = ("p_hit", "expected_surplus")


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
        else:
            self._score_totals = _score_totals_surplus
            self._score_swap_candidates = _score_swap_candidates_surplus

    # ---- public entry point ---------------------------------------- #

    def run(self, seed: int) -> Tuple[Lineup, float]:
        rng = np.random.default_rng(seed)
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

        Batches all candidate evaluations for a given slot into a single matrix
        operation instead of looping over candidates one at a time.
        """
        ids = list(lineup.player_ids)
        lineup_set = set(ids)
        current_score = self._score_totals(totals, self.target)

        # Identify stacked teams: teams with 3+ hitters in the lineup.
        team_counts: Dict[str, int] = {}
        for pid in ids:
            meta = self.player_meta[pid]
            if meta['position'] != 'P':
                team_counts[meta['team']] = team_counts.get(meta['team'], 0) + 1
        stacked_teams = {t for t, c in team_counts.items() if c >= 3}

        for idx in rng.permutation(len(ids)).tolist():
            pid = ids[idx]
            meta = self.player_meta[pid]
            elig = meta['eligible_positions']
            col_out = self.col_map[pid]

            # Build candidate pool
            cand_set: set = set()
            for pos in elig:
                cand_set.update(self.players_by_pos[pos])

            # Stack protection: if this player belongs to a stacked team,
            # only consider same-team replacements.
            player_team = meta['team']
            is_stacked = meta['position'] != 'P' and player_team in stacked_teams
            if is_stacked:
                cand_ids = [
                    c for c in cand_set
                    if c not in lineup_set and self.player_meta[c]['team'] == player_team
                ]
            else:
                cand_ids = [c for c in cand_set if c not in lineup_set]
            if not cand_ids:
                continue

            cand_cols = np.array([self.col_map[c] for c in cand_ids])

            cand_scores = self._score_swap_candidates(
                self.sim_matrix, totals, col_out, cand_cols, self.target
            )

            # Walk candidates best-first; check validity only until the first
            # valid improvement is found (that candidate is necessarily the best).
            order = np.argsort(-cand_scores)
            best_new_id: Optional[int] = None
            best_new_totals: Optional[np.ndarray] = None
            best_score = current_score

            for i in order:
                if cand_scores[i] <= best_score:
                    break  # remaining candidates can only be worse
                test_ids = list(ids)
                test_ids[idx] = cand_ids[i]
                if Lineup(test_ids).is_valid(self.player_meta, salary_floor=self.salary_floor):
                    best_score = float(cand_scores[i])
                    best_new_id = cand_ids[i]
                    best_new_totals = (
                        totals
                        - self.sim_matrix[:, col_out]
                        + self.sim_matrix[:, cand_cols[i]]
                    )
                    break  # first valid candidate in score order is the global best

            if best_new_id is not None:
                old_team = self.player_meta[pid]['team']
                new_team = self.player_meta[best_new_id]['team']
                ids[idx] = best_new_id
                lineup_set.discard(pid)
                lineup_set.add(best_new_id)
                totals = best_new_totals
                current_score = best_score

                # Update team counts and stacked_teams if team changed
                if old_team != new_team and self.player_meta[pid]['position'] != 'P':
                    team_counts[old_team] = team_counts.get(old_team, 1) - 1
                    team_counts[new_team] = team_counts.get(new_team, 0) + 1
                    stacked_teams = {t for t, c in team_counts.items() if c >= 3}

        return Lineup(ids), totals


# Module-level worker function required for ProcessPoolExecutor pickling
def _chain_worker(args: tuple) -> Tuple[Lineup, float]:
    seed, shm_name, shm_shape, shm_dtype, player_meta, col_map, players_by_pos, target, temperature, n_steps, niter_success, salary_floor, objective = args
    shm = SharedMemory(name=shm_name)
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
        )
        return runner.run(seed)
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
        )

    def _run_chains(
        self,
        executor: Optional[ProcessPoolExecutor] = None,
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
            chain_args = [
                (
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
                )
                for seed in seeds
            ]
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
                shm.unlink()
                if owned_executor and executor is not None:
                    executor.shutdown(wait=False)
        else:
            best_so_far = -1.0
            steps_since_improvement = 0
            for seed in seeds:
                lineup, score = self._runner.run(seed)
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
    ) -> Tuple[Lineup, float]:
        """Run all chains and return the best (Lineup, score) found.

        Parameters
        ----------
        executor:
            Optional pre-created ``ProcessPoolExecutor`` to reuse across
            multiple calls. The caller is responsible for shutting it down.
        """
        results = self._run_chains(executor=executor)
        return max(results, key=lambda x: x[1])

    def optimize_top_k(self, k: int, executor: Optional[ProcessPoolExecutor] = None) -> List[Tuple[Lineup, float]]:
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
        results = self._run_chains(executor=executor)
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
