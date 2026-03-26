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


def _score_totals(totals: np.ndarray, target: float) -> float:
    return float((totals >= target).mean())


@njit(cache=True)
def _score_swap_candidates(
    sim_matrix: np.ndarray,
    totals: np.ndarray,
    col_out: int,
    cand_cols: np.ndarray,
    target: float,
) -> np.ndarray:
    """Score all swap candidates for one slot without allocating a swapped matrix.

    Replaces:
        swapped = totals[:,None] - col_out_scores[:,None] + sim_matrix[:,cand_cols]
        cand_scores = (swapped >= target).mean(axis=0)

    The inner loop over n_sims is compiled to native SIMD code, and no
    (n_sims × n_cands) intermediate array is allocated.
    """
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
    ):
        self.sim_matrix = sim_matrix
        self.player_meta = player_meta
        self.col_map = col_map
        self.players_by_pos = players_by_pos
        self.target = target
        self.temperature = temperature
        self.n_steps = n_steps
        self.niter_success = niter_success
        self.salary_floor = salary_floor

    # ---- public entry point ---------------------------------------- #

    def run(self, seed: int) -> Tuple[Lineup, float]:
        rng = np.random.default_rng(seed)
        lineup = self._random_valid_lineup(rng)
        cols = [self.col_map[pid] for pid in lineup.player_ids]
        totals = self.sim_matrix[:, cols].sum(axis=1)
        score = _score_totals(totals, self.target)

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
            cand_score = _score_totals(cand_totals, self.target)

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

    def _random_valid_lineup(
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

    def _mutate(
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

    def _local_search(
        self,
        lineup: Lineup,
        totals: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[Lineup, np.ndarray]:
        """One greedy pass: for each slot, accept the best single-player swap.

        Batches all candidate evaluations for a given slot into a single matrix
        operation instead of looping over candidates one at a time:

            swapped[:, i] = totals - sim_matrix[:, col_out] + sim_matrix[:, col_in_i]

        Candidates are then ranked by score; validity is checked in score order
        so we stop as soon as we find the best valid swap.
        """
        ids = list(lineup.player_ids)
        lineup_set = set(ids)
        current_score = _score_totals(totals, self.target)

        for idx in rng.permutation(len(ids)).tolist():
            pid = ids[idx]
            elig = self.player_meta[pid]['eligible_positions']
            col_out = self.col_map[pid]

            cand_set: set = set()
            for pos in elig:
                cand_set.update(self.players_by_pos[pos])
            cand_ids = [c for c in cand_set if c not in lineup_set]
            if not cand_ids:
                continue

            cand_cols = np.array([self.col_map[c] for c in cand_ids])

            cand_scores = _score_swap_candidates(
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
                ids[idx] = best_new_id
                lineup_set.discard(pid)
                lineup_set.add(best_new_id)
                totals = best_new_totals
                current_score = best_score

        return Lineup(ids), totals


# Module-level worker function required for ProcessPoolExecutor pickling
def _chain_worker(args: tuple) -> Tuple[Lineup, float]:
    seed, shm_name, shm_shape, shm_dtype, player_meta, col_map, players_by_pos, target, temperature, n_steps, niter_success, salary_floor = args
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
    ):
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
