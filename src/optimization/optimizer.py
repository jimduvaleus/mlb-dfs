"""
Basin-Hopping optimizer for DraftKings MLB lineup construction.

Phase 3 implementation: maximizes P(lineup total >= target) over Monte Carlo
simulation results, subject to DK Classic salary and position constraints.
"""
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    meta: PlayerMeta = {}
    for _, row in players_df.iterrows():
        meta[int(row['player_id'])] = {
            'position': row['position'],
            'salary': float(row['salary']),
            'team': row['team'],
            'game': str(row['game']) if has_game else '',
        }
    return meta


def _score_totals(totals: np.ndarray, target: float) -> float:
    return float((totals >= target).mean())


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
    ):
        self.sim_matrix = sim_matrix
        self.player_meta = player_meta
        self.col_map = col_map
        self.players_by_pos = players_by_pos
        self.target = target
        self.temperature = temperature
        self.n_steps = n_steps

    # ---- public entry point ---------------------------------------- #

    def run(self, seed: int) -> Tuple[Lineup, float]:
        rng = np.random.default_rng(seed)
        lineup = self._random_valid_lineup(rng)
        cols = [self.col_map[pid] for pid in lineup.player_ids]
        totals = self.sim_matrix[:, cols].sum(axis=1)
        score = _score_totals(totals, self.target)

        best_lineup = lineup
        best_score = score

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
                chosen = rng.choice(pool, size=count, replace=False).tolist()
                player_ids.extend(chosen)
                used.update(chosen)
            if ok:
                lineup = Lineup(player_ids)
                if lineup.is_valid(self.player_meta):
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
                pos = self.player_meta[old_id]['position']
                pool = [p for p in self.players_by_pos[pos] if p not in used]
                if not pool:
                    ok = False
                    break
                new_id = int(rng.choice(pool))
                used.discard(old_id)
                used.add(new_id)
                new_ids[idx] = new_id

            if ok:
                cand = Lineup(new_ids)
                if cand.is_valid(self.player_meta):
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
            pos = self.player_meta[pid]['position']
            col_out = self.col_map[pid]

            cand_ids = [c for c in self.players_by_pos[pos] if c not in lineup_set]
            if not cand_ids:
                continue

            cand_cols = np.array([self.col_map[c] for c in cand_ids])

            # (n_sims, n_cands) — one delta-updated total vector per candidate
            col_out_scores = self.sim_matrix[:, col_out]
            swapped = (
                totals[:, np.newaxis]
                - col_out_scores[:, np.newaxis]
                + self.sim_matrix[:, cand_cols]
            )
            cand_scores = (swapped >= self.target).mean(axis=0)  # (n_cands,)

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
                if Lineup(test_ids).is_valid(self.player_meta):
                    best_score = float(cand_scores[i])
                    best_new_id = cand_ids[i]
                    best_new_totals = swapped[:, i].copy()
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
    seed, sim_matrix, player_meta, col_map, players_by_pos, target, temperature, n_steps = args
    runner = _ChainRunner(
        sim_matrix=sim_matrix,
        player_meta=player_meta,
        col_map=col_map,
        players_by_pos=players_by_pos,
        target=target,
        temperature=temperature,
        n_steps=n_steps,
    )
    return runner.run(seed)


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
        n_workers: int = 1,
        rng_seed: Optional[int] = None,
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
        self.n_workers = n_workers
        self.rng_seed = rng_seed

        # Pre-group player IDs by position for fast pool queries
        self._players_by_pos: Dict[str, List[int]] = {
            pos: [] for pos in ROSTER_REQUIREMENTS
        }
        for pid, meta in self.player_meta.items():
            pos = meta['position']
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
        )

    def optimize(self) -> Tuple[Lineup, float]:
        """Run all chains and return the best (Lineup, score) found."""
        seeds = [
            (self.rng_seed + i if self.rng_seed is not None else i)
            for i in range(self.n_chains)
        ]

        best_lineup: Optional[Lineup] = None
        best_score = -1.0

        if self.n_workers > 1:
            chain_args = [
                (
                    seed,
                    self.sim_matrix,
                    self.player_meta,
                    self.col_map,
                    self._players_by_pos,
                    self.target,
                    self.temperature,
                    self.n_steps,
                )
                for seed in seeds
            ]
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(_chain_worker, a) for a in chain_args]
                for fut in as_completed(futures):
                    lineup, score = fut.result()
                    logger.debug("Chain completed score=%.4f", score)
                    if score > best_score:
                        best_score = score
                        best_lineup = lineup
        else:
            for seed in seeds:
                lineup, score = self._runner.run(seed)
                logger.debug("Chain seed=%d score=%.4f", seed, score)
                if score > best_score:
                    best_score = score
                    best_lineup = lineup

        return best_lineup, best_score
